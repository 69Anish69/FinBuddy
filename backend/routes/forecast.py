import os
from fastapi import APIRouter, Request, Response
import modules.data_store as data_store
from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import traceback

import re
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")

def no_emoji(s: str) -> str:
    return EMOJI_RE.sub("", s)

router = APIRouter()

# Path to your logo used as a watermark in PDFs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "..", "static", "logo.png")  # ensure backend/static/logo.png exists

# ---------------------------
# Load last uploaded DataFrame
# ---------------------------
def load_dataframe():
    df = getattr(data_store, 'stored_df', None)

    if df is not None and not df.empty:
        print("📂 Loaded from memory")
        print("🧠 Memory preview:\n", df.head(3))
        return df.copy(), "memory"

    # fallback if backend restarted
    fallback_path = "uploaded_files/latest_upload.csv"
    if os.path.exists(fallback_path):
        try:
            df = pd.read_csv(fallback_path)
            # Normalize headings
            df.columns = [col.strip().capitalize() for col in df.columns]
            df.rename(columns={
                "Date": "Month",
                "Credit": "Income",
                "Debit": "Expense",
                "Amount": "Income"
            }, inplace=True)
            df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
            if "Income" not in df.columns:
                df["Income"] = 0
            if "Expense" not in df.columns:
                df["Expense"] = 0

            data_store.stored_df = df  # recache
            print("📄 Loaded from disk (fallback)")
            print("💾 Disk preview:\n", df.head(3))
            return df.copy(), "disk"
        except Exception as e:
            print("❌ Fallback load failed:", e)

    print("🚫 No data available in memory or disk.")
    return None, None


# ---------------------------
# Helper analytics
# ---------------------------
def last_month_row(df):
    return df.sort_values("Month").iloc[-1]

def savings_rate_series(df):
    # (Income - Expense) / Income
    sr = (df["Income"] - df["Expense"]) / df["Income"].replace(0, np.nan)
    return sr.fillna(0)

def risk_level(df):
    """LOW / MEDIUM / HIGH based on savings rate & income volatility."""
    sr = savings_rate_series(df)
    income_vol = df["Income"].pct_change().rolling(3).std().iloc[-1]
    income_vol = 0 if pd.isna(income_vol) else float(income_vol)
    last_sr = float(sr.iloc[-1]) if not sr.empty else 0.0

    score = 0
    if last_sr < 0.05: score += 2
    if last_sr < 0: score += 3
    if income_vol > 0.25: score += 2
    if income_vol > 0.40: score += 3

    if score <= 1: return "LOW"
    if score <= 3: return "MEDIUM"
    return "HIGH"

def detect_anomalies(df, target, top_k: int = 5):
    """
    Detect anomalies on MONTHLY totals with adaptive fallbacks:
    1) 2.0σ rule
    2) 1.5σ rule (if none)
    3) Top-k largest deviations vs mean (guarantee up to 3 rows)
    4) If σ==0, use top-k largest absolute MoM % changes
    Returns newest-first, with % vs avg.
    """
    # Monthly totals for the chosen target
    monthly = (
        df.set_index("Month")
          .resample("MS")
          .agg({target: "sum"})
          .rename(columns={target: "val"})
          .dropna()
    )
    if len(monthly) < 3:
        return []  # not enough history

    mu = float(monthly["val"].mean())
    sigma = float(monthly["val"].std(ddof=0))

    def format_rows(sel):
        out = []
        for dt, row in sel.sort_index(ascending=False).head(top_k).iterrows():
            val = float(row["val"])
            pct_vs_avg = 0 if mu == 0 else (val - mu) / abs(mu) * 100.0
            level = "high" if val > mu else "low"
            out.append(
                f"{dt.strftime('%B %Y')} {target.lower()} was unusually {level}: "
                f"${val:.2f} ({'+' if pct_vs_avg>0 else ''}{pct_vs_avg:.0f}% vs avg)"
            )
        return out

    # σ==0 → use MoM % changes as a fallback signal
    if sigma == 0 or pd.isna(sigma):
        mom = monthly["val"].pct_change().dropna()
        if mom.empty:
            return []
        biggest = mom.abs().sort_values(ascending=False).head(min(top_k, 3)).index
        sel = monthly.loc[biggest]
        return format_rows(sel)

    # 1) strict: 2σ
    dev = (monthly["val"] - mu).abs()
    sel = monthly.loc[dev >= 2.0 * sigma]
    if not sel.empty:
        return format_rows(sel)

    # 2) softer: 1.5σ
    sel = monthly.loc[dev >= 1.5 * sigma]
    if not sel.empty:
        return format_rows(sel)

    # 3) guaranteed: top-3 largest deviations
    top = dev.sort_values(ascending=False).head(min(top_k, 3)).index
    sel = monthly.loc[top]
    return format_rows(sel)

def top_category_spikes(df):
    """Return list of (category, pct_change, abs_change) for spikes vs 3m avg (Expense)."""
    if "Category" not in df.columns:
        return []
    tmp = df.copy()
    tmp["Month"] = pd.to_datetime(tmp["Month"], errors="coerce")
    tmp = tmp.dropna(subset=["Month"])
    by_month_cat = tmp.groupby([pd.Grouper(key="Month", freq="MS"), "Category"])["Expense"].sum().reset_index()
    last_m = by_month_cat["Month"].max()
    prev_3m = (last_m - pd.DateOffset(months=3))

    last = by_month_cat[by_month_cat["Month"] == last_m]
    hist = by_month_cat[(by_month_cat["Month"] > prev_3m) & (by_month_cat["Month"] < last_m)]
    if hist.empty or last.empty:
        return []

    spikes = []
    for _, r in last.iterrows():
        cat = r["Category"]
        last_val = float(r["Expense"])
        avg3 = hist[hist["Category"] == cat]["Expense"].mean()
        if pd.isna(avg3) or avg3 == 0:
            continue
        pct = (last_val - avg3) / avg3
        if pct >= 0.30:  # 30%+ spike
            spikes.append((cat, float(pct), float(last_val - avg3)))
    spikes.sort(key=lambda x: x[1], reverse=True)
    return spikes[:3]

def generate_dynamic_suggestions(df, target):
    """Diverse, file‑specific suggestions (max 8)."""
    suggestions = []

    df = df.sort_values("Month")
    last = last_month_row(df)
    last_month = last["Month"].strftime("%B %Y")
    avg3_income = df["Income"].tail(3).mean()
    avg3_expense = df["Expense"].tail(3).mean()

    # 1) Income/Expense directional
    if target == "Income":
        base = avg3_income if avg3_income else 1.0
        change = (last["Income"] - avg3_income) / base
        if change <= -0.10:
            suggestions.append({
                "text": f"Income in {last_month} fell {abs(change)*100:.0f}% vs your 3‑month average. Add a time‑boxed weekend gig (6–8 hrs) to recover ${max(25, avg3_income*0.10):.0f}+ next month."
            })
        elif change >= 0.10:
            suggestions.append({
                "text": f"Income in {last_month} rose {change*100:.0f}% vs your 3‑month average. Auto‑allocate ${max(20, (last['Income']-avg3_income)*0.5):.0f} to your Emergency Fund."
            })
        else:
            suggestions.append({"text": "Income is within a normal range. Keep consistent hours; avoid one‑offs that create volatility."})
    else:
        base = avg3_expense if avg3_expense else 1.0
        change = (last["Expense"] - avg3_expense) / base
        if change >= 0.15:
            suggestions.append({
                "text": f"Expenses in {last_month} are {change*100:.0f}% higher than your 3‑month average. Set a weekly cap = 3‑month average ÷ 4 and stick to it."
            })
        else:
            suggestions.append({"text": "Expenses are near normal. Shave 3–5% this month to accelerate savings."})

    # 2) Category spikes
    spikes = top_category_spikes(df)
    for cat, pct, delta in spikes:
        suggestions.append({
            "text": f"{cat} spending spiked {pct*100:.0f}% (~${delta:.0f}). Roll back to the 3‑month average to save about ${delta:.0f} this month."
        })

    # 3) Savings rate gap
    sr = savings_rate_series(df)
    target_sr = 0.15
    if not sr.empty and sr.iloc[-1] < target_sr:
        need = max(0.0, target_sr - float(sr.iloc[-1]))
        base_income = float(df["Income"].iloc[-1])
        if base_income > 0:
            amt = base_income * need
            amt = max(25, amt)                 # at least $25/month
            amt = int(round(amt / 10.0)) * 10  # round to nearest $10
            suggestions.append({
                "text": f"Savings rate is {sr.iloc[-1]*100:.1f}% (target {target_sr*100:.0f}%). Auto-transfer ${amt}/month to savings on payday."
            })

    # 4) Debt servicing (optional columns)
    for col in ["Debt", "Emi", "LoanPayment", "Loan_Payment"]:
        if col in df.columns:
            d = float(df[col].tail(1).mean())
            inc = float(df["Income"].tail(1).mean())
            if inc > 0 and d / inc > 0.25:
                suggestions.append({"text": "Debt payments exceed 25% of income. Consider consolidation/refi to lower APR and target <20%."})
            break

    # 5) Recent month-on-month notes (true monthly totals, no duplicates)
    monthly = (
        df.set_index("Month")
          .resample("MS")
          .agg({"Income": "sum", "Expense": "sum"})
          .dropna(how="all")
    )

    def add_mom_note(cur_idx: int):
        cur_dt = monthly.index[cur_idx]
        prev_dt = monthly.index[cur_idx - 1]
        if target == "Income":
            prev_val = float(monthly.iloc[cur_idx - 1]["Income"])
            cur_val  = float(monthly.iloc[cur_idx]["Income"])
            label = "Income"
        else:
            prev_val = float(monthly.iloc[cur_idx - 1]["Expense"])
            cur_val  = float(monthly.iloc[cur_idx]["Expense"])
            label = "Expense"

        if prev_val == 0:
            return  # avoid divide-by-zero / nonsense %
        pct = (cur_val - prev_val) / abs(prev_val) * 100.0
        arrow = "↑" if pct > 0 else "↓"
        suggestions.append({
            "text": f"{cur_dt.strftime('%B %Y')}: {label} {arrow}{abs(pct):.0f}% vs {prev_dt.strftime('%B %Y')}."
        })

    if len(monthly) >= 2:
        add_mom_note(len(monthly) - 1)  # latest month vs previous
    if len(monthly) >= 3:
        add_mom_note(len(monthly) - 2)  # previous month vs one before

    # De‑dupe & limit
    out, seen = [], set()
    for item in suggestions:
        if item["text"] not in seen:
            out.append(item); seen.add(item["text"])
    return out[:8]

def estimate_6m_impact(df):
    """$ impact if user rolls back largest category spike for 6 months (abs, no noise)."""
    spikes = top_category_spikes(df)
    if not spikes:
        impact = max(0.0, (df["Expense"].tail(1).mean()) * 0.05) * 6
    else:
        _, _, delta = spikes[0]
        impact = abs(float(delta)) * 6  # ensure positive

    impact = round(float(impact), 2)
    return impact if impact >= 25 else None  # hide if too small to matter


# ---------------------------
# UI helpers
# ---------------------------
def calculate_goal_progress(df):
    total_saved = float((df['Income'] - df['Expense']).sum())
    return [
        {"label": "Emergency Fund", "percent": min(100, round((total_saved / 10000) * 100))},
        {"label": "Travel Fund", "percent": min(100, round((total_saved / 3000) * 100))}
    ]


# ---------------------------
# PDF watermark helper
# ---------------------------
def add_watermark(fig, logo_path=LOGO_PATH, alpha=0.08, rel_width=0.5):
    """
    Place a semi-transparent logo centered on the figure.
    rel_width is fraction of figure width.
    """
    if not os.path.exists(logo_path):
        return  # logo optional; skip if not found

    import matplotlib.image as mpimg
    img = mpimg.imread(logo_path)
    h, w = img.shape[:2]
    aspect = h / float(w)

    width = rel_width
    height = width * aspect
    left = (1 - width) / 2
    bottom = (1 - height) / 2

    ax_wm = fig.add_axes([left, bottom, width, height], zorder=0)
    ax_wm.imshow(img, alpha=alpha)
    ax_wm.axis("off")


# ---------------------------
# Core forecast
# ---------------------------
def run_forecast(df, target, months, scenario="current"):
    df = df.copy()
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    if df["Month"].isnull().all():
        raise ValueError("❌ Failed to parse 'Month' column into datetime.")
    df = df.dropna(subset=["Month"]).sort_values("Month")

    # --- Aggregate to monthly totals first (key fix) ---
    monthly = (
        df.set_index("Month")
          .resample("MS")  # month start
          .agg({"Income": "sum", "Expense": "sum"})
          .reset_index()
          .rename(columns={"Month": "ds"})
    )

    # Choose series to forecast from monthly totals
    y_col = target if target in ["Income", "Expense"] else "Income"
    y_series = monthly[y_col].astype(float)
    prophet_df = monthly[["ds", y_col]].rename(columns={y_col: "y"})

    # --- Prophet on monthly totals ---
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=months, freq="MS")
    forecast_base = model.predict(future).copy()

    # --- Scenario scaling (income up in best, expense down; reversed for worst) ---
    is_income = (y_col.lower() == "income")
    adj = 0.0
    if scenario == "best":
        adj = 0.07 if is_income else -0.07
    elif scenario == "worst":
        adj = -0.07 if is_income else 0.07
    scenario_label = (
        f"{'Best' if scenario=='best' else 'Worst' if scenario=='worst' else 'Current'} "
        f"Case ({'+' if adj>0 else ''}{int(round(adj*100))}%)" if scenario != "current" else "Current"
    )
    scenario_note  = "No special adjustments applied." if scenario == "current" else \
                     f"Applies {('+' if adj>0 else '')}{int(round(adj*100))}% to projected {y_col.lower()}."
    print(f"[Forecast] scenario='{scenario}' adj={adj}")

    forecast = forecast_base.copy()
    if adj != 0.0:
        for col in ("yhat", "yhat_lower", "yhat_upper"):
            if col in forecast.columns:
                forecast[col] = forecast[col] * (1 + adj)

    # --- Next month delta (vs baseline) ---
    last_date = prophet_df["ds"].max()
    next_month = pd.date_range(start=last_date, periods=2, freq="MS")[1]
    if next_month in forecast["ds"].values:
        scen_next = float(forecast.loc[forecast["ds"] == next_month, "yhat"].values[0])
        base_next = float(forecast_base.loc[forecast_base["ds"] == next_month, "yhat"].values[0])
    else:
        scen_next = float(forecast["yhat"].iloc[-1])
        base_next = float(forecast_base["yhat"].iloc[-1])

    delta = scen_next - base_next
    denom = abs(base_next) if abs(base_next) > 1e-6 else (abs(scen_next) if abs(scen_next) > 1e-6 else 1.0)
    raw_pct = (delta / denom) * 100.0
    improve_dollars = delta if is_income else -delta
    improve_pct = raw_pct if is_income else -raw_pct
    arrow = "↑" if improve_dollars > 0 else ("↓" if improve_dollars < 0 else "→")
    good_word = "better" if improve_dollars > 0 else ("worse" if improve_dollars < 0 else "unchanged")
    target_word = "income" if is_income else "expense"

    if scenario == "best":
        scenario_line = (
            f"✅ Scenario: {scenario_label}. Next month looks {good_word}: "
            f"{arrow} {abs(improve_pct):.1f}% (~${abs(improve_dollars):.0f}) {target_word} vs current path."
        )
    elif scenario == "worst":
        scenario_line = (
            f"⚠️ Scenario: {scenario_label}. Next month looks {good_word}: "
            f"{arrow} {abs(improve_pct):.1f}% (~${abs(improve_dollars):.0f}) {target_word} vs current path."
        )
    else:
        scenario_line = f"ℹ️ Scenario: {scenario_label}. {scenario_note}"

    # --- Plot chart ---
    fig = model.plot(forecast)
    ax = fig.gca()
    ax.set_title(f"{y_col} Forecast • {scenario_label}", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(f"{y_col} ($)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Forecast Line (yhat)", "Upper Bound", "Lower Bound"], loc="upper left")

    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    # --- Strategy text from monthly totals (fixes the $0 confusion) ---
    avg_val = float(y_series.mean())
    last_val = float(y_series.iloc[-1])

    if y_col.lower() == "income" and last_val == 0:
        status_line = "🧾 No income recorded last month (check if data is missing or if this is expected)."
    else:
        status_line = f"🧾 Last month’s {y_col.lower()}: ${last_val:,.2f}."

    if len(y_series) >= 2:
        prev = float(y_series.iloc[-2])
        if prev == 0 and last_val == 0:
            trend_comment = "Trend last month: flat at $0."
        elif prev == 0:
            trend_comment = f"Trend last month: up to ${last_val:,.0f} from $0."
        elif last_val == 0:
            trend_comment = f"Trend last month: down to $0 from ${prev:,.0f}."
        else:
            pct = (last_val - prev) / abs(prev) * 100.0
            arrow = "↑" if pct > 0 else "↓"
            trend_comment = f"Trend last month: {arrow} {abs(pct):.0f}% vs prior month."
    else:
        trend_comment = "Trend: not enough data."

    r = risk_level(df)

    avg_line = f"📈 Average monthly {y_col.lower()}: ${avg_val:,.2f}."
    strategy_text = f"{avg_line}\n{status_line}\n\n{trend_comment}\n{scenario_line}\n"

    return {
        "chart": chart_base64,
        "strategy": strategy_text,
        "goal_progress": calculate_goal_progress(df),
        "anomalies": detect_anomalies(df, y_col),              # still works; it resamples monthly internally
        "suggestions": generate_dynamic_suggestions(df, y_col),
        "risk_level": r,
        "impact_estimate_6m": estimate_6m_impact(df),
        "scenario_applied": scenario,
        "scenario_label": scenario_label,
        "scenario_note": scenario_note,
        "status": "success",
    }



# ---------------------------
# Routes
# ---------------------------
@router.post("/forecast/")
async def forecast_post(req: Request):
    try:
        body = await req.json()
        target = body.get("target", "Income")
        months = int(body.get("months", 6))
        scenario = body.get("scenario", "current")  # new

        df, source = load_dataframe()
        if df is None or df.empty:
            return {"error": "❌ No uploaded data found. Please upload a file first."}

        print(f"✅ Forecast is using source: {source}")
        return run_forecast(df, target, months, scenario=scenario)  # pass through

    except Exception as e:
        traceback.print_exc()
        return {"error": f"❌ Forecasting failed: {str(e)}"}


@router.post("/forecast/pdf")
async def forecast_pdf(req: Request):
    try:
        body = await req.json()
        target = body.get("target", "Income")
        months = int(body.get("months", 6))
        scenario = body.get("scenario", "current")

        df, source = load_dataframe()
        if df is None or df.empty:
            return Response(content="❌ No uploaded data found.", status_code=400)

        # Run the same forecast you use for the UI
        res = run_forecast(df, target, months, scenario=scenario)

        # Build a multi-page PDF using matplotlib only
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.image as mpimg
        import textwrap
        import datetime as _dt

        buf = BytesIO()
        with PdfPages(buf) as pdf:
            # --- Page 1: Cover / Summary ---
            fig = plt.figure(figsize=(8.5, 11))
            fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
            add_watermark(fig, alpha=0.08, rel_width=0.55)  # watermark
            ax = fig.add_axes([0,0,1,1]); ax.axis('off')

            title = "FinBuddy — Strategy & Forecast"
            subtitle = f"{target} • Scenario: {res.get('scenario_label','Current')}"
            meta = f"Generated: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            risk = f"Risk Level: {res.get('risk_level','—')}"

            fig.text(0.06, 0.95, title, fontsize=20, weight='bold', va='top')
            fig.text(0.06, 0.915, subtitle, fontsize=12)
            fig.text(0.06, 0.895, meta + "   |   " + risk, fontsize=10, color="#555")

            # Strategy text split
            lines = [ln.strip() for ln in (res.get("strategy") or "").split("\n") if ln.strip()]
            wrap = textwrap.TextWrapper(width=95)
            y = 0.85
            for ln in lines:
                blks = wrap.wrap(ln)
                for b in blks:
                    fig.text(0.06, y, b, fontsize=11)
                    y -= 0.022
                y -= 0.01

            # Timeless money moves (actionable content)
            moves = [
                "Build an emergency fund: start with $1,000 → 1 month → 3–6 months of expenses.",
                "Use a simple budget rule: 50% needs / 30% wants / 20% saving & debt.",
                "Automate savings on payday; pay yourself first.",
                "Audit subscriptions quarterly; cancel or downgrade unused ones.",
                "Tackle debt via Avalanche (highest APR first) or Snowball (smallest balance first).",
                "Move idle cash to a high-yield savings account.",
                "Set weekly caps for variable categories (e.g., groceries, dining, transport).",
            ]
            fig.text(0.06, max(0.15, y-0.03), "Timeless Money Moves", fontsize=14, weight='bold')
            moves_text = "\n".join("• " + m for m in moves)
            fig.text(0.06, max(0.12, y-0.08), wrap.fill(moves_text), fontsize=11)

            ax.set_title(no_emoji("📈 Forecast — Income vs Expense"))
            fig.suptitle(no_emoji("🧾 Report — August"))

            pdf.savefig(fig, dpi=150)
            plt.close(fig)

            # --- Page 2: Chart ---
            fig2 = plt.figure(figsize=(11, 8.5))
            add_watermark(fig2, alpha=0.05, rel_width=0.45)  # watermark
            ax2 = fig2.add_subplot(111); ax2.axis('off')
            chart_b64 = res.get("chart")
            if chart_b64:
                chart_bytes = base64.b64decode(chart_b64)
                img = mpimg.imread(BytesIO(chart_bytes), format='png')
                ax2.imshow(img); ax2.set_title("Forecast Chart", fontsize=14)
            pdf.savefig(fig2, dpi=150)
            plt.close(fig2)

            # --- Page 3: Monthly suggestions ---
            sugg = res.get("suggestions") or []
            # normalize to text list
            sugg_texts = [
                (s["text"] if isinstance(s, dict) and "text" in s else str(s)).strip()
                for s in sugg
                if (isinstance(s, dict) and s.get("text")) or (isinstance(s, str) and s.strip())
            ]
            if sugg_texts:
                fig3 = plt.figure(figsize=(8.5, 11))
                fig3.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
                add_watermark(fig3, alpha=0.08, rel_width=0.5)  # watermark
                ax3 = fig3.add_axes([0,0,1,1]); ax3.axis('off')
                fig3.text(0.06, 0.95, "Monthly Suggestions", fontsize=16, weight='bold', va='top')

                y = 0.91
                for t in sugg_texts:
                    for line in wrap.wrap("• " + t):
                        fig3.text(0.06, y, line, fontsize=11)
                        y -= 0.022
                        if y < 0.08: break
                    y -= 0.01
                    if y < 0.08: break

                pdf.savefig(fig3, dpi=150)
                plt.close(fig3)

        pdf_bytes = buf.getvalue()
        headers = {"Content-Disposition": "attachment; filename=FinBuddy_Forecast.pdf"}
        return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)

    except Exception as e:
        traceback.print_exc()
        return Response(content=f"PDF generation failed: {e}", status_code=500)


# --- Debug & maintenance endpoints ------------------------------------------
@router.post("/upload/clear")
def clear_upload():
    # resets the in-memory cached dataframe so the next upload is used
    import modules.data_store as data_store
    data_store.stored_df = None
    return {"ok": True, "message": "stored_df cleared"}

@router.get("/debug/summary")
def debug_summary():
    # shows what source is being used and monthly totals the model sees
    df, src = load_dataframe()
    if df is None or df.empty:
        return {"source": src, "rows": 0, "monthly": {}}

    # make sure Month is datetime and resample by month
    tmp = df.copy()
    tmp["Month"] = pd.to_datetime(tmp["Month"], errors="coerce")
    tmp = tmp.dropna(subset=["Month"]).set_index("Month")

    monthly = (tmp
               .resample("MS")
               .agg({"Income": "sum", "Expense": "sum"})
               .round(2))

    return {
        "source": src,                         # "memory" or "disk"
        "rows": int(len(df)),
        "monthly": monthly.to_dict(orient="index")
    }