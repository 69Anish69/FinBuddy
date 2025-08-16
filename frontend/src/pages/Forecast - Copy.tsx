import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";

const Forecast = () => {
  const [target, setTarget] = useState('Income');
  const [months, setMonths] = useState(6);
  const [forecastText, setForecastText] = useState('');
  const [chartUrl, setChartUrl] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [goalProgress, setGoalProgress] = useState<any[]>([]);
  const [anomalyData, setAnomalyData] = useState<any[]>([]);
  const [strategyData, setStrategyData] = useState<any[]>([]);
  const [lastUploadTime, setLastUploadTime] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);

  // NEW: scenario + risk/impact
  const [scenario, setScenario] = useState<'current' | 'best' | 'worst'>(
    (localStorage.getItem('finbuddy_scenario') as any) || 'current'
  );
  const [riskLevel, setRiskLevel] = useState<string | null>(null);
  const [impact6m, setImpact6m] = useState<number | null>(null);

  const navigate = useNavigate();

  const fetchUploadTime = async () => {
    try {
      const res = await axios.get("http://localhost:8000/upload/last");
      if (res.data?.last_upload) {
        setLastUploadTime(res.data.last_upload);
        return true;
      }
    } catch {
      console.warn("❌ Could not fetch upload time");
    }
    return false;
  };

  const generateForecast = async () => {
    setError('');
    setForecastText('');
    setChartUrl('');
    setGoalProgress([]);
    setAnomalyData([]);
    setStrategyData([]);
    setRiskLevel(null);
    setImpact6m(null);
    setLoading(true);

    try {
      const res = await axios.post('http://localhost:8000/forecast/', {
        target,
        months: Number(months),
        scenario, // <-- send scenario to backend
      });

      const {
        strategy,
        chart,
        goal_progress,
        anomalies,
        suggestions,
        risk_level,
        impact_estimate_6m,
      } = res.data;

      if (chart) {
        const cacheBustedChart = `data:image/png;base64,${chart}#${Date.now()}`;
        setChartUrl(cacheBustedChart);
      }
      if (strategy) setForecastText(strategy);
      if (goal_progress) setGoalProgress(goal_progress);

      if (anomalies) {
        setAnomalyData(
          anomalies.map((msg: string) => ({
            label: msg,
            color: msg.includes("unusually high") ? "bg-red-500" : "bg-blue-500",
          }))
        );
      }

      if (Array.isArray(suggestions)) {
        const isStringList = typeof suggestions[0] === 'string';
        setStrategyData(isStringList ? suggestions.map((t: string) => ({ text: t })) : suggestions);
      }

      if (risk_level) setRiskLevel(risk_level);
      setImpact6m(typeof impact_estimate_6m === 'number' ? impact_estimate_6m : null);
    } catch (err) {
      console.error(err);
      setError('❌ Upload or forecast failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // auto-run forecast when scenario changes (if file uploaded)
  useEffect(() => {
    generateForecast(); // always re-run when scenario changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scenario]);


  useEffect(() => {
    const autoRun = async () => {
      const hasUpload = await fetchUploadTime();
      if (hasUpload) generateForecast();
    };
    autoRun();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const riskBadge = (risk?: string | null) => {
    if (!risk) return null;
    const map: Record<string, string> = {
      LOW: 'bg-green-100 text-green-700',
      MEDIUM: 'bg-yellow-100 text-yellow-700',
      HIGH: 'bg-red-100 text-red-700',
    };
    return (
      <span className={`ml-2 px-2 py-0.5 rounded text-xs font-semibold ${map[risk] || 'bg-gray-100 text-gray-600'}`}>
        {risk}
      </span>
    );
  };

  // simple print/export of the strategy card (no deps)
  const printStrategy = () => {
    const node = document.getElementById('strategyCard');
    const chartImg = document.getElementById('forecastChartImg') as HTMLImageElement | null;

    const section = `
      <div style="border:1px solid #e5e7eb;border-radius:8px;padding:16px;margin-top:12px">
        ${chartImg ? `<img src="${chartImg.src}" style="max-width:100%;height:auto" />` : ''}
        ${node ? node.innerHTML : ''}
      </div>`;

    const html = `
      <html>
        <head>
          <title>FinBuddy Strategy</title>
          <style>
            body{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto; padding:16px; }
            h1{ font-size:20px; margin:0 0 8px; }
            .muted{ color:#4b5563; }
          </style>
        </head>
        <body>
          <h1>FinBuddy – Strategy & Forecast</h1>
          <div class="muted">Generated on ${new Date().toLocaleString()}</div>
          ${section}
        </body>
      </html>`;
    const w = window.open('', '_blank', 'width=900,height=700');
    if (!w) return;
    w.document.open();
    w.document.write(html);
    w.document.close();
    w.focus();
    w.print();
  };

  const exportPDF = async () => {
  try {
    setExporting(true);
    const res = await axios.post(
      'http://localhost:8000/forecast/pdf',
      { target, months: Number(months), scenario },
      { responseType: 'blob' }
    );
    const blob = new Blob([res.data], { type: 'application/pdf' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `FinBuddy_Forecast_${scenario}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  } catch (e) {
    console.error(e);
    setError('❌ PDF export failed. Please try again.');
  } finally {
    setExporting(false);
  }
};

  // === Strategy UI helpers (REPLACE) ===
  const scenarioLabel =
    scenario === 'best' ? 'Best Case (+7%)' :
    scenario === 'worst' ? 'Worst Case (-7%)' : 'Current';

  const strategyLines = (forecastText || '')
    .split('\n')
    .map(s => s.trim())
    .filter(Boolean);

  // only the line that actually contains "Scenario:"
  const scenarioLine = strategyLines.find(l => l.includes('Scenario:')) || null;
  // everything else = key takeaways
  const nonScenarioLines = strategyLines.filter(l => !l.includes('Scenario:'));

  // callout color by scenario
  const calloutClass =
    scenario === 'best'
      ? 'border-emerald-200 bg-emerald-50 text-emerald-900'
      : 'border-amber-200 bg-amber-50 text-amber-900';

  // Impact line (good = green "savings", bad = red "extra cost")
  const impactIsGood = (impact6m ?? 0) >= 0;
  const impactText = impact6m !== null
    ? `Estimated 6-month ${impactIsGood ? 'savings' : 'extra cost'}: ~$${Math.abs(impact6m).toLocaleString()}`
    : null;

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Navigation />

      <main className="flex-grow container mx-auto px-4 py-16">
        <div className="mx-auto max-w-6xl">
          <h1 className="text-4xl font-bold text-foreground mb-2 text-center">
            Forecast Your Financial Future
          </h1>
          {lastUploadTime && (
            <p className="text-sm text-muted-foreground text-center mb-6">
              📂 Last uploaded at: <b>{lastUploadTime}</b>
            </p>
          )}
          <p className="text-xl text-muted-foreground mb-12 text-center">
            See how your savings will grow and reach your goals
          </p>

          <div className="grid lg:grid-cols-4 gap-8">
            <div className="lg:col-span-3 space-y-6">
              <div
                className={`rounded-lg border shadow-card-hover min-h-[320px] flex justify-center items-center ${
                  !chartUrl ? "bg-muted" : "bg-card"
                }`}
              >
                {loading ? (
                  <p className="text-gray-500 animate-pulse">Generating forecast...</p>
                ) : chartUrl ? (
                  <img id="forecastChartImg" src={chartUrl} alt="Forecast Chart" className="w-full max-h-[280px] object-contain" />
                ) : (
                  <p className="text-muted-foreground text-sm text-center px-4">
                    📈 Upload your financial data and click <b>"Generate Strategy"</b> to view forecast.
                  </p>
                )}
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-card rounded-lg border shadow-card-hover p-6">
                  <h3 className="text-lg font-semibold mb-2">Goal Progress</h3>
                  <div className="space-y-3">
                    {goalProgress.map((goal, idx) => (
                      <div key={idx}>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm">{goal.label}</span>
                          <span className="text-sm text-primary">{goal.percent}%</span>
                        </div>
                        <div className="w-full bg-secondary rounded-full h-2">
                          <div
                            className="bg-gradient-financial h-2 rounded-full"
                            style={{ width: `${goal.percent}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                  <button
                    onClick={() => navigate('/goals')}
                    className="w-full mt-6 bg-primary text-white py-3 rounded-md font-semibold"
                  >
                    Continue
                  </button>
                </div>

                <div className="bg-card rounded-lg border shadow-card-hover p-6">
                  <h3 className="text-lg font-semibold mb-2 text-red-600">Anomaly Detection</h3>
                  <div className="space-y-2 text-sm">
                    {anomalyData.length === 0 ? (
                      <div className="text-gray-500">No notable anomalies found in recent months.</div>
                    ) : (
                      anomalyData.map((item, idx) => (
                        <div className="flex items-center gap-2" key={idx}>
                          <span className={`w-2 h-2 ${item.color} rounded-full`} />
                          <span>{item.label}</span>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>

              {forecastText && (
                <div
                  id="strategyCard"
                  className="bg-card border rounded-xl shadow-sm p-5 md:p-6 text-[15px] text-foreground"
                >
                  {/* Header */}
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex flex-wrap items-center gap-3">
                      <h3
                        className="text-xl font-semibold flex items-center"
                        style={{
                          color:
                            riskLevel === 'HIGH'
                              ? '#b91c1c'
                              : riskLevel === 'MEDIUM'
                              ? '#b45309'
                              : '#166534',
                        }}
                      >
                        🧠 AI Strategy & Forecast
                      </h3>

                      {/* Risk pill */}
                      {riskBadge(riskLevel)}

                      {/* Scenario pill */}
                      <span className="inline-flex items-center rounded-full border px-2 py-0.5 text-xs bg-blue-50 text-blue-700 border-blue-200">
                        {scenarioLabel}
                      </span>
                    </div>

                    <div className="flex flex-wrap items-center gap-3">
                     {impactText && (
                      <div className={`text-xs md:text-sm ${impactIsGood ? 'text-green-700' : 'text-red-700'}`}>
                        <span className="font-semibold text-foreground">{impactText}</span>
                      </div>
                    )}
                      <button
                        type="button"
                        onClick={exportPDF}
                        disabled={exporting}
                        className={`text-xs px-3 py-1.5 rounded-md border bg-white hover:bg-gray-50 ${exporting ? 'opacity-60 cursor-not-allowed' : ''}`}
                        title="Download strategy as PDF"
                        aria-busy={exporting}
                      >
                        {exporting ? 'Exporting…' : 'Export PDF'}
                      </button>
                    </div>
                  </div>

                  {/* Scenario callout (only for Best/Worst, and only if present) */}
                  {scenario !== 'current' && scenarioLine && (
                    <div className={`mt-4 rounded-md border px-3 py-2 text-sm ${calloutClass}`}>
                      {scenarioLine}
                    </div>
                  )}

                  {/* Key takeaways */}
                  <div className="mt-4 space-y-1.5">
                    {nonScenarioLines.map((line, i) => (
                      <p key={i} className="leading-6">{line}</p>
                    ))}
                  </div>
                  {/* Suggestions */}
                  {strategyData.length > 0 && (
                    <div className="mt-6">
                      <div className="mb-2 flex items-center justify-between">
                        <h4 className="text-[16px] font-semibold text-blue-600">📅 Monthly Suggestions</h4>
                        <button
                          onClick={() => navigate('/reports?view=spending&range=last-3m')}
                          className="text-xs px-2 py-1 border rounded hover:bg-gray-50"
                          title="Open Reports to act on this"
                        >
                          View in Reports
                        </button>
                      </div>

                      <ul className="space-y-2">
                        {strategyData.map((item, idx) => (
                          <li key={idx} className="flex items-start gap-2">
                            <span className="mt-[6px] inline-block h-2 w-2 rounded-full bg-gradient-to-r from-blue-400 to-blue-600" />
                            <span className="flex-1">
                              {item.text ? item.text : (<><strong>{item.month}:</strong> {item.suggestion}</>)}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {error && (
                <div className="bg-red-100 text-red-700 border border-red-300 px-4 py-3 rounded">
                  {error}
                </div>
              )}
            </div>

            <div className="space-y-4">
              <div className="bg-card rounded-lg border p-4">
                <h3 className="font-semibold mb-4">Filters</h3>
                <div className="space-y-3">
                  <div>
                    <label className="text-sm font-medium">Target</label>
                    <select
                      value={target}
                      onChange={(e) => setTarget(e.target.value)}
                      className="w-full mt-1 px-3 py-2 border border-input rounded-md text-sm"
                    >
                      <option>Income</option>
                      <option>Expense</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-sm font-medium">Forecast Months</label>
                    <input
                      type="number"
                      value={months}
                      onChange={(e) => setMonths(Number(e.target.value))}
                      className="w-full mt-1 px-3 py-2 border border-input rounded-md text-sm"
                    />
                  </div>

                  {/* Scenario toggle */}
                  <div>
                    <label className="text-sm font-medium">Scenario</label>
                    <div className="mt-1 grid grid-cols-3 gap-1 text-xs">
                      {['current','best','worst'].map((opt) => (
                        <button
                          key={opt}
                          type="button"
                          onClick={() => setScenario(opt as any)}
                          className={`py-2 rounded border ${scenario === opt ? 'bg-primary text-white' : 'bg-white'}`}
                        >
                          {opt === 'current' ? 'Current' : opt === 'best' ? 'Best' : 'Worst'}
                        </button>
                      ))}
                    </div>
                  </div>

                  <button
                    onClick={generateForecast}
                    className="w-full bg-gradient-financial text-white py-2 rounded-md font-semibold text-sm"
                  >
                    Generate Strategy
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Forecast;
