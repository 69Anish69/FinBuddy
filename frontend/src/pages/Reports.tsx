'use client';

import { useEffect, useMemo, useState } from 'react';
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";


// helps confirm which file is actually rendering
const BUILD_TAG = "reports-v3-deduped";

type ReportStatus = 'ready' | 'processing' | 'error';
type Overview = {
  updated_at: string;
  status: { forecast: ReportStatus; goals: ReportStatus; loans: ReportStatus }; // risk removed
  score: number;
  insights: {
    ef_months: number;
    apr_save_12m: number;
    invest_uplift_pct: number;
    goal_buffer_months: number;
  };
  improvements: string[];
  next_move: string;
};

type Sections = {
  executive_summary: string;
  data_methods: { sources: string[]; processing: string[]; modeling: string[] };
  insights: { title: string; value: string }[];
  goals: any[];
  recommendations: string[];
  charts: {
    trend: { Month: string; Income: number; Expense: number }[];
    categories: { category: string; amount: number }[];
  };
  updated_at: string;
};

// ---------- Utils ----------
const API_BASE =
  (typeof process !== 'undefined' && process.env?.NEXT_PUBLIC_API_URL
    ? process.env.NEXT_PUBLIC_API_URL.replace(/\/+$/, '')
    : (typeof window !== 'undefined' && (window.location.port === '3000' || window.location.port === '8080'))
      ? 'http://127.0.0.1:8000'
      : '');

const fmt = (n: number) =>
  n.toLocaleString(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });

const DEFAULT_OVERVIEW: Overview = {
  updated_at: new Date().toISOString(),
  status: { forecast: 'ready', goals: 'ready', loans: 'ready' }, // risk removed
  score: 82,
  insights: { ef_months: 6.0, apr_save_12m: 3240, invest_uplift_pct: 2.3, goal_buffer_months: 2 },
  improvements: [
    'Pay high-APR first: +$200/mo',
    'Rebalance quarterly; deploy $1,500 idle cash',
    'Automate goal transfers',
  ],
  next_move: 'Auto-transfer $300/mo to HYSA for Car goal',
};

async function fetchOverview(): Promise<Overview> {
  const url = `${API_BASE}/report/overview`;
  const res = await fetch(url, { headers: { 'Content-Type': 'application/json' } });
  if (!res.ok) throw new Error(`GET ${url} -> ${res.status}`);
  return res.json();
}

async function fetchSections(): Promise<Sections | null> {
  try {
    const url = `${API_BASE}/report/sections`;
    const res = await fetch(url);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

async function generatePdf(): Promise<string | null> {
  const url = `${API_BASE}/report/generate`;
  const res = await fetch(url, { method: 'POST' });
  if (!res.ok) return null;
  const data = (await res.json()) as { download?: string };
  return data?.download ?? null;
}

// ---------- Component ----------
const Reports = () => {
  const [data, setData] = useState<Overview | null>(null);
  const [sections, setSections] = useState<Sections | null>(null);
  const [loading, setLoading] = useState(true);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    console.info(`[${BUILD_TAG}] mounted`);
    (async () => {
      try {
        const [d, s] = await Promise.all([fetchOverview(), fetchSections()]);
        setData(d);
        setSections(s);
      } catch {
        setData(DEFAULT_OVERVIEW);
        setError('Live data not available yet. Showing sample values.');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const scoreWidth = useMemo(
    () => `${Math.min(100, Math.max(0, data?.score ?? 0))}%`,
    [data?.score]
  );
  const statuses = useMemo(() => data?.status ?? DEFAULT_OVERVIEW.status, [data?.status]);

  // helpers
  const efMonths = data?.insights?.ef_months ?? DEFAULT_OVERVIEW.insights.ef_months;
  const aprSave = data?.insights?.apr_save_12m ?? DEFAULT_OVERVIEW.insights.apr_save_12m;
  const investUplift = data?.insights?.invest_uplift_pct ?? DEFAULT_OVERVIEW.insights.invest_uplift_pct;

  const hasSections = !!(sections && sections.executive_summary);

  async function handleGenerate() {
    setPdfLoading(true);
    try {
      const link = await generatePdf();
      if (link) window.open(link, '_blank');
      else window.print();
    } finally {
      setPdfLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-subtle flex flex-col">
      <Navigation />

      <div className="flex-grow container mx-auto px-4 py-16">
        <div className="mx-auto max-w-4xl">
          <h1 className="text-4xl font-bold text-foreground mb-2 text-center">Get Personalized Reports</h1>
          <p className="text-xl text-muted-foreground mb-6 text-center">
            Comprehensive financial analysis and recommendations
          </p>

          {error && (
            <div className="mb-6 rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800">
              {error}
            </div>
          )}

          {!loading && data && (
            <p className="text-xs text-muted-foreground mb-8 text-center">
              {/* Last updated: {new Date(data.updated_at).toLocaleString()} */}
            </p>
          )}

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Report Summary (risk removed) */}
            <div className="bg-card rounded-lg border shadow-card-hover p-6">
              <h3 className="text-lg font-semibold mb-4">Report Summary</h3>
              <div className="space-y-4">
                {[
                  ['Savings Forecast', statuses.forecast],
                  ['Goal Progress',   statuses.goals],
                  ['Loan Strategy',   statuses.loans],
                ].map(([label, status]) => {
                  const ready = status === 'ready';
                  return (
                    <div key={label as string} className="flex items-center justify-between p-3 bg-gradient-subtle rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 bg-gradient-financial rounded-full flex items-center justify-center">
                          <span className="text-white text-sm">•</span>
                        </div>
                        <span className="font-medium">{label}</span>
                      </div>
                      <span className={`${ready ? 'text-green-600' : 'text-yellow-600'} font-semibold`}>
                        {ready ? '✓ Complete' : '⏳ Processing'}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Key Insights */}
            <div className="bg-card rounded-lg border shadow-card-hover p-6">
              <h3 className="text-lg font-semibold mb-4">Key Insights</h3>
              {loading ? (
                <div className="animate-pulse space-y-3">
                  <div className="h-16 rounded bg-gray-200" />
                  <div className="h-16 rounded bg-gray-200" />
                  <div className="h-16 rounded bg-gray-200" />
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <h4 className="font-medium text-green-800 mb-1">✅ Strong Emergency Fund</h4>
                    <p className="text-sm text-green-700">You have {efMonths.toFixed(1)} months of expenses saved — excellent!</p>
                  </div>
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                    <h4 className="font-medium text-yellow-800 mb-1">⚠️ High Credit Card Interest</h4>
                    <p className="text-sm text-yellow-700">Prioritize payoff to save {fmt(aprSave)} over 12 months.</p>
                  </div>
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <h4 className="font-medium text-blue-800 mb-1">💡 Investment Opportunity</h4>
                    <p className="text-sm text-blue-700">You could increase returns by {investUplift}% with an optimized portfolio.</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Advisor Summary (fallback only; hidden when sections are loaded) */}
          {!hasSections && (
            <div className="mt-8 bg-card border-l-4 border-indigo-500 rounded-lg p-6 shadow-sm">
              <h4 className="font-semibold text-indigo-800 text-lg mb-2">💬 Advisor Summary</h4>
              <p className="text-muted-foreground text-sm">
                Your foundation looks solid. Emergency fund coverage is strong, and goals are on track.
                Address high-APR debt first to avoid compounding costs and consider a mild rebalance to
                lift expected returns without increasing risk.
              </p>
            </div>
          )}

          {/* Score */}
          <div className="mt-6 bg-white rounded-lg p-4 border shadow-sm">
            <h4 className="text-md font-semibold mb-2 text-gray-800">📈 Financial Health Score</h4>
            <div className="flex items-center gap-4">
              <div className="text-4xl font-bold text-green-600">{data?.score ?? 0}</div>
              <div className="flex-1 bg-gray-200 h-2 rounded-full overflow-hidden">
                <div className="bg-green-500 h-2 rounded-full" style={{ width: scoreWidth }} />
              </div>
            </div>
            <p className="text-xs text-muted-foreground mt-1">Excellent! Keep optimizing debt payments and allocation.</p>
          </div>

          {/* Improvements */}
          <div className="bg-amber-50 border border-amber-300 rounded-xl p-5 mt-6">
            <h4 className="font-semibold text-amber-800 mb-2">⚠️ What You Can Improve</h4>
            <ul className="list-disc ml-6 text-sm text-amber-900/90 space-y-1">
              {(data?.improvements ?? DEFAULT_OVERVIEW.improvements).map((x) => <li key={x}>{x}</li>)}
            </ul>
          </div>

          {/* Next move */}
          <div className="mt-6 rounded-xl border-l-4 border-blue-500 bg-blue-50 p-5">
            <h4 className="font-semibold text-blue-900 mb-1">🧠 Your Next Smart Move</h4>
            <p className="text-sm text-blue-900/90">{data?.next_move ?? DEFAULT_OVERVIEW.next_move}</p>
          </div>

          {/* Report Sections Preview (dynamic) */}
          {hasSections && (
            <div className="mt-8 space-y-6">
              {/* Executive Summary */}
              <div className="rounded-xl border bg-white p-6 shadow-md">
                <div className="flex items-center gap-2 mb-3">
                  <span className="inline-grid place-items-center h-8 w-8 rounded-full bg-indigo-100 text-indigo-700">💬</span>
                  <h3 className="text-lg font-semibold text-foreground">Executive Summary</h3>
                </div>
                <p className="text-sm leading-relaxed text-foreground/85">{sections!.executive_summary}</p>
              </div>

              {/* Recommendations – Next 30 Days */}
              <div className="rounded-xl border bg-white p-6 shadow-md">
                <div className="flex items-center gap-2 mb-3">
                  <span className="inline-grid place-items-center h-8 w-8 rounded-full bg-emerald-100 text-emerald-700">✅</span>
                  <h3 className="text-lg font-semibold text-foreground">Recommendations – Next 30 Days</h3>
                </div>
                {sections!.recommendations?.length ? (
                  <ul className="space-y-2">
                    {sections!.recommendations.map((r, i) => (
                      <li key={`${i}-${r.slice(0, 24)}`} className="flex gap-3">
                        <span className="mt-2 h-2 w-2 rounded-full bg-emerald-500" />
                        <span className="text-sm text-foreground/85">{r}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-muted-foreground">No recommendations yet.</p>
                )}
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="mt-10 text-center">
            <div className="bg-card rounded-lg border shadow-card-hover p-8">
              <h3 className="text-lg font-semibold mb-4">Generate Complete Report</h3>
              <p className="text-muted-foreground mb-6">
                Get a downloadable AI-powered PDF with strategy tables, charts, and action items
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                {/* <button
                  onClick={handleGenerate}
                  disabled={pdfLoading}
                  className="bg-black text-white px-6 py-3 rounded-md font-semibold hover:opacity-90 transition-all disabled:opacity-60"
                >
                  {pdfLoading ? 'Generating…' : 'Generate PDF'}
                </button> */}
                <button
                  onClick={() => window.print()}
                  className="bg-secondary text-secondary-foreground px-6 py-3 rounded-md font-semibold hover:bg-secondary/80 transition-all"
                >
                  🖨️ Print Report
                </button>
              </div>

              {/* Which file is loaded? */}
              <p className="mt-4 text-[10px] text-muted-foreground" data-build={BUILD_TAG}>
                Loaded: {BUILD_TAG}
              </p>

              {!API_BASE && (
                <p className="mt-2 text-xs text-muted-foreground">
                  Tip: set <code>NEXT_PUBLIC_API_URL</code> to enable live data.
                </p>
              )}
            </div>
          </div>
        </div>
      </div>

      <Footer />
    </div>
  );
};

export default Reports;
