'use client';

import React, { useEffect, useMemo, useState } from 'react';
import Navigation from '@/components/Navigation';
import Footer from '@/components/Footer';

// shadcn/ui
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';

// Icons
import { TrendingDown, PiggyBank, FileUp, Save, ScanSearch } from 'lucide-react';

// Charts
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from 'recharts';

/* ===========================
   Types
=========================== */
export type DebtType =
  | 'installment'
  | 'revolving'
  | 'loan'
  | 'credit_card'
  | 'auto'
  | 'mortgage'
  | 'student'
  | 'bnpl';

export type Loan = {
  id?: string;
  name: string;
  type?: DebtType;
  balance: number;
  apr: number | null;
  minPayment: number | null;
  termMonths?: number | null;
  source?: string;
};

export type PlanInput = {
  loans: Loan[];
  monthlyBudget: number;
  strategy: 'avalanche' | 'snowball' | 'equal';
};

export type PlanResult = {
  schedule: Array<{
    monthIndex: number;
    items: Array<{ id: string; interest: number; principal: number; endBalance: number }>;
    totalInterest: number;
    totalPaid: number;
  }>;
  payoffMonthByLoan: Record<string, number>;
  totalInterest: number;
  monthsToDebtFree: number;
};

/* ===========================
   Helpers
=========================== */
const API_BASE =
  ((import.meta as any).env?.VITE_API_URL as string | undefined)?.replace(/\/+$/, '') ||
  (typeof window !== 'undefined' && (window.location.port === '8080' || window.location.port === '3000')
    ? 'http://127.0.0.1:8000'
    : '');

const API = API_BASE;

const fmt = (v: number) =>
  v.toLocaleString(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
const fmt2 = (v: number) => v.toLocaleString(undefined, { style: 'currency', currency: 'USD' });

const monthlyRate = (apr: number | null) => (apr ? apr / 100 / 12 : 0);
const ready = (p: Loan) => Number(p.balance) > 0 && (p.apr ?? 0) > 0 && (p.minPayment ?? 0) > 0;

// ---- Layout locks to stop UI jumping ----
const CHART_HEIGHT = 280; // px
const YAXIS_WIDTH = 84;   // px

function minDue(l: Loan): number {
  if (l.type && ['installment', 'auto', 'mortgage', 'student'].includes(l.type)) {
    if (l.termMonths && l.termMonths > 0) {
      const r = monthlyRate(l.apr);
      if (r === 0) return l.balance / l.termMonths;
      const emi = (l.balance * r) / (1 - Math.pow(1 + r, -l.termMonths));
      return Math.min(emi, l.balance);
    }
  }
  const floor = l.minPayment ?? 0;
  return Math.min(floor, l.balance);
}

function orderLoans(loans: Loan[], strategy: PlanInput['strategy']): Loan[] {
  const alive = loans.filter((l) => l.balance > 1e-6);
  if (strategy === 'avalanche') return alive.sort((a, b) => (b.apr ?? 0) - (a.apr ?? 0));
  if (strategy === 'snowball') return alive.sort((a, b) => a.balance - b.balance);
  return alive.sort((a, b) => a.name.localeCompare(b.name));
}

function buildPlan(input: PlanInput): PlanResult {
  const loans = input.loans.map((l) => ({ ...l }));
  const schedule: PlanResult['schedule'] = [];
  let months = 0,
    totalInterest = 0;
  const cap = 600;

  while (months < cap && loans.some((l) => l.balance > 1e-6)) {
    months += 1;

    const mins = loans.map((l) => ({ id: l.id ?? l.name, due: Math.min(minDue(l), l.balance) }));
    const sumMins = mins.reduce((s, m) => s + m.due, 0);
    if (sumMins - 1e-6 > input.monthlyBudget) break;

    let extra = input.monthlyBudget - sumMins;
    const priority = orderLoans(loans, input.strategy);
    const extraMap: Record<string, number> = {};
    priority.forEach((l) => (extraMap[l.id ?? l.name] = 0));

    if (input.strategy === 'equal') {
      const actives = priority.filter((l) => l.balance > 1e-6);
      actives.forEach((l) => (extraMap[l.id ?? l.name] = extra / actives.length));
      extra = 0;
    } else {
      for (const l of priority) {
        if (extra <= 1e-6) break;
        const key = l.id ?? l.name;
        const minForThis = mins.find((m) => m.id === key)?.due ?? 0;
        const capPay = Math.max(l.balance - minForThis, 0);
        const pay = Math.min(extra, capPay);
        extraMap[key] = pay;
        extra -= pay;
      }
    }

    const items: PlanResult['schedule'][number]['items'] = [];
    let monthInterest = 0,
      monthPaid = 0;

    for (let i = 0; i < loans.length; i++) {
      const l = loans[i];
      if (l.balance <= 1e-6) {
        items.push({ id: l.id ?? l.name, interest: 0, principal: 0, endBalance: 0 });
        continue;
      }

      const r = monthlyRate(l.apr);
      const interest = l.balance * r;
      let payment = (mins[i].due || 0) + (extraMap[l.id ?? l.name] || 0);

      if ((!l.type || l.type === 'revolving' || l.type === 'credit_card') && payment < interest) payment = interest;

      payment = Math.min(payment, l.balance + interest);
      const principal = payment - interest;
      l.balance = Math.max(l.balance + interest - payment, 0);

      if (l.termMonths && l.termMonths > 0 && principal > 0) l.termMonths = Math.max(l.termMonths - 1, 0);

      monthInterest += interest;
      monthPaid += payment;
      items.push({
        id: l.id ?? l.name,
        interest: Number(interest.toFixed(2)),
        principal: Number(principal.toFixed(2)),
        endBalance: Number(l.balance.toFixed(2)),
      });
    }

    totalInterest += monthInterest;
    schedule.push({
      monthIndex: months - 1,
      items,
      totalInterest: Number(monthInterest.toFixed(2)),
      totalPaid: Number(monthPaid.toFixed(2)),
    });
  }

  const payoffMonthByLoan: Record<string, number> = {};
  for (const l of input.loans) {
    const key = l.id ?? l.name;
    const lastZero = [...schedule]
      .reverse()
      .find((s) => (s.items.find((it) => it.id === key)?.endBalance ?? 1) === 0);
    payoffMonthByLoan[key] = lastZero ? lastZero.monthIndex : schedule.length - 1;
  }

  return {
    schedule,
    payoffMonthByLoan,
    totalInterest: Number(totalInterest.toFixed(2)),
    monthsToDebtFree: schedule.length,
  };
}

function sumMinimums(loans: Loan[]): number {
  return loans.reduce((s, l) => s + Math.min(minDue(l), l.balance), 0);
}

/* ===========================
   Component
=========================== */
const LoansPage: React.FC = () => {
  const [loans, setLoans] = useState<Loan[]>([]);
  const [proposals, setProposals] = useState<Loan[]>([]);
  const [loading, setLoading] = useState(false);
  const [strategy, setStrategy] = useState<'avalanche' | 'snowball' | 'equal'>('avalanche');
  const [extra, setExtra] = useState<number>(0);
  const [grossMonthlyIncome, setGrossMonthlyIncome] = useState<number>(0);
  const [error, setError] = useState<string>('');

  // Load current loans
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${API}/loans`);
        const data = await r.json();
        const normalized: Loan[] = (data || []).map((d: any) => ({
          id: d.id || d.name,
          name: d.name,
          type: (d.type as DebtType) || 'loan',
          balance: Number(d.balance ?? 0),
          apr: d.rate ?? d.apr ?? null,
          minPayment:
            d.payment != null ? Number(d.payment) : d.minPayment != null ? Number(d.minPayment) : null,
          termMonths: d.termMonths ?? null,
          source: d.source ?? 'manual',
        }));
        setLoans(normalized);
      } catch (e) {
        console.error(e);
      }
    })();
  }, []);

  // Stats
  const totalDebt = useMemo(() => loans.reduce((s, l) => s + (l.balance || 0), 0), [loans]);
  const monthlyMins = useMemo(() => sumMinimums(loans), [loans]);
  const monthlyBudget = monthlyMins + Math.max(extra, 0);
  const dti = grossMonthlyIncome ? (monthlyMins / grossMonthlyIncome) * 100 : 0;

  // Plans
  const basePlan = useMemo(
    () => (loans.length ? buildPlan({ loans, monthlyBudget: monthlyMins, strategy }) : null),
    [loans, monthlyMins, strategy]
  );
  const extraPlan = useMemo(
    () => (loans.length ? buildPlan({ loans, monthlyBudget, strategy }) : null),
    [loans, monthlyBudget, strategy]
  );

  const recommended = useMemo(() => {
    if (!loans.length) return null;
    const budgets = monthlyBudget;
    const opts = ['avalanche', 'snowball', 'equal'] as const;
    const scores = opts.map((s) => ({ s, res: buildPlan({ loans, monthlyBudget: budgets, strategy: s }) }));
    scores.sort((a, b) => a.res.totalInterest - b.res.totalInterest);
    return scores[0].s as 'avalanche' | 'snowball' | 'equal';
  }, [loans, monthlyBudget]);

  const interestSaved = useMemo(
    () => (!basePlan || !extraPlan ? 0 : Math.max(0, basePlan.totalInterest - extraPlan.totalInterest)),
    [basePlan, extraPlan]
  );
  const termReduced = useMemo(
    () => (!basePlan || !extraPlan ? 0 : Math.max(0, basePlan.monthsToDebtFree - extraPlan.monthsToDebtFree)),
    [basePlan, extraPlan]
  );

  // Chart data
  const chartData = useMemo(() => {
    if (!basePlan || !extraPlan) return [];
    const toMap = (plan: PlanResult) => {
      const m = new Map<number, number>();
      for (const s of plan.schedule) {
        const total = s.items.reduce((sum, it) => sum + (it.endBalance || 0), 0);
        m.set(s.monthIndex, total);
      }
      return m;
    };
    const base = toMap(basePlan);
    const withExtra = toMap(extraPlan);
    const maxMonth = Math.max(basePlan.schedule.length, extraPlan.schedule.length);
    const data: Array<{ label: string; base: number | null; with: number | null }> = [];
    for (let i = 0; i < maxMonth; i++) {
      data.push({ label: `M${i + 1}`, base: base.get(i) ?? null, with: withExtra.get(i) ?? null });
    }
    return data;
  }, [basePlan, extraPlan]);

  /* ---- Actions ---- */
  async function importFromBank() {
    try {
      setLoading(true);
      setError('');
      const r = await fetch(`${API}/loans/detect`, { method: 'POST' });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      const normalized: Loan[] = (data || []).map((d: any) => ({
        id: d.id || d.name,
        name: d.name,
        type: (d.type as DebtType) || 'loan',
        balance: Number(d.balance ?? 0),
        apr: d.rate ?? d.apr ?? null,
        minPayment: d.minPayment != null ? Number(d.minPayment) : null,
        termMonths: d.termMonths ?? null,
        source: d.source ?? 'bank_ledger',
      }));
      setProposals(normalized);
    } catch (e: any) {
      setError(e?.message || 'Failed to scan bank data');
    } finally {
      setLoading(false);
    }
  }

  async function saveProposal(p: Loan) {
    if (!p.balance || p.balance <= 0) return setError('Please enter a balance for ' + p.name);
    try {
      setError('');
      const body = {
        name: p.name,
        type: p.type,
        balance: Number(p.balance),
        apr: p.apr,
        payment: p.minPayment,
        termMonths: p.termMonths,
        source: p.source || 'bank_ledger',
      };
      const resp = await fetch(`${API}/loans`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(`POST /loans ${resp.status}: ${txt || 'server error'}`);
      }
      setLoans((prev) => {
        const next = prev.filter((x) => x.name !== p.name);
        next.push({
          id: p.id || p.name,
          name: p.name,
          type: p.type,
          balance: Number(p.balance),
          apr: p.apr ?? null,
          minPayment: p.minPayment ?? null,
          termMonths: p.termMonths ?? null,
          source: p.source || 'bank_ledger',
        });
        return next;
      });
      setProposals((prev) => prev.filter((x) => x.name !== p.name));
    } catch (e: any) {
      setError(e?.message || 'Failed to save loan');
    }
  }

  async function saveAll() {
    for (const p of proposals) {
      if (ready(p)) {
        // eslint-disable-next-line no-await-in-loop
        await saveProposal(p);
      }
    }
  }

  function updateLoanInline(idx: number, patch: Partial<Loan>) {
    setLoans((prev) => {
      const next = [...prev];
      next[idx] = { ...next[idx], ...patch } as Loan;
      return next;
    });
  }

  async function saveLoanInline(l: Loan) {
    const body = {
      name: l.name,
      type: l.type,
      balance: Number(l.balance),
      apr: l.apr,
      payment: l.minPayment,
      termMonths: l.termMonths,
      source: l.source || 'manual',
    };
    try {
      const resp = await fetch(`${API}/loans`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) console.warn('Save inline failed', await resp.text());
    } catch (e) {
      console.error(e);
    }
  }

  const [newLoan, setNewLoan] = useState<Loan>({
    name: '',
    type: 'loan',
    balance: 0,
    apr: null,
    minPayment: 0,
    termMonths: null,
    source: 'manual',
  });
  const canAdd =
    newLoan.name.trim().length > 0 &&
    (newLoan.balance || 0) > 0 &&
    ((newLoan.minPayment || 0) > 0 || (newLoan.termMonths || 0) > 0);

  async function addManualLoan() {
    if (!canAdd) return;
    const payload: Loan = { ...newLoan };
    if ((!payload.minPayment || payload.minPayment === 0) && (payload.termMonths || 0) > 0) {
      const r = payload.apr ? payload.apr / 100 / 12 : 0;
      payload.minPayment =
        r === 0
          ? payload.balance / (payload.termMonths as number)
          : (payload.balance * r) / (1 - Math.pow(1 + r, -(payload.termMonths as number)));
    }
    await saveProposal(payload);
    setNewLoan({ name: '', type: 'loan', balance: 0, apr: null, minPayment: 0, termMonths: null, source: 'manual' });
  }

  async function uploadStatement(i: number, file: File) {
    if (!file) return;
    try {
      const p = proposals[i];
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch(`${API}/loan-docs/parse?name=${encodeURIComponent(p.name)}`, {
        method: 'POST',
        body: fd,
      });
      const data = await res.json();
      setProposals((prev) => {
        const n = [...prev];
        n[i] = {
          ...n[i],
          apr: data.apr ?? n[i].apr,
          balance: data.balance ?? n[i].balance,
          minPayment: data.minPayment ?? n[i].minPayment,
          termMonths: data.termMonths ?? n[i].termMonths,
        };
        return n;
      });
    } catch (e) {
      console.error(e);
    }
  }

  /* ===========================
     UI
  =========================== */
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Navigation />

      <main className="flex-1">
        <div className="max-w-6xl mx-auto px-4 py-8">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-semibold tracking-tight">Optimize Your Loans</h1>
              <p className="text-sm text-gray-500">
                Smarter payoff strategies powered by simple math (and a bit of AI).
              </p>
            </div>
            <Button onClick={importFromBank} disabled={loading} className="gap-2">
              <ScanSearch className="size-4" />
              {loading ? 'Scanning…' : 'Import from Bank Statement'}
            </Button>
          </div>

          {error && (
            <Card className="mb-6 border-red-200 bg-red-50">
              <CardContent className="p-3 text-sm text-red-700"> {error} </CardContent>
            </Card>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Current Loans */}
            <Card className="lg:col-span-2">
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Current Loans</CardTitle>
                <CardDescription>Inline edit any field. Changes save on blur.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {loans.length === 0 && (
                  <div className="text-sm text-gray-500">
                    No loans yet. Import from your bank statement or add one below.
                  </div>
                )}

                {loans.map((l, idx) => (
                  <div key={l.id ?? l.name} className="grid sm:grid-cols-12 items-center gap-3">
                    <div className="sm:col-span-3">
                      <div className="font-medium truncate">{l.name}</div>
                      <div className="text-xs text-gray-500">{l.type ?? 'loan'}</div>
                    </div>

                    <div className="sm:col-span-9 grid grid-cols-2 sm:grid-cols-8 gap-3">
                      <div className="col-span-2 sm:col-span-2">
                        <Label className="text-xs text-gray-500">Balance</Label>
                        <Input
                          type="number"
                          value={l.balance || ''}
                          placeholder="—"
                          onChange={(e) => updateLoanInline(idx, { balance: Number(e.target.value || 0) })}
                          onBlur={() => saveLoanInline(loans[idx])}
                        />
                      </div>

                      <div className="col-span-1">
                        <Label className="text-xs text-gray-500">APR %</Label>
                        <Input
                          type="number"
                          step="0.01"
                          value={l.apr ?? ''}
                          placeholder="—"
                          onChange={(e) =>
                            updateLoanInline(idx, { apr: e.target.value === '' ? null : Number(e.target.value) })
                          }
                          onBlur={() => saveLoanInline(loans[idx])}
                        />
                      </div>

                      <div className="col-span-2 sm:col-span-2">
                        <Label className="text-xs text-gray-500">Min $</Label>
                        <Input
                          type="number"
                          value={l.minPayment ?? ''}
                          placeholder="—"
                          onChange={(e) =>
                            updateLoanInline(idx, {
                              minPayment: e.target.value === '' ? null : Number(e.target.value),
                            })
                          }
                          onBlur={() => saveLoanInline(loans[idx])}
                        />
                      </div>

                      <div className="col-span-1">
                        <Label className="text-xs text-gray-500">Term</Label>
                        <Input
                          type="number"
                          value={l.termMonths ?? ''}
                          placeholder="mo"
                          onChange={(e) =>
                            updateLoanInline(idx, { termMonths: e.target.value === '' ? null : Number(e.target.value) })
                          }
                          onBlur={() => saveLoanInline(loans[idx])}
                        />
                      </div>
                    </div>
                  </div>
                ))}

                {/* Add manual */}
                <div className="rounded-xl border bg-gray-50 p-4">
                  <div className="text-sm font-medium mb-3">Add New Loan</div>
                  <div className="grid grid-cols-1 sm:grid-cols-12 gap-3">
                    <div className="sm:col-span-3">
                      <Label className="text-xs">Name</Label>
                      <Input
                        placeholder="e.g. Chase Card"
                        value={newLoan.name}
                        onChange={(e) => setNewLoan({ ...newLoan, name: e.target.value })}
                      />
                    </div>

                    <div className="sm:col-span-2">
                      <Label className="text-xs">Type</Label>
                      <Select value={newLoan.type} onValueChange={(v) => setNewLoan({ ...newLoan, type: v as DebtType })}>
                        <SelectTrigger>
                          <SelectValue placeholder="Loan" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="loan">Loan</SelectItem>
                          <SelectItem value="credit_card">Credit Card</SelectItem>
                          <SelectItem value="installment">Installment</SelectItem>
                          <SelectItem value="auto">Auto</SelectItem>
                          <SelectItem value="mortgage">Mortgage</SelectItem>
                          <SelectItem value="student">Student</SelectItem>
                          <SelectItem value="bnpl">BNPL</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="sm:col-span-2">
                      <Label className="text-xs">Balance ($)</Label>
                      <Input
                        type="number"
                        placeholder="e.g. 3500"
                        value={newLoan.balance ? newLoan.balance : ''}
                        onChange={(e) =>
                          setNewLoan({ ...newLoan, balance: e.target.value === '' ? 0 : Number(e.target.value) })
                        }
                      />
                    </div>

                    <div className="sm:col-span-2">
                      <Label className="text-xs">APR %</Label>
                      <Input
                        type="number"
                        step="0.01"
                        placeholder="e.g. 19.99"
                        value={newLoan.apr ?? ''}
                        onChange={(e) =>
                          setNewLoan({ ...newLoan, apr: e.target.value === '' ? null : Number(e.target.value) })
                        }
                      />
                    </div>

                    <div className="sm:col-span-2">
                      <Label className="text-xs">Min payment ($/mo)</Label>
                      <Input
                        type="number"
                        placeholder="or enter term"
                        value={newLoan.minPayment ? newLoan.minPayment : ''}
                        onChange={(e) =>
                          setNewLoan({
                            ...newLoan,
                            minPayment: e.target.value === '' ? 0 : Number(e.target.value),
                          })
                        }
                      />
                    </div>

                    <div className="sm:col-span-1">
                      <Label className="text-xs">Term (mo)</Label>
                      <Input
                        type="number"
                        placeholder="e.g. 36"
                        value={newLoan.termMonths ?? ''}
                        onChange={(e) =>
                          setNewLoan({
                            ...newLoan,
                            termMonths: e.target.value === '' ? null : Number(e.target.value),
                          })
                        }
                      />
                    </div>
                  </div>

                  <div className="mt-3 flex items-center justify-between">
                    <p className="text-xs text-gray-500">
                      Tip: if you don’t know the exact minimum, enter balance, APR and a{' '}
                      <span className="font-medium">term</span>. We’ll estimate the monthly payment.
                    </p>
                    <Button onClick={addManualLoan} disabled={!canAdd}>
                      Add
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Quick Stats */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Quick Stats</CardTitle>
                <CardDescription>Snapshot of your debt picture.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-gray-500">Total Debt</span>
                  <span className="font-semibold">{fmt(totalDebt)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-500">Monthly Mins</span>
                  <span className="font-semibold">{fmt2(monthlyMins)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-500">DTI</span>
                  <span
                    className={`font-semibold ${
                      dti > 43 ? 'text-red-600' : dti > 36 ? 'text-amber-600' : 'text-gray-900'
                    }`}
                  >
                    {grossMonthlyIncome ? dti.toFixed(1) + '%' : '—'}
                  </span>
                </div>
                <div>
                  <Label className="text-xs text-gray-500">Enter income to compute DTI</Label>
                  <Input
                    type="number"
                    placeholder="Gross monthly income"
                    value={grossMonthlyIncome || ''}
                    onChange={(e) => setGrossMonthlyIncome(Number(e.target.value || 0))}
                    className="mt-1"
                  />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Strategy + Chart */}
          <div className="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Repayment Strategies</CardTitle>
                <CardDescription>Select a rule to prioritize extra payments.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-2">
                  {(['avalanche','snowball','equal'] as const).map((s) => (
                    <Button
                      key={s}
                      variant={strategy===s ? 'default' : 'outline'}
                      className="w-full justify-start h-auto py-2 px-3 flex-col items-start gap-1 min-h-[56px]
                                whitespace-normal text-left break-words leading-tight"
                      onClick={() => setStrategy(s)}
                    >
                      <span className="font-medium capitalize">{
                        s === 'avalanche' ? 'Avalanche' : s === 'snowball' ? 'Snowball' : 'Equal'
                      }</span>

                      <span className="text-[11px] opacity-80 leading-tight">
                        {s === 'avalanche'
                          ? 'Pay highest APR first'
                          : s === 'snowball'
                          ? 'Pay smallest balance first'
                          : 'Split extras equally'}
                      </span>
                    </Button>
                  ))}
                </div>

                <div className="mt-4">
                  <Label className="text-sm font-medium flex items-center gap-2">
                    <PiggyBank className="size-4" /> Extra payment per month
                  </Label>
                  <div className="mt-3 flex items-center gap-3">
                    <Slider
                      value={[extra]}
                      min={0}
                      max={Math.max(2000, Math.ceil(monthlyMins))}
                      step={5}
                      onValueChange={(v) => setExtra(v[0] ?? 0)}
                      className="w-full"
                    />
                    <Input type="number" className="w-28" value={extra} onChange={(e) => setExtra(Number(e.target.value || 0))} />
                  </div>
                  <p className="text-xs text-gray-500 mt-2">
                    Budget used: {fmt2(monthlyBudget)} (mins {fmt2(monthlyMins)} + extra {fmt2(extra)})
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card className="lg:col-span-2">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">AI Suggestion</CardTitle>
                  {loans.length > 0 && (
                    <Badge
                      variant="outline"
                      className="text-xs h-7 leading-7 whitespace-nowrap overflow-hidden text-ellipsis max-w-full"
                    >
                      Comparing Minimum only vs <span className="ml-1 capitalize">{strategy}</span> + extra ({fmt2(extra)}/mo)
                    </Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                {!basePlan || !extraPlan ? (
                  <p className="text-sm text-gray-500">Add at least one loan to see suggestions.</p>
                ) : (
                  <>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                      <div className="rounded-lg bg-indigo-50 border border-indigo-100 p-3 min-h-[72px] flex flex-col justify-center">
                        <div className="text-xs text-indigo-700 flex items-center gap-1">
                          <TrendingDown className="size-3" /> Interest saved
                        </div>
                        <div className="text-lg font-semibold">{fmt2(interestSaved)}</div>
                      </div>
                      <div className="rounded-lg bg-emerald-50 border border-emerald-100 p-3 min-h-[72px] flex flex-col justify-center">
                        <div className="text-xs text-emerald-700">Debt-free sooner by</div>
                        <div className="text-lg font-semibold">{termReduced} months</div>
                      </div>
                      <div className="rounded-lg bg-gray-50 border p-3 min-h-[72px] flex flex-col justify-center">
                        <div className="text-xs text-gray-600">Best strategy at this budget</div>
                        <div className="text-lg font-semibold capitalize">{recommended || strategy}</div>
                      </div>
                    </div>

                    <div className="mt-4">
                      <div className="w-full h-[280px]">
                        <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
                          <LineChart data={chartData} margin={{ top: 8, right: 12, left: 8, bottom: 8 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="label" tickMargin={6} />
                            <YAxis
                              width={YAXIS_WIDTH}
                              tickFormatter={(v) => '$' + Math.round(Number(v)).toLocaleString()}
                              allowDecimals={false}
                            />
                            <Tooltip formatter={(v) => fmt2(Number(v))} />
                            <Legend wrapperStyle={{ whiteSpace: 'nowrap' }} />
                            <Line
                              isAnimationActive={false}
                              type="monotone"
                              dataKey="base"
                              name="Minimum only"
                              dot={false}
                              strokeWidth={2}
                            />
                            <Line
                              isAnimationActive={false}
                              type="monotone"
                              dataKey="with"
                              name="With extra"
                              dot={false}
                              strokeWidth={2}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Detected from bank ledger */}
          {proposals.length > 0 && (
            <Card className="mt-6">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-base">Detected recurring payments</CardTitle>
                    <CardDescription>
                      Enter remaining balance & APR from your statement. Term is optional for installment loans.
                    </CardDescription>
                  </div>
                  <Button variant="secondary" onClick={saveAll} disabled={!proposals.some(ready)} className="gap-2">
                    <Save className="size-4" /> Save all
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                {proposals.map((p, i) => (
                  <div key={p.id ?? p.name} className="rounded-xl border p-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium">{p.name}</div>
                        <div className="text-xs text-gray-500">
                          Typical payment: {p.minPayment ? fmt2(p.minPayment) : '—'} / mo • Source: {p.source || 'bank_ledger'}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <input
                          id={`pdf-${i}`}
                          type="file"
                          accept="application/pdf"
                          className="hidden"
                          onChange={(e) => e.target.files && uploadStatement(i, e.target.files[0])}
                        />
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => document.getElementById(`pdf-${i}`)?.click()}
                          className="gap-2"
                        >
                          <FileUp className="size-4" /> Auto-fill from statement
                        </Button>
                      </div>
                    </div>

                    <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-3">
                      <div>
                        <Label className="text-xs text-gray-600">Balance ($)</Label>
                        <Input
                          type="number"
                          placeholder="e.g. 3500"
                          value={p.balance === 0 || p.balance == null ? '' : p.balance}
                          onChange={(e) =>
                            setProposals((prev) => {
                              const n = [...prev];
                              n[i] = { ...n[i], balance: e.target.value === '' ? 0 : Number(e.target.value) };
                              return n;
                            })
                          }
                        />
                      </div>

                      <div>
                        <Label className="text-xs text-gray-600">APR %</Label>
                        <Input
                          type="number"
                          step="0.01"
                          placeholder="e.g. 19.99"
                          value={p.apr ?? ''}
                          onChange={(e) =>
                            setProposals((prev) => {
                              const n = [...prev];
                              n[i] = { ...n[i], apr: e.target.value === '' ? null : Number(e.target.value) };
                              return n;
                            })
                          }
                        />
                      </div>

                      <div>
                        <Label className="text-xs text-gray-600">Min payment ($/mo)</Label>
                        <Input
                          type="number"
                          placeholder="auto-filled"
                          value={p.minPayment ?? ''}
                          onChange={(e) =>
                            setProposals((prev) => {
                              const n = [...prev];
                              n[i] = { ...n[i], minPayment: e.target.value === '' ? null : Number(e.target.value) };
                              return n;
                            })
                          }
                        />
                      </div>

                      <div>
                        <Label className="text-xs text-gray-600">Term (months, optional)</Label>
                        <Input
                          type="number"
                          placeholder="e.g. 36"
                          value={p.termMonths ?? ''}
                          onChange={(e) =>
                            setProposals((prev) => {
                              const n = [...prev];
                              n[i] = { ...n[i], termMonths: e.target.value === '' ? null : Number(e.target.value) };
                              return n;
                            })
                          }
                        />
                      </div>
                    </div>

                    <div className="mt-3 flex justify-end">
                      <Button onClick={() => saveProposal(p)} disabled={!ready(p)}>
                        Save
                      </Button>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default LoansPage;
