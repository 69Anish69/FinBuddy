'use client';


import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import React, { useEffect, useState } from "react";

type PlanGoal = {
  name: string;
  target: string;
  year: number;
  monthly: string;
  monthly_raw?: number;
  current: string;
  progress: number;
  feasible?: boolean;
  status?: "On Track" | "Behind";
  surplus?: string;
  affordable?: boolean;
  over_by?: number;
};

type EmergencyHint = {
  avg_monthly_expense: string;
  months: number;
  recommended: string;
  raw?: { avg_exp: number; ef_target: number };
};

type Budget = {
  surplus_raw: number;
  used_raw: number;
  remaining_raw: number;
  surplus: string;
  used: string;
  remaining: string;
};

const Goals = () => {
  const [selectedItem, setSelectedItem] = useState("Car");
  const [price, setPrice] = useState("");
  const [year, setYear] = useState("");
  const [goals, setGoals] = useState<{ name: string; price: number; year: number }[]>([]);
  const [planResults, setPlanResults] = useState<PlanGoal[]>([]);
  const [efHint, setEfHint] = useState<EmergencyHint | null>(null);
  const [budget, setBudget] = useState<Budget | null>(null);

  const currentYear = new Date().getFullYear();

// ADD BELOW existing useState lines
const [editingIdx, setEditingIdx] = useState<number | null>(null);
const [editPrice, setEditPrice] = useState<string>("");
const [editYear, setEditYear] = useState<string>("");


useEffect(() => {
  const onAdd = (e: any) => {
    const { name, price, year } = e.detail || {};
    if (!name || !price || !year) return;
    setSelectedItem(name);
    setPrice(String(price));
    setYear(String(year));
    // Give state a tick, then reuse your existing add handler
    setTimeout(() => handleAddToPlan(), 0);
  };

  const onEdit = async (e: any) => {
    const { name, price, year } = e.detail || {};
    if (!name) return;
    const i = goals.findIndex((g) => g.name.toLowerCase() === name.toLowerCase());
    if (i < 0) return;
    setEditingIdx(i);
    setEditPrice(String(price ?? goals[i].price));
    setEditYear(String(year ?? goals[i].year));
    setTimeout(() => handleEditSave(i), 0);
  };

  const onDelete = async (e: any) => {
    const { name } = e.detail || {};
    if (!name) return;
    const i = goals.findIndex((g) => g.name.toLowerCase() === name.toLowerCase());
    if (i < 0) return;
    handleDelete(i);
  };

  window.addEventListener("voice:addGoal", onAdd as EventListener);
  window.addEventListener("voice:editGoal", onEdit as EventListener);
  window.addEventListener("voice:deleteGoal", onDelete as EventListener);
  return () => {
    window.removeEventListener("voice:addGoal", onAdd as EventListener);
    window.removeEventListener("voice:editGoal", onEdit as EventListener);
    window.removeEventListener("voice:deleteGoal", onDelete as EventListener);
  };
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, [goals]); // depends on goals for edit/delete lookup


// Recompute plan (shared by edit/delete)
const recomputePlan = async (updatedGoals: { name: string; price: number; year: number }[]) => {
  setGoals(updatedGoals);
  try {
    const res = await fetch("http://localhost:8000/goals/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ goals: updatedGoals }),
    });
    const data = await res.json();
    if (data.goals) setPlanResults(data.goals as PlanGoal[]);
    if (data.budget) setBudget(data.budget as Budget);
    if (data.emergency) setEfHint(data.emergency as EmergencyHint);
  } catch (e) {
    console.error("Failed to update plan:", e);
    alert("Failed to update plan. Check backend and CORS.");
  }
};

// Handlers
const handleEditStart = (i: number) => {
  const g = goals[i];
  if (!g) return;
  setEditingIdx(i);
  setEditPrice(String(g.price));
  setEditYear(String(g.year));
};

const handleEditCancel = () => {
  setEditingIdx(null);
  setEditPrice("");
  setEditYear("");
};

const handleEditSave = async (i: number) => {
  const priceN = parseFloat(editPrice);
  const yearN = parseInt(editYear);
  if (!isFinite(priceN) || !isFinite(yearN)) {
    alert("Enter a valid number for price and year.");
    return;
  }
  const updated = goals.map((g, idx) => (idx === i ? { ...g, price: priceN, year: yearN } : g));
  await recomputePlan(updated);
  handleEditCancel();
};

const handleDelete = async (i: number) => {
  const updated = goals.filter((_, idx) => idx !== i);
  await recomputePlan(updated);
};


  const handleAddToPlan = async () => {
    if (!price || !year) {
      alert("Please enter price and year");
      return;
    }

    const newGoal = {
      name: selectedItem,
      price: parseFloat(price),
      year: parseInt(year),
    };

    // Client-side affordability guard before hitting backend
    const monthsLeft = Math.max(1, newGoal.year - currentYear) * 12;
    const monthlyNeeded = +(newGoal.price / monthsLeft).toFixed(2);
    if (budget && monthlyNeeded > budget.remaining_raw) {
      const over = (monthlyNeeded - budget.remaining_raw).toFixed(2);
      alert(
        `This goal needs $${monthlyNeeded}/mo but only $${budget.remaining_raw}/mo is available. You are over by $${over}/mo.\n` +
        `Try delaying the deadline or reducing the target.`
      );
      return;
    }

    const updatedGoals = [...goals, newGoal];
    setGoals(updatedGoals);

    try {
      const response = await fetch("http://localhost:8000/goals/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goals: updatedGoals }),
      });

      const data = await response.json();
      if (data.goals) {
        setPlanResults(data.goals as PlanGoal[]);
      }
      if (data.budget) {
        setBudget(data.budget as Budget);
      }
      if (data.emergency) {
        setEfHint(data.emergency as EmergencyHint);
      }
      setPrice("");
      setYear("");
    } catch (error) {
      console.error("Failed to add goals:", error);
      alert("Failed to add goal. Check backend is running on :8000 and CORS.");
    }
  };

  const getSmartSuggestions = (goal: PlanGoal) => {
    if (goal.affordable === false || goal.feasible === false) {
      return [
        "Delay goal by ~1 year to reduce monthly pressure",
        "Cut subscriptions to add ~$100/month",
        "Refinance high-interest debt to free cash flow",
      ];
    }
    return ["Automate monthly transfers", "Use a cashback card for routine spend"];
  };

  // Allow EF always. For other goals, require EF 100% *after* an EF exists.
  const isGoalEnabled = (goalName: string) => {
    if (goalName === "Emergency Fund") return true;
    const emergencyFund = planResults.find((g) => g.name === "Emergency Fund");
    if (!emergencyFund) return true;
    return (emergencyFund.progress ?? 0) >= 100;
  };

  const canAffordSelected = () => {
    if (!budget || !price || !year) return true;
    const monthsLeft = Math.max(1, parseInt(year) - currentYear) * 12;
    const need = +(parseFloat(price) / monthsLeft).toFixed(2);
    return need <= budget.remaining_raw + 1e-9;
  };

  const handlePDFDownload = () => {
    alert("🚧 PDF export coming soon (placeholder)!");
  };

  return (
    <div className="min-h-screen bg-gradient-subtle flex flex-col">
      <Navigation />
      <main className="flex-grow container mx-auto px-4 py-16">
        <h1 className="text-4xl font-bold text-center mb-2">🎯 Your Goal Planner</h1>
        <p className="text-center text-muted-foreground mb-10">Let AI help you save smarter and faster</p>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Add Goal */}
          <div className="bg-card p-6 rounded-lg border shadow-card-hover">
            <h3 className="text-lg font-semibold mb-4">➕ Add New Goal</h3>

            {/* EF Recommender — compact, single line */}
            {efHint && (
              <div className="rounded-md border p-3 mb-4 bg-muted/30">
                <div className="text-sm whitespace-nowrap overflow-hidden text-ellipsis">
                  <strong>Recommended Emergency Fund:</strong>{" "}
                  {efHint.recommended} • {efHint.months} mo @ {efHint.avg_monthly_expense}/mo
                </div>
                <button
                  className="mt-2 text-sm px-3 py-2 rounded-md bg-primary text-white"
                  onClick={() => {
                    setSelectedItem("Emergency Fund");
                    const amt = efHint.raw?.ef_target ?? 0;
                    if (amt > 0) setPrice(String(Math.round(amt)));
                    setYear(String(currentYear + 1));
                  }}
                >
                  One‑click add Emergency Fund goal
                </button>
              </div>
            )}

            <div className="space-y-4">
              <select
                value={selectedItem}
                onChange={(e) => setSelectedItem(e.target.value)}
                className="w-full px-3 py-2 border border-input rounded-md"
              >
                <option>Car</option>
                <option>House</option>
                <option>Vacation</option>
                <option>Education</option>
                <option>Wedding</option>
                <option>Emergency Fund</option>
              </select>

              <input
                type="number"
                value={price}
                onChange={(e) => setPrice(e.target.value)}
                placeholder="$50,000"
                className="w-full px-3 py-2 border border-input rounded-md"
              />
              <input
                type="number"
                value={year}
                onChange={(e) => setYear(e.target.value)}
                placeholder={`${currentYear + 1}`}
                className="w-full px-3 py-2 border border-input rounded-md"
              />

              <button
                onClick={handleAddToPlan}
                className="w-full bg-gradient-financial text-white py-3 rounded-md font-semibold disabled:opacity-60"
                disabled={!isGoalEnabled(selectedItem) || !canAffordSelected()}
              >
                {isGoalEnabled(selectedItem)
                  ? (canAffordSelected() ? "Add to Your Plan" : "Over Budget – Adjust Goal")
                  : "Complete Emergency Fund First"}
              </button>

              {!isGoalEnabled(selectedItem) && (
                <p className="text-sm text-muted-foreground mt-2">
                  Complete your <strong>Emergency Fund</strong> (100%) to unlock {selectedItem}.
                </p>
              )}

              {budget && !canAffordSelected() && price && year && (
                <p className="text-sm text-muted-foreground mt-2">
                  Available: <strong>{budget.remaining}</strong> per month. This goal needs{" "}
                  <strong>
                    $
                    {(
                      parseFloat(price) /
                      (Math.max(1, parseInt(year) - currentYear) * 12)
                    ).toFixed(2)}
                    /mo
                  </strong>
                  . Try reducing the target or extending the deadline.
                </p>
              )}

              {budget && (
                <div className="text-xs text-muted-foreground mt-2">
                  Budget summary — Used: <strong>{budget.used}</strong>, Remaining:{" "}
                  <strong>{budget.remaining}</strong> (of {budget.surplus} /mo)
                </div>
              )}
            </div>
          </div>

          {/* Plan Details */}
          <div className="bg-card p-6 rounded-lg border shadow-card-hover">
            <h3 className="text-lg font-semibold mb-4">📋 Your Savings Plan</h3>
            <div className="space-y-4">
              {planResults.map((goal, idx) => (
                <div key={idx} className="p-4 bg-gradient-subtle rounded-lg">
                  {/* Header + actions */}
                  <div className="flex items-start justify-between gap-2">
                    <h4 className="font-medium text-primary mb-1">
                      {goal.name} {!goal.affordable && <span className="text-red-500 text-xs">(over budget)</span>}
                    </h4>
                    <div className="shrink-0 space-x-2">
                      {editingIdx === idx ? (
                        <>
                          <button
                            className="px-2 py-1 text-xs rounded bg-emerald-600 text-white"
                            onClick={() => handleEditSave(idx)}
                          >
                            Save
                          </button>
                          <button
                            className="px-2 py-1 text-xs rounded bg-zinc-200"
                            onClick={handleEditCancel}
                          >
                            Cancel
                          </button>
                        </>
                      ) : (
                        <>
                          <button
                            className="px-2 py-1 text-xs rounded bg-zinc-200"
                            onClick={() => handleEditStart(idx)}
                          >
                            Edit
                          </button>
                          <button
                            className="px-2 py-1 text-xs rounded bg-red-600 text-white"
                            onClick={() => handleDelete(idx)}
                          >
                            Delete
                          </button>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Details */}
                  {editingIdx === idx ? (
                    // --- Edit mode ---
                    <div className="mt-2 grid grid-cols-2 gap-3">
                      <div>
                        <label className="text-xs text-muted-foreground">Price</label>
                        <input
                          type="number"
                          value={editPrice}
                          onChange={(e) => setEditPrice(e.target.value)}
                          className="mt-1 w-full px-3 py-2 border border-input rounded-md"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-muted-foreground">Year</label>
                        <input
                          type="number"
                          value={editYear}
                          onChange={(e) => setEditYear(e.target.value)}
                          className="mt-1 w-full px-3 py-2 border border-input rounded-md"
                        />
                      </div>
                    </div>
                  ) : (
                    // --- Read mode ---
                    <>
                      <p className="text-sm text-muted-foreground">
                        Target: {goal.target} by {goal.year}
                        <br />
                        Monthly Needed: <strong>{goal.monthly}</strong>
                        <br />
                        Avg Monthly Surplus (from your file): <strong>{goal.surplus ?? "$0.00"}</strong>
                        <br />
                        Progress: {goal.current} ({Math.round(goal.progress || 0)}%)
                        <br />
                        Feasibility: {goal.feasible ? "🟢 On Track" : "🔴 Behind"} ({goal.status})
                      </p>

                      {/* Progress bar (clamped + clipped) */}
                      {(() => {
                        const pct = Math.min(100, Math.max(0, Number(goal.progress) || 0));
                        return (
                          <div className="mt-2 w-full h-2 rounded-full bg-secondary overflow-hidden">
                            <div
                              className="h-full rounded-full bg-gradient-financial transition-all"
                              style={{ width: `${pct}%` }}
                              role="progressbar"
                              aria-valuemin={0}
                              aria-valuemax={100}
                              aria-valuenow={pct}
                            />
                          </div>
                        );
                      })()}

                      <div className="mt-2 text-sm text-muted-foreground">
                        💡 <strong>Smart Suggestions:</strong>
                        <ul className="list-disc ml-5">
                          {getSmartSuggestions(goal).map((s, i) => (
                            <li key={i}>{s}</li>
                          ))}
                        </ul>
                      </div>
                    </>
                  )}
                </div>
              ))}

              {planResults.length > 0 && (
                <button
                  className="w-full bg-primary text-white py-3 rounded-md font-semibold mt-4"
                  onClick={handlePDFDownload}
                >
                  📝 Export Plan as PDF
                </button>
              )}
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Goals;
