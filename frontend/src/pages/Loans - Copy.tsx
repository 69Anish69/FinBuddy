'use client';

import React, { useState } from 'react';
import Nav from '@/components/Navigation';
import Footer from '@/components/Footer';
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  Title,
  Tooltip,
  Legend,
  CategoryScale,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  LineElement,
  PointElement,
  LinearScale,
  Title,
  Tooltip,
  Legend,
  CategoryScale
);

const Loans = () => {
  const [loans, setLoans] = useState([
    { id: 1, name: 'Credit Card', balance: 5500, rate: 19.9, payment: 165 },
    { id: 2, name: 'Student Loan', balance: 25000, rate: 6.5, payment: 275 },
    { id: 3, name: 'Car Loan', balance: 18500, rate: 4.2, payment: 385 },
  ]);

  // Yearly contribution slider state, default $1200 (equiv $100/month)
  const [yearlyContribution, setYearlyContribution] = useState(1200);

  const [newLoan, setNewLoan] = useState({ name: '', balance: '', rate: '', payment: '' });

  const handleAddLoan = () => {
    setLoans([
      ...loans,
      {
        ...newLoan,
        id: Date.now(),
        balance: parseFloat(newLoan.balance),
        rate: parseFloat(newLoan.rate),
        payment: parseFloat(newLoan.payment),
      },
    ]);
    setNewLoan({ name: '', balance: '', rate: '', payment: '' });
  };

  const chartData = {
    labels: ['2025', '2026', '2027', '2028', '2029', '2030'],
    datasets: [
      {
        label: 'Only Minimum EMI',
        data: [49000, 42000, 35000, 28000, 21000, 14000],
        borderColor: 'red',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        fill: true,
      },
      {
        label: 'With Extra Contribution',
        data: [49000, 38000, 26000, 15000, 5000, 0],
        borderColor: 'green',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        fill: true,
      },
    ],
  };

  // Calculate interest saved and years reduced based on yearlyContribution
  const interestSaved = Math.min(15000, yearlyContribution * 8); // example factor
  const yearsReduced = Math.min(5, Math.floor(yearlyContribution / 1500));

  return (
    <>
      <Nav />
      <div className="min-h-screen bg-background">
        <div className="container mx-auto px-4 py-16 max-w-6xl">
          <h1 className="text-4xl font-bold text-foreground text-center mb-2">Optimize Your Loans</h1>
          <p className="text-lg text-muted-foreground text-center mb-10">Smarter payoff strategies powered by AI</p>

          <div className="grid lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 space-y-6">
              <div className="bg-card rounded-lg border shadow-sm p-6">
                <h3 className="text-lg font-semibold mb-4">Current Loans</h3>
                {loans.map((loan) => (
                  <div key={loan.id} className="border rounded-lg p-4 mb-2">
                    <div className="flex justify-between">
                      <strong>{loan.name}</strong>
                      <span>{loan.rate}% APR</span>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Balance: ${loan.balance.toLocaleString()} | Min Pay: ${loan.payment}/mo
                    </p>
                  </div>
                ))}

                <h4 className="mt-6 mb-2 font-medium">Add New Loan</h4>
                <div className="grid md:grid-cols-4 gap-2">
                  <input
                    type="text"
                    placeholder="Loan"
                    className="border p-2 rounded"
                    value={newLoan.name}
                    onChange={(e) => setNewLoan({ ...newLoan, name: e.target.value })}
                  />
                  <input
                    type="number"
                    placeholder="Balance"
                    className="border p-2 rounded"
                    value={newLoan.balance}
                    onChange={(e) => setNewLoan({ ...newLoan, balance: e.target.value })}
                  />
                  <input
                    type="number"
                    placeholder="Rate %"
                    className="border p-2 rounded"
                    value={newLoan.rate}
                    onChange={(e) => setNewLoan({ ...newLoan, rate: e.target.value })}
                  />
                  <input
                    type="number"
                    placeholder="Payment"
                    className="border p-2 rounded"
                    value={newLoan.payment}
                    onChange={(e) => setNewLoan({ ...newLoan, payment: e.target.value })}
                  />
                </div>
                <button onClick={handleAddLoan} className="mt-2 bg-blue-600 text-white px-4 py-2 rounded text-sm">
                  Add Loan
                </button>
              </div>

              <div className="bg-card rounded-lg border shadow-sm p-6">
                <h3 className="text-lg font-semibold mb-4">Repayment Strategies</h3>
                <div className="grid md:grid-cols-3 gap-4">
                  <div className="border p-3 rounded hover:border-primary">
                    <h4 className="font-medium">🔥 Avalanche</h4>
                    <p className="text-xs">High Interest First</p>
                    <p className="text-xs mt-1 text-muted-foreground">
                      Pay off loans with the highest interest rates first to save on overall interest.
                    </p>
                  </div>
                  <div className="border p-3 rounded hover:border-primary">
                    <h4 className="font-medium">⚡ Snowball</h4>
                    <p className="text-xs">Smallest Balance First</p>
                    <p className="text-xs mt-1 text-muted-foreground">
                      Pay off smallest debts first to gain momentum and motivation as debts are cleared quickly.
                    </p>
                  </div>
                  <div className="border p-3 rounded hover:border-primary">
                    <h4 className="font-medium">⚖️ Equal EMI</h4>
                    <p className="text-xs">Even Split</p>
                    <p className="text-xs mt-1 text-muted-foreground">
                      Divide your payments equally among all loans, paying consistent amounts each month.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-card rounded-lg border p-6">
                <h3 className="text-lg font-semibold mb-3">Normal vs Extra Payment Strategy</h3>
                <Line data={chartData} />
                {/* <p className="mt-3 text-sm text-gray-600">
                  {yearlyContribution > 0 ? (
                    <>By paying <strong>${yearlyContribution.toLocaleString()}</strong> extra yearly, you save approximately <strong>${interestSaved.toLocaleString()}</strong> in interest and reduce your loan term by <strong>{yearsReduced}</strong> years.</>
                  ) : (
                    <>Adjust the slider below to see potential savings and loan term reduction.</>
                  )}
                </p> */}
              </div>

              <div className="bg-card border rounded-lg p-4 mt-6">
                <h3 className="font-semibold mb-3">🧠 AI Suggestion Based on Your Yearly Savings</h3>
                <input
                  type="range"
                  min="0"
                  max="60000"
                  step="100"
                  value={yearlyContribution}
                  onChange={(e) => setYearlyContribution(Number(e.target.value))}
                  className="w-full"
                />
                <p className="text-sm text-muted-foreground mt-1">You can contribute: ${yearlyContribution.toLocaleString()} CAD/year</p>
                <div className="mt-4 space-y-1 text-sm">
                  <p>📉 <strong>Interest Saved:</strong> ${interestSaved.toLocaleString()}</p>
                  <p>⏱️ <strong>Loan Term Reduced:</strong> {yearsReduced} years</p>
                  <p>💡 <strong>Best Strategy:</strong> Avalanche Method recommended</p>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-card border rounded-lg p-4">
                <h3 className="font-semibold mb-3">Quick Stats</h3>
                <div className="text-sm space-y-1">
                  <div className="flex justify-between"><span>Total Debt:</span><span>$49,000</span></div>
                  <div className="flex justify-between"><span>Monthly Payment:</span><span>$825</span></div>
                  <div className="flex justify-between"><span>Debt-to-Income:</span><span>18.5%</span></div>
                </div>
              </div>

              {/* Download button removed as requested */}
            </div>
          </div>
        </div>
      </div>
      <Footer />
    </>
  );
};

export default Loans;
