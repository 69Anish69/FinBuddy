'use client';

import { useRef, useState } from "react";
import Navigation from "@/components/Navigation";
import { Link, useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Upload as UploadIcon, TrendingUp, Target, Shield, DollarSign } from "lucide-react";
import axios from "axios";
import Footer from "@/components/Footer";

type NextStep = {
  icon: any;
  title: string;
  description: string;
  href: string;
  emoji: string;
};

const API_BASE =
  // prefer env var if provided
  ((import.meta as any)?.env?.VITE_API_URL ||
    (typeof process !== "undefined" && (process as any).env?.NEXT_PUBLIC_API_URL) ||
    "").toString().replace(/\/+$/, "") ||
  // dev fallback: if UI is on 3000/8080, assume FastAPI on 8000
  (typeof window !== "undefined" && (window.location.port === "3000" || window.location.port === "8080")
    ? "http://127.0.0.1:8000"
    : "");

const Upload = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [message, setMessage] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement | null>(null);

  const nextSteps: NextStep[] = [
    { icon: TrendingUp, title: "View Forecast", description: "Predict if you'll hit your savings goals", href: "/forecast", emoji: "📈" },
    { icon: Target,     title: "Plan a Goal",   description: "Buy a car, house or save for college with AI planning", href: "/goals",    emoji: "🎯" },
    { icon: DollarSign, title: "Manage Loans",  description: "Track and optimize your loan payments", href: "/loans",  emoji: "💰" },
  ];

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSelectedFile(e.target.files?.[0] ?? null);
    setMessage("");
  };

  const handleUploadAndForecast = async () => {
    if (!selectedFile) {
      setMessage("⚠️ Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile); // name must be 'file' for FastAPI

    setIsUploading(true);
    setMessage("");

    try {
      const res = await axios.post(`${API_BASE}/upload`, formData, {
        // do NOT set Content-Type; axios will add multipart boundary automatically
        withCredentials: false,
        validateStatus: () => true, // handle errors by inspecting response
      });

      const data = res.data || {};
      if (res.status >= 200 && res.status < 300 && !data.error && (data.ok ?? true)) {
        setMessage("✅ File uploaded successfully! Redirecting to forecast...");
        setTimeout(() => navigate("/forecast"), 800);
      } else {
        const reason = data?.error || `Upload failed (HTTP ${res.status}).`;
        throw new Error(reason);
      }
    } catch (err: any) {
      console.error("Upload error:", err);
      setMessage(`❌ Upload failed. ${err?.message ?? ""}`);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <div className="container mx-auto px-4 py-16">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div className="text-center lg:text-left">
            <div className="mb-8">
              <div className="w-24 h-24 bg-gradient-financial rounded-full flex items-center justify-center mx-auto lg:mx-0 mb-6">
                <Shield className="w-12 h-12 text-white" />
              </div>
              <h1 className="text-4xl lg:text-5xl font-bold text-foreground mb-4">
                Upload Financial Documents
              </h1>
              <p className="text-xl text-muted-foreground mb-6">
                Securely upload PDF, CSV, or Excel files to get personalized insights for your Canadian financial goals.
              </p>
              <div className="space-y-4 text-left">
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 bg-primary rounded-full"></div>
                  <span className="text-muted-foreground">Bank statements & transaction records</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 bg-primary rounded-full"></div>
                  <span className="text-muted-foreground">Investment portfolios & RRSP statements</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 bg-primary rounded-full"></div>
                  <span className="text-muted-foreground">Expense reports & financial invoices</span>
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-8">
            <div
              className="bg-card rounded-lg border-2 border-dashed border-border p-8 text-center hover:border-primary transition-colors shadow-card-hover cursor-pointer"
              onClick={() => inputRef.current?.click()}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".pdf,.csv,.xls,.xlsx"
                hidden
                onChange={handleFileChange}
              />
              <div className="text-6xl mb-4">📄</div>
              <h3 className="text-lg font-semibold mb-2">Drop files here or click to browse</h3>
              <p className="text-muted-foreground mb-4">PDF, CSV, XLS files up to 10MB</p>
              <p className="text-sm text-muted-foreground mb-6">
                <span className="inline-flex items-center gap-2">
                  <Shield className="w-4 h-4" />
                  Data is processed locally in-browser for privacy
                </span>
              </p>
              <Button className="bg-gradient-financial text-white px-8 py-3 shadow-financial hover:shadow-glow">
                <UploadIcon className="w-4 h-4 mr-2" />
                {selectedFile ? selectedFile.name : "Select Files"}
              </Button>
            </div>

            <Button
              size="lg"
              className="w-full bg-gradient-financial text-white shadow-financial hover:shadow-glow disabled:opacity-60"
              onClick={handleUploadAndForecast}
              disabled={isUploading}
            >
              {isUploading ? "⌛ Uploading..." : "Continue to Analysis"}
            </Button>

            <div className="min-h-[24px] text-center">
              {message && (
                <p className={`text-sm ${message.startsWith("✅") ? "text-green-500" : "text-red-500"}`}>
                  {message}
                </p>
              )}
            </div>
          </div>
        </div>

        <div className="mt-20">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-foreground mb-4">What's Next?</h2>
            <p className="text-xl text-muted-foreground">
              Choose your next step to maximize your financial potential
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {nextSteps.map((step, index) => {
              const Icon = step.icon;
              return (
                <Link key={index} to={step.href}>
                  <Card className="text-center hover:shadow-card-hover transition-all duration-300 hover:-translate-y-2 cursor-pointer">
                    <CardHeader>
                      <div className="mx-auto w-16 h-16 bg-gradient-financial rounded-full flex items-center justify-center mb-4">
                        <Icon className="w-8 h-8 text-white" />
                      </div>
                      <CardTitle className="text-xl flex items-center justify-center gap-2">
                        <span>{step.emoji}</span>
                        {step.title}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-muted-foreground">{step.description}</p>
                    </CardContent>
                  </Card>
                </Link>
              );
            })}
          </div>
        </div>
      </div>

      <Footer />
    </div>
  );
};

export default Upload;
