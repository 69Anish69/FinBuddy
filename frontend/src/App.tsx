// App.tsx
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useNavigate, useLocation } from "react-router-dom";

import AppVoiceAssistant from "@/components/AppVoiceAssistant";

import Index from "@/pages/Index";
import Upload from "@/pages/Upload";
import Chat from "@/pages/Chat";
import Forecast from "@/pages/Forecast";
import Goals from "@/pages/Goals";
import Loans from "@/pages/Loans";
import Reports from "@/pages/Reports";
import Settings from "@/pages/Settings";
import NotFound from "@/pages/NotFound";
import LoginPage from "@/pages/Login";
import Signup from "@/pages/Signup";
import ForgotPassword from "@/pages/ForgotPassword";
// import ProtectedRoute from "@/components/ProtectedRoute";

const queryClient = new QueryClient();

// Small helper so we can pass `navigate` into the assistant
const VoiceLayer = () => {
  const navigate = useNavigate();
  const { pathname } = useLocation();   // <— NEW

  return (
    <>
      <AppVoiceAssistant
        page={pathname}                  // <— NEW
        lang="en-US"
        navigate={(route) => navigate(route)}
        api={{
          forecast: "http://localhost:8000/assistant/forecast",
          suggestions: "http://localhost:8000/assistant/suggestions",
          loanPlan: "http://localhost:8000/assistant/loans/plan",
          uploadSummary: "http://localhost:8000/assistant/upload/summary",
        }}
      />
    </>
  );
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          {/* Public Routes */}
          <Route path="/" element={<Index />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/forgot-password" element={<ForgotPassword />} />
          <Route path="/settings" element={<Settings />} />

          {/* Protected Routes */}
          {/* <Route element={<ProtectedRoute />}> */}
          <Route path="/upload" element={<Upload />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/forecast" element={<Forecast />} />
          <Route path="/goals" element={<Goals />} />
          <Route path="/loans" element={<Loans />} />
          <Route path="/reports" element={<Reports />} />
          {/* </Route> */}

          {/* Catch-all */}
          <Route path="*" element={<NotFound />} />
        </Routes>

        {/* Mount once so it appears on every page */}
        <VoiceLayer />
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
