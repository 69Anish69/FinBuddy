import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { signInWithEmailAndPassword } from "firebase/auth";
import { auth } from "@/firebase"; // ✅ Make sure this file exists correctly
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Card, CardContent } from "@/components/ui/card";
import loginIllustration from "@/assets/login-illustration.jpg";

const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [rememberMe, setRememberMe] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    try {
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      const idToken = await userCredential.user.getIdToken();

      // Optionally: send token to backend
      const res = await fetch("http://localhost:8000/auth/firebase-login", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${idToken}`
        }
      });

      const result = await res.json();
      if (result.message) {
        console.log("✅ Logged in:", result.message);
        navigate("/"); // or your dashboard route
      } else {
        setError("Login failed. Please try again.");
      }
    } catch (err) {
      setError(err.message || "Something went wrong.");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-subtle">
      {/* Top Navigation */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link to="/" className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-financial rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">F</span>
              </div>
              <span className="text-xl font-bold text-foreground">FinBuddy</span>
            </Link>
            <Link to="/register">
              <Button variant="outline">Create Account</Button>
            </Link>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="flex-1 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
        <Card className="w-full max-w-6xl shadow-glow">
          <CardContent className="p-0">
            <div className="flex flex-col lg:flex-row">
              {/* Left Side - Illustration */}
              <div className="lg:w-1/2 bg-gradient-to-br from-primary/5 to-secondary/10 p-8 lg:p-12 flex items-center justify-center">
                <div className="text-center max-w-md">
                  <img
                    src={loginIllustration}
                    alt="Office reception illustration"
                    className="w-full h-auto rounded-lg mb-6"
                  />
                  <h2 className="text-2xl font-semibold text-foreground mb-4">
                    Welcome to FinBuddy
                  </h2>
                  <p className="text-muted-foreground">
                    Your trusted AI-powered financial assistant for Canadians
                  </p>
                </div>
              </div>

              {/* Right Side - Login Form */}
              <div className="lg:w-1/2 p-8 lg:p-12">
                <div className="max-w-md mx-auto">
                  <div className="text-center mb-8">
                    <h1 className="text-3xl font-bold text-foreground mb-2">
                      Welcome Back 🙂
                    </h1>
                    <p className="text-muted-foreground">
                      To keep connected with us, please login with your personal info
                    </p>
                  </div>

                  <form onSubmit={handleSubmit} className="space-y-6">
                    <div className="space-y-2">
                      <Label htmlFor="email">Email Address</Label>
                      <Input
                        id="email"
                        type="email"
                        placeholder="Enter your email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        required
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="password">Password</Label>
                      <Input
                        id="password"
                        type="password"
                        placeholder="Enter your password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="remember"
                          checked={rememberMe}
                          onCheckedChange={(checked) => setRememberMe(Boolean(checked))}
                        />
                        <Label htmlFor="remember" className="text-sm">
                          Remember Me
                        </Label>
                      </div>
                      <Link
                        to="/forgot-password"
                        className="text-sm text-primary hover:text-primary-hover transition-colors"
                      >
                        Forgot Password?
                      </Link>
                    </div>

                    {error && (
                      <div className="text-sm text-red-500 text-center">{error}</div>
                    )}

                    <Button type="submit" className="w-full">
                      Login
                    </Button>

                    <div className="text-center">
                      <span className="text-muted-foreground text-sm">
                        Don't have an account?{" "}
                      </span>
                      <Link
                        to="/register"
                        className="text-primary hover:text-primary-hover font-medium text-sm transition-colors"
                      >
                        Create Account
                      </Link>
                    </div>

                    {/* Divider */}
                    <div className="relative">
                      <div className="absolute inset-0 flex items-center">
                        <div className="w-full border-t border-border"></div>
                      </div>
                      <div className="relative flex justify-center text-sm">
                        <span className="px-4 bg-background text-muted-foreground">
                          or login with
                        </span>
                      </div>
                    </div>

                    {/* Social Login Buttons */}
                    <div className="grid grid-cols-3 gap-3">
                      {["Google", "Facebook", "Twitter"].map((provider, i) => (
                        <Button
                          key={provider}
                          variant="outline"
                          className="p-3"
                          onClick={() => alert(`${provider} login coming soon!`)}
                        >
                          <span className="sr-only">{provider}</span>
                          <svg
                            className="w-5 h-5"
                            viewBox="0 0 24 24"
                            fill="currentColor"
                          >
                            <circle cx="12" cy="12" r="10" />
                          </svg>
                        </Button>
                      ))}
                    </div>
                  </form>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Footer */}
      <footer className="bg-white border-t py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center text-muted-foreground text-sm">
            © {new Date().getFullYear()} FinBuddy. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Login;
