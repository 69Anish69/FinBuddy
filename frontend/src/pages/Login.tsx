import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { signInWithEmailAndPassword } from "firebase/auth";
import { auth } from "@/firebase";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Card, CardContent } from "@/components/ui/card";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import loginIllustration from "@/assets/login-illustration.jpg";

const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [rememberMe, setRememberMe] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      console.log("Logged in:", userCredential.user);
      navigate("/");
    } catch (err: any) {
      setError("Invalid email or password. Please try again.");
      console.error("Login error:", err.message);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-subtle flex flex-col">
      <Navigation />

      <main className="flex-1 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
        <Card className="w-full max-w-6xl shadow-xl">
          <CardContent className="p-0">
            <div className="flex flex-col lg:flex-row">
              {/* Left Side - Illustration */}
              <div className="lg:w-1/2 bg-gradient-to-br from-primary/5 to-secondary/10 p-10 flex items-center justify-center">
                <img src={loginIllustration} alt="Illustration" className="rounded-lg w-full max-w-sm" />
              </div>

              {/* Right Side - Login Form */}
              <div className="lg:w-1/2 p-8 lg:p-12">
                <div className="max-w-md mx-auto">
                  <div className="text-center mb-6">
                    <h1 className="text-3xl font-bold text-foreground">Welcome Back :)</h1>
                    <p className="text-muted-foreground mt-2">
                      Please login with your email and password.
                    </p>
                  </div>

                  <form onSubmit={handleSubmit} className="space-y-6">
                    <div className="space-y-2">
                      <Label htmlFor="email">Email</Label>
                      <Input
                        id="email"
                        type="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="Enter your email"
                        required
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="password">Password</Label>
                      <Input
                        id="password"
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="Enter your password"
                        required
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="remember"
                          checked={rememberMe}
                          onCheckedChange={(checked) => setRememberMe(!!checked)}
                        />
                        <Label htmlFor="remember" className="text-sm">
                          Remember Me
                        </Label>
                      </div>
                      <Link to="/forgot-password" className="text-sm text-primary hover:underline">
                        Forgot Password?
                      </Link>
                    </div>

                    {error && (
                      <div className="text-red-500 text-sm font-medium">{error}</div>
                    )}

                    <Button type="submit" className="w-full">
                      Login
                    </Button>

                    <div className="text-center text-sm text-muted-foreground">
                      Don't have an account?{" "}
                      <Link to="/signup" className="text-primary hover:underline">
                        Create Account
                      </Link>
                    </div>
                  </form>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </main>

      <Footer />
    </div>
  );
};

export default Login;
