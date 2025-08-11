import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { createUserWithEmailAndPassword, updateProfile } from "firebase/auth";
import { auth } from "@/firebase";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import signupIllustration from "@/assets/login-illustration.jpg";

const Signup = () => {
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    confirmPassword: "",
    agreeToTerms: false,
  });

  const [errors, setErrors] = useState<{ [key: string]: string }>({});
  const [firebaseError, setFirebaseError] = useState("");

  const validate = () => {
    const newErrors: { [key: string]: string } = {};

    if (!formData.firstName.trim()) newErrors.firstName = "First name is required.";
    if (!formData.lastName.trim()) newErrors.lastName = "Last name is required.";

    if (!formData.email) {
      newErrors.email = "Email is required.";
    } else if (!/^[\w-.]+@([\w-]+\.)+[\w-]{2,4}$/.test(formData.email)) {
      newErrors.email = "Enter a valid email address.";
    }

    if (!formData.password) {
      newErrors.password = "Password is required.";
    } else if (formData.password.length < 6) {
      newErrors.password = "Password must be at least 6 characters.";
    }

    if (formData.confirmPassword !== formData.password) {
      newErrors.confirmPassword = "Passwords do not match.";
    }

    if (!formData.agreeToTerms) {
      newErrors.agreeToTerms = "You must agree to the terms.";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate()) return;

    try {
      const userCredential = await createUserWithEmailAndPassword(
        auth,
        formData.email,
        formData.password
      );

      await updateProfile(userCredential.user, {
        displayName: `${formData.firstName} ${formData.lastName}`,
      });

      navigate("/");
    } catch (err: any) {
      console.error("Registration error:", err.message);
      setFirebaseError(err.message);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-subtle flex flex-col">
      <Navigation />

      <main className="flex-1 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
        <div className="w-full max-w-6xl bg-card shadow-xl rounded-xl overflow-hidden">
          <div className="grid grid-cols-1 md:grid-cols-2">
            {/* Illustration */}
            <div className="hidden md:flex bg-gradient-to-br from-primary/5 to-primary/10 p-10 items-center justify-center relative">
              <div className="text-left space-y-4 z-10">
                <h2 className="text-2xl font-bold text-foreground">
                  Join FinBuddy Today
                </h2>
                <p className="text-muted-foreground">
                  Start your journey to financial wellness with Canada's most trusted personal finance platform.
                </p>

                {/* Features */}
                <ul className="space-y-3 mt-6 text-sm text-muted-foreground">
                  <li className="flex items-center gap-2 hover:text-primary transition">
                    <span className="bg-blue-100 text-blue-600 rounded-full p-1">
                      🔒
                    </span>
                    Bank-level security
                  </li>
                  <li className="flex items-center gap-2 hover:text-primary transition">
                    <span className="bg-green-100 text-green-600 rounded-full p-1">
                      🇨🇦
                    </span>
                    Built for Canadians
                  </li>
                  <li className="flex items-center gap-2 hover:text-primary transition">
                    <span className="bg-yellow-100 text-yellow-600 rounded-full p-1">
                      📊
                    </span>
                    AI-powered insights
                  </li>
                </ul>
              </div>

              {/* Background Illustration Image */}
              <img
                src={signupIllustration}
                alt="Signup Illustration"
                className="absolute inset-0 w-full h-full object-cover opacity-10 pointer-events-none"
              />
            </div>


            <div className="p-8 sm:p-12">
              <div className="max-w-md mx-auto space-y-6">
                <div className="text-center">
                  <h1 className="text-3xl font-bold text-foreground">Create Account</h1>
                  <p className="mt-2 text-sm text-muted-foreground">
                    Join thousands of Canadians taking control of their finances.
                  </p>
                </div>

                <form onSubmit={handleSubmit} className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="firstName">First Name</Label>
                      <Input
                        id="firstName"
                        name="firstName"
                        placeholder="John"
                        value={formData.firstName}
                        onChange={handleInputChange}
                      />
                      {errors.firstName && <p className="text-xs text-red-500">{errors.firstName}</p>}
                    </div>
                    <div>
                      <Label htmlFor="lastName">Last Name</Label>
                      <Input
                        id="lastName"
                        name="lastName"
                        placeholder="Doe"
                        value={formData.lastName}
                        onChange={handleInputChange}
                      />
                      {errors.lastName && <p className="text-xs text-red-500">{errors.lastName}</p>}
                    </div>
                  </div>

                  <div>
                    <Label htmlFor="email">Email</Label>
                    <Input
                      id="email"
                      name="email"
                      type="email"
                      placeholder="you@example.com"
                      value={formData.email}
                      onChange={handleInputChange}
                    />
                    {errors.email && <p className="text-xs text-red-500">{errors.email}</p>}
                  </div>

                  <div>
                    <Label htmlFor="password">Password</Label>
                    <Input
                      id="password"
                      name="password"
                      type="password"
                      placeholder="••••••••"
                      value={formData.password}
                      onChange={handleInputChange}
                    />
                    {errors.password && <p className="text-xs text-red-500">{errors.password}</p>}
                  </div>

                  <div>
                    <Label htmlFor="confirmPassword">Confirm Password</Label>
                    <Input
                      id="confirmPassword"
                      name="confirmPassword"
                      type="password"
                      placeholder="••••••••"
                      value={formData.confirmPassword}
                      onChange={handleInputChange}
                    />
                    {errors.confirmPassword && (
                      <p className="text-xs text-red-500">{errors.confirmPassword}</p>
                    )}
                  </div>

                  <div className="flex items-start space-x-2">
                    <Checkbox
                      id="agreeToTerms"
                      checked={formData.agreeToTerms}
                      onCheckedChange={(checked) =>
                        setFormData((prev) => ({ ...prev, agreeToTerms: checked as boolean }))
                      }
                    />
                    <Label htmlFor="agreeToTerms" className="text-sm text-muted-foreground">
                      I agree to the{" "}
                      <Link to="/terms" className="text-primary underline hover:text-primary-hover">
                        Terms of Service
                      </Link>{" "}
                      and{" "}
                      <Link to="/privacy" className="text-primary underline hover:text-primary-hover">
                        Privacy Policy
                      </Link>
                    </Label>
                  </div>
                  {errors.agreeToTerms && (
                    <p className="text-xs text-red-500">{errors.agreeToTerms}</p>
                  )}

                  {firebaseError && (
                    <p className="text-sm text-red-500 font-medium">{firebaseError}</p>
                  )}

                  <Button type="submit" className="w-full" variant="hero">
                    Create Account
                  </Button>
                </form>

                <p className="text-sm text-center text-muted-foreground">
                  Already have an account?{" "}
                  <Link to="/login" className="text-primary font-medium hover:underline">
                    Sign in
                  </Link>
                </p>

                {/* <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <Separator className="w-full" />
                  </div>
                  <div className="relative flex justify-center text-xs uppercase">
                    <span className="bg-card px-2 text-muted-foreground">or sign up with</span>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-3">
                  <Button variant="outline">Google</Button>
                  <Button variant="outline">Facebook</Button>
                  <Button variant="outline">Twitter</Button>
                </div> */}
              </div>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Signup;
