import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import Navigation from "@/components/Navigation";
import heroImage from "@/assets/hero-dashboard.jpg";
import Footer from "@/components/Footer";
import {
  TrendingUp,
  Target,
  Upload,
  Shield,
  Zap,
  Users,
  Star,
  Banknote
} from "lucide-react";
import { Link } from "react-router-dom";

const Index = () => {
  const features = [
    {
      icon: TrendingUp,
      title: "Smart Forecast",
      description: "Predict if you'll hit your savings goals"
    },
    {
      icon: Target,
      title: "Goal Planner",
      description: "Buy a car, house or save for college with AI-backed planning"
    },
    {
      icon: Banknote,
      title: "Loan Optimizer",
      description: "Cut interest & pay debt smarter using optimized repayment strategies"
    }
  ];

  const testimonials = [
    {
      name: "Sarah M.",
      location: "Toronto, ON",
      text: "FinBuddy helped me save $15,000 for my house down payment in just 18 months!",
      rating: 5
    },
    {
      name: "Mike L.",
      location: "Vancouver, BC",
      text: "The loan optimization feature saved me over $3,000 in interest payments.",
      rating: 5
    },
    {
      name: "Jessica K.",
      location: "Montreal, QC",
      text: "Finally understand my finances thanks to the clear forecasting tools.",
      rating: 5
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-hero py-20 lg:py-32">
        <div className="container mx-auto px-4">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="text-center lg:text-left">
              <h1 className="text-4xl lg:text-6xl font-bold text-white mb-6">
                AI-Powered Personal Finance Assistant for{" "}
                <span className="text-red-500">Canadians</span>
              </h1>
              <p className="text-xl lg:text-2xl text-white/90 mb-8">
                Forecast your future. Plan your goals. Optimize your savings. All in one place.
              </p>

              <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
                <Link
                  to="/upload"
                  className="relative z-10 inline-flex items-center justify-center text-lg px-8 py-2 bg-white/10 text-white border border-white/30 hover:bg-white/20 rounded-md transition-all"
                >
                  <Upload className="w-5 h-5 mr-2" />
                  Upload Documents
                </Link>
                {/* <Button
                  variant="outline"
                  size="lg"
                  className="text-lg px-8 py-4 bg-white/10 text-white border-white/30 hover:bg-white/20"
                >
                  Book a Demo
                </Button> */}
              </div>
            </div>

            <div className="relative">
              <div className="relative z-10">
                <img
                  src={heroImage}
                  alt="FinBuddy Dashboard Interface"
                  className="rounded-lg shadow-glow w-full h-auto animate-float"
                />
              </div>
              <div className="absolute inset-0 bg-gradient-financial opacity-20 rounded-lg blur-xl"></div>
            </div>
          </div>
        </div>

        {/* Background Pattern */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-20 left-20 w-72 h-72 bg-white rounded-full mix-blend-multiply filter blur-xl animate-pulse"></div>
          <div className="absolute top-40 right-20 w-72 h-72 bg-white rounded-full mix-blend-multiply filter blur-xl animate-pulse" style={{ animationDelay: '2s' }}></div>
          <div className="absolute -bottom-20 left-1/2 w-72 h-72 bg-white rounded-full mix-blend-multiply filter blur-xl animate-pulse" style={{ animationDelay: '4s' }}></div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gradient-subtle">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-foreground mb-4">
              Everything You Need to Master Your Finances
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Powerful AI tools designed specifically for Canadian financial planning
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <Card key={index} className="text-center hover:shadow-card-hover transition-all duration-300 hover:-translate-y-2">
                  <CardHeader>
                    <div className="mx-auto w-16 h-16 bg-gradient-financial rounded-full flex items-center justify-center mb-4">
                      <Icon className="w-8 h-8 text-white" />
                    </div>
                    <CardTitle className="text-xl">{feature.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">{feature.description}</p>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>
      </section>

      {/* Trust Section */}
      <section className="py-20 bg-background">
        <div className="container mx-auto px-4">
          <div className="grid lg:grid-cols-3 gap-8 text-center">
            <div className="flex flex-col items-center">
              <div className="w-16 h-16 bg-gradient-financial rounded-full flex items-center justify-center mb-4">
                <Shield className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Privacy First</h3>
              <p className="text-muted-foreground">Your data is processed locally in-browser for maximum security</p>
            </div>

            <div className="flex flex-col items-center">
              <div className="w-16 h-16 bg-gradient-financial rounded-full flex items-center justify-center mb-4">
                <Zap className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Instant Analysis</h3>
              <p className="text-muted-foreground">Get real-time insights and recommendations powered by AI</p>
            </div>

            <div className="flex flex-col items-center">
              <div className="w-16 h-16 bg-gradient-financial rounded-full flex items-center justify-center mb-4">
                <Users className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Made for Canadians</h3>
              <p className="text-muted-foreground">Tailored for Canadian banks, taxes, and financial institutions</p>
            </div>
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="py-20 bg-gradient-subtle">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-foreground mb-4">
              Trusted by Thousands of Canadians
            </h2>
            <p className="text-xl text-muted-foreground">
              See how FinBuddy is helping Canadians achieve their financial goals
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <Card key={index} className="hover:shadow-card-hover transition-all duration-300">
                <CardContent className="p-6">
                  <div className="flex mb-4">
                    {[...Array(testimonial.rating)].map((_, i) => (
                      <Star key={i} className="w-5 h-5 fill-yellow-400 text-yellow-400" />
                    ))}
                  </div>
                  <p className="text-muted-foreground mb-4 italic">
                    "{testimonial.text}"
                  </p>
                  <div className="font-semibold text-foreground">
                    {testimonial.name}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {testimonial.location}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      {/* <footer className="bg-foreground text-background py-12">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <div className="w-8 h-8 bg-gradient-financial rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold">$</span>
                </div>
                <span className="text-xl font-bold">FinBuddy</span>
              </div>
              <p className="text-background/70">
                AI-powered financial assistant for Canadians
              </p>
            </div>

            <div>
              <h4 className="font-semibold mb-4">Features</h4>
              <ul className="space-y-2 text-background/70">
                <li>Smart Forecasting</li>
                <li>Goal Planning</li>
                <li>Loan Optimization</li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold mb-4">Support</h4>
              <ul className="space-y-2 text-background/70">
                <li>Help Center</li>
                <li>Contact Us</li>
                <li>Privacy Policy</li>
                <li>Terms of Service</li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold mb-4">Contact</h4>
              <ul className="space-y-2 text-background/70">
                <li>help@finbuddy.ca</li>
                <li>1-800-FINBUDDY</li>
                <li>Toronto, ON, Canada</li>
              </ul>
            </div>
          </div>

          <div className="border-t border-background/20 mt-8 pt-8 text-center text-background/70">
            <p>&copy; 2024 FinBuddy. All rights reserved. Made with 💙 in Canada.</p>
          </div>
        </div>
      </footer> */}
      <Footer />
    </div>
  );
};

export default Index;
