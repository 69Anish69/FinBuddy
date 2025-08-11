import { useState, useEffect } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import {
  Home,
  Upload,
  TrendingUp,
  Target,
  CreditCard,
  FileText,
  Settings,
  Menu,
  X
} from "lucide-react";
import logo from "@/assets/logo.png";
import { auth } from "@/firebase";
import { onAuthStateChanged, signOut } from "firebase/auth";

const Navigation = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [userName, setUserName] = useState("");
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user) {
        setUserName(user.displayName || user.email);
      } else {
        setUserName("");
      }
    });
    return () => unsubscribe();
  }, []);

  const navigationItems = [
    { name: "Home", href: "/", icon: Home },
    { name: "Upload", href: "/upload", icon: Upload },
    { name: "Forecast", href: "/forecast", icon: TrendingUp },
    { name: "Goals", href: "/goals", icon: Target },
    { name: "Loans", href: "/loans", icon: CreditCard },
    { name: "Reports", href: "/reports", icon: FileText },
    { name: "Settings", href: "/settings", icon: Settings },
  ];

  const isActive = (path: string) => location.pathname === path;

  const handleLogout = async () => {
    await signOut(auth);
    navigate("/login");
  };

  return (
    <nav className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          {/* Logo and Brand */}
          <Link to="/" className="flex items-center space-x-3">
            <img
              src={logo}
              alt="SmartFinance Logo"
              className="h-10 w-10 rounded-full object-cover shadow-sm"
            />
            <span className="text-xl font-bold text-foreground">FinBuddy</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex md:items-center md:space-x-1">
            {navigationItems.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`flex items-center space-x-2 rounded-md px-3 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground ${
                    isActive(item.href)
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </div>

          {/* CTA Buttons */}
          <div className="hidden md:flex md:items-center md:space-x-3">
            {userName ? (
              <>
                <span className="text-sm font-medium text-muted-foreground">
                  Hi, {userName.split(" ")[0]}
                </span>
                <Button variant="ghost" size="sm" onClick={handleLogout}>
                  Logout
                </Button>
              </>
            ) : (
              <>
                <Link to="/login">
                  <Button variant="outline" size="sm">
                    Login
                  </Button>
                </Link>
                <Link to="/signup">
                  <Button variant="hero" size="sm">
                    Registration
                  </Button>
                </Link>
              </>
            )}
          </div>

          {/* Mobile Menu Toggle */}
          <div className="md:hidden">
            <Button variant="ghost" size="sm" onClick={() => setIsOpen(!isOpen)}>
              {isOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </Button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isOpen && (
          <div className="md:hidden pb-4">
            <div className="space-y-1 pt-2">
              {navigationItems.map((item) => {
                const Icon = item.icon;
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={`flex items-center space-x-2 rounded-md px-3 py-2 text-base font-medium transition-colors hover:bg-accent hover:text-accent-foreground ${
                      isActive(item.href)
                        ? "bg-primary text-primary-foreground"
                        : "text-muted-foreground"
                    }`}
                    onClick={() => setIsOpen(false)}
                  >
                    <Icon className="h-5 w-5" />
                    <span>{item.name}</span>
                  </Link>
                );
              })}

              {/* Mobile CTA */}
              <div className="mt-4 space-y-2">
                {userName ? (
                  <>
                    <div className="text-sm px-3 text-muted-foreground">
                      Logged in as <strong>{userName.split(" ")[0]}</strong>
                    </div>
                    <Button variant="ghost" className="w-full" onClick={handleLogout}>
                      Logout
                    </Button>
                  </>
                ) : (
                  <>
                    <Link to="/login">
                      <Button variant="outline" className="w-full" onClick={() => setIsOpen(false)}>
                        Login
                      </Button>
                    </Link>
                    <Link to="/signup">
                      <Button variant="hero" className="w-full" onClick={() => setIsOpen(false)}>
                        Registration
                      </Button>
                    </Link>
                  </>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navigation;
