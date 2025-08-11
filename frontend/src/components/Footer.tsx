// src/components/Footer.jsx
import { Link } from "react-router-dom";
import logo from "../assets/logo.png";

const Footer = () => {
  return (
    <footer className="bg-foreground text-background py-12 mt-20">
      <div className="container mx-auto px-4">
        <div className="grid md:grid-cols-3 gap-8 text-center md:text-left">
          
          {/* Logo + Tagline */}
          <div>
            <div className="flex items-center justify-center md:justify-start space-x-3 mb-4">
              <img 
                src={logo} 
                alt="SmartFinance Logo" 
                className="h-10 w-10 rounded-full object-cover shadow-sm" 
              />
              <span className="text-xl font-bold">SmartFinance</span>
            </div>
            <p className="text-background/70">
              AI-powered financial assistant for Canadians
            </p>
          </div>

          {/* Contact Info */}
          <div>
            <h4 className="font-semibold mb-4">Contact</h4>
            <div className="space-y-2 text-background/70">
              <p>help@finbuddy.ca</p>
              <p>1-800-800-0088</p>
              <p>Kitchener, ON, Canada</p>
            </div>
          </div>

          {/* Legal Links */}
          <div>
            <h4 className="font-semibold mb-4">Legal</h4>
            <div className="space-y-2 text-background/70">
              <Link to="/terms" className="block hover:text-background transition-colors">
                Terms of Service
              </Link>
              <Link to="/privacy" className="block hover:text-background transition-colors">
                Privacy Policy
              </Link>
              <Link to="/feedback" className="block hover:text-background transition-colors">
                Feedback
              </Link>
            </div>
          </div>
        </div>

        {/* Bottom Note */}
        <div className="border-t border-background/20 mt-8 pt-8 text-center text-background/70">
          <p>&copy; 2025 FinBuddy. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
