import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";

const Settings = () => {
  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Navigation />

      <div className="flex-grow">
        <div className="container mx-auto px-4 py-16">
          <div className="mx-auto max-w-4xl">
            <h1 className="text-4xl font-bold text-foreground mb-4 text-center">
              Settings & About
            </h1>
            <p className="text-xl text-muted-foreground mb-12 text-center">
              Customize your FinBuddy experience
            </p>

            <div className="grid lg:grid-cols-2 gap-8">
              <div className="space-y-6">
                <div className="bg-card rounded-lg border shadow-card-hover p-6">
                  <h3 className="text-lg font-semibold mb-4">Preferences</h3>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <label className="font-medium">Theme</label>
                      <select className="px-3 py-2 border border-input rounded-md">
                        <option>Light</option>
                        <option>Dark</option>
                        <option>System</option>
                      </select>
                    </div>

                    <div className="flex items-center justify-between">
                      <label className="font-medium">Currency</label>
                      <select className="px-3 py-2 border border-input rounded-md">
                        <option>CAD - Canadian Dollar</option>
                        <option>USD - US Dollar</option>
                      </select>
                    </div>

                    <div className="flex items-center justify-between">
                      <label className="font-medium">Email Notifications</label>
                      <input type="checkbox" className="w-4 h-4" />
                    </div>

                    <div className="flex items-center justify-between">
                      <label className="font-medium">Weekly Reports</label>
                      <input type="checkbox" className="w-4 h-4" defaultChecked />
                    </div>
                  </div>
                </div>

                <div className="bg-card rounded-lg border shadow-card-hover p-6">
                  <h3 className="text-lg font-semibold mb-4">Account</h3>
                  <div className="space-y-4">
                    <button className="w-full text-left px-4 py-2 bg-gradient-subtle rounded-md hover:bg-secondary transition-colors">
                      Export My Data
                    </button>
                    <button className="w-full text-left px-4 py-2 bg-gradient-subtle rounded-md hover:bg-secondary transition-colors">
                      Delete Account
                    </button>
                  </div>
                </div>
              </div>

              <div className="space-y-6">
                <div className="bg-card rounded-lg border shadow-card-hover p-6">
                  <h3 className="text-lg font-semibold mb-4">About FinBuddy</h3>
                  <div className="space-y-4">
                    <div className="flex items-center gap-3">
                      <div className="w-12 h-12 bg-gradient-financial rounded-lg flex items-center justify-center">
                        <span className="text-white text-xl">💰</span>
                      </div>
                      <div>
                        <h4 className="font-semibold">FinBuddy</h4>
                        <p className="text-sm text-muted-foreground">Version 1.0.0</p>
                      </div>
                    </div>

                    <p className="text-sm text-muted-foreground">
                      Your AI-powered financial assistant designed specifically for Canadian users. 
                      Helping you forecast your future, plan your goals, and optimize your savings.
                    </p>

                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Privacy Policy</span>
                        <span className="text-primary cursor-pointer">View</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Terms of Service</span>
                        <span className="text-primary cursor-pointer">View</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Contact Support</span>
                        <span className="text-primary cursor-pointer">help@finbuddy.ca</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-card rounded-lg border shadow-card-hover p-6">
                  <h3 className="text-lg font-semibold mb-4">FAQ</h3>
                  <div className="space-y-3">
                    <details className="group">
                      <summary className="cursor-pointer font-medium">How secure is my data?</summary>
                      <p className="text-sm text-muted-foreground mt-2 pl-4">
                        All data is processed locally in your browser and encrypted in transit. We never store sensitive financial information on our servers.
                      </p>
                    </details>

                    <details className="group">
                      <summary className="cursor-pointer font-medium">What file formats are supported?</summary>
                      <p className="text-sm text-muted-foreground mt-2 pl-4">
                        We support PDF, CSV, and Excel files from major Canadian banks and financial institutions.
                      </p>
                    </details>

                    <details className="group">
                      <summary className="cursor-pointer font-medium">Is this service free?</summary>
                      <p className="text-sm text-muted-foreground mt-2 pl-4">
                        Basic features are free. Premium features with advanced AI analysis are available with our Pro plan.
                      </p>
                    </details>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <Footer />
    </div>
  );
};

export default Settings;
