"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import {
  User,
  Bell,
  Globe,
  Palette,
  Key,
  Shield,
  // Database,
  Brain,
  Save,
  RefreshCw,
} from "lucide-react";
// import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { useAuthStore } from "@/stores/authStore";

interface SettingSection {
  id: string;
  title: string;
  icon: React.ComponentType<{ className?: string }>;
}

const sections: SettingSection[] = [
  { id: "profile", title: "Profile", icon: User },
  { id: "notifications", title: "Notifications", icon: Bell },
  { id: "language", title: "Language & Region", icon: Globe },
  { id: "appearance", title: "Appearance", icon: Palette },
  { id: "model", title: "ML Model", icon: Brain },
  { id: "api", title: "API Keys", icon: Key },
  { id: "security", title: "Security", icon: Shield },
];

export default function SettingsPage() {
  const _t = useTranslations("settings");
  const user = useAuthStore((state) => state.user);
  const [activeSection, setActiveSection] = useState("profile");
  const [isSaving, setIsSaving] = useState(false);

  // Form states
  const [profile, setProfile] = useState({
    fullName: user?.full_name || "",
    email: user?.email || "",
    department: "Engineering",
    role: "Cost Engineer",
  });

  const [notifications, setNotifications] = useState({
    emailOnComplete: true,
    emailOnError: true,
    browserNotifications: false,
    weeklyDigest: true,
  });

  const [language, setLanguage] = useState("en");
  const [theme, setTheme] = useState("system");

  const [modelConfig, setModelConfig] = useState({
    autoRetrain: true,
    retrainThreshold: 5,
    confidenceThreshold: 0.7,
    useSimilarPRs: true,
    maxSimilarPRs: 5,
  });

  const handleSave = async () => {
    setIsSaving(true);
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setIsSaving(false);
  };

  const renderSection = () => {
    switch (activeSection) {
      case "profile":
        return (
          <Card>
            <CardHeader>
              <CardTitle>Profile Settings</CardTitle>
              <CardDescription>Manage your account information</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Full Name</label>
                  <Input
                    value={profile.fullName}
                    onChange={(e) =>
                      setProfile({ ...profile, fullName: e.target.value })
                    }
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Email</label>
                  <Input value={profile.email} disabled />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Department</label>
                  <Input
                    value={profile.department}
                    onChange={(e) =>
                      setProfile({ ...profile, department: e.target.value })
                    }
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Role</label>
                  <Input
                    value={profile.role}
                    onChange={(e) =>
                      setProfile({ ...profile, role: e.target.value })
                    }
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        );

      case "notifications":
        return (
          <Card>
            <CardHeader>
              <CardTitle>Notification Preferences</CardTitle>
              <CardDescription>
                Choose how you want to be notified
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {[
                {
                  key: "emailOnComplete",
                  label: "Email when estimation completes",
                },
                { key: "emailOnError", label: "Email on processing errors" },
                { key: "browserNotifications", label: "Browser notifications" },
                { key: "weeklyDigest", label: "Weekly activity digest" },
              ].map(({ key, label }) => (
                <div key={key} className="flex items-center justify-between">
                  <span className="text-sm">{label}</span>
                  <Button
                    variant={
                      notifications[key as keyof typeof notifications]
                        ? "default"
                        : "outline"
                    }
                    size="sm"
                    onClick={() =>
                      setNotifications({
                        ...notifications,
                        [key]:
                          !notifications[key as keyof typeof notifications],
                      })
                    }
                  >
                    {notifications[key as keyof typeof notifications]
                      ? "On"
                      : "Off"}
                  </Button>
                </div>
              ))}
            </CardContent>
          </Card>
        );

      case "language":
        return (
          <Card>
            <CardHeader>
              <CardTitle>Language & Region</CardTitle>
              <CardDescription>
                Set your preferred language and regional settings
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Display Language</label>
                <div className="flex gap-2">
                  <Button
                    variant={language === "en" ? "default" : "outline"}
                    onClick={() => setLanguage("en")}
                  >
                    🇬🇧 English
                  </Button>
                  <Button
                    variant={language === "it" ? "default" : "outline"}
                    onClick={() => setLanguage("it")}
                  >
                    🇮🇹 Italiano
                  </Button>
                </div>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Number Format</label>
                <p className="text-sm text-muted-foreground">
                  {language === "it" ? "1.234,56 €" : "€1,234.56"}
                </p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Date Format</label>
                <p className="text-sm text-muted-foreground">
                  {language === "it" ? "17/12/2024" : "Dec 17, 2024"}
                </p>
              </div>
            </CardContent>
          </Card>
        );

      case "appearance":
        return (
          <Card>
            <CardHeader>
              <CardTitle>Appearance</CardTitle>
              <CardDescription>Customize the look and feel</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Theme</label>
                <div className="flex gap-2">
                  <Button
                    variant={theme === "light" ? "default" : "outline"}
                    onClick={() => setTheme("light")}
                  >
                    ☀️ Light
                  </Button>
                  <Button
                    variant={theme === "dark" ? "default" : "outline"}
                    onClick={() => setTheme("dark")}
                  >
                    🌙 Dark
                  </Button>
                  <Button
                    variant={theme === "system" ? "default" : "outline"}
                    onClick={() => setTheme("system")}
                  >
                    💻 System
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        );

      case "model":
        return (
          <Card>
            <CardHeader>
              <CardTitle>ML Model Settings</CardTitle>
              <CardDescription>
                Configure machine learning behavior
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Auto-Retrain</p>
                  <p className="text-sm text-muted-foreground">
                    Automatically retrain model on feedback
                  </p>
                </div>
                <Button
                  variant={modelConfig.autoRetrain ? "default" : "outline"}
                  size="sm"
                  onClick={() =>
                    setModelConfig({
                      ...modelConfig,
                      autoRetrain: !modelConfig.autoRetrain,
                    })
                  }
                >
                  {modelConfig.autoRetrain ? "Enabled" : "Disabled"}
                </Button>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">
                  Retrain Threshold (corrections)
                </label>
                <Input
                  type="number"
                  value={modelConfig.retrainThreshold}
                  onChange={(e) =>
                    setModelConfig({
                      ...modelConfig,
                      retrainThreshold: Number(e.target.value),
                    })
                  }
                  min={1}
                  max={50}
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">
                  Confidence Threshold
                </label>
                <Input
                  type="number"
                  value={modelConfig.confidenceThreshold}
                  onChange={(e) =>
                    setModelConfig({
                      ...modelConfig,
                      confidenceThreshold: Number(e.target.value),
                    })
                  }
                  min={0.1}
                  max={1}
                  step={0.1}
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">
                  Max Similar PRs to Consider
                </label>
                <Input
                  type="number"
                  value={modelConfig.maxSimilarPRs}
                  onChange={(e) =>
                    setModelConfig({
                      ...modelConfig,
                      maxSimilarPRs: Number(e.target.value),
                    })
                  }
                  min={1}
                  max={20}
                />
              </div>

              <div className="rounded-lg bg-muted p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Current Model Version</p>
                    <p className="text-sm text-muted-foreground">
                      v2.1.0 - Trained Dec 15, 2024
                    </p>
                  </div>
                  <Button variant="outline" className="gap-2">
                    <RefreshCw className="h-4 w-4" />
                    Force Retrain
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        );

      case "api":
        return (
          <Card>
            <CardHeader>
              <CardTitle>API Configuration</CardTitle>
              <CardDescription>Manage external API keys</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">
                  OpenRouter API Key
                </label>
                <Input type="password" placeholder="sk-or-..." />
                <p className="text-xs text-muted-foreground">
                  Used for LLM inference (DeepSeek, Gemini)
                </p>
              </div>
            </CardContent>
          </Card>
        );

      case "security":
        return (
          <Card>
            <CardHeader>
              <CardTitle>Security</CardTitle>
              <CardDescription>Manage your account security</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button variant="outline" className="w-full justify-start">
                Change Password
              </Button>
              <Button variant="outline" className="w-full justify-start">
                Two-Factor Authentication
              </Button>
              <Button variant="outline" className="w-full justify-start">
                Active Sessions
              </Button>
              <Button
                variant="outline"
                className="w-full justify-start text-destructive hover:text-destructive"
              >
                Delete Account
              </Button>
            </CardContent>
          </Card>
        );

      default:
        return null;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
          <p className="mt-1 text-muted-foreground">
            Manage your preferences and configuration
          </p>
        </div>
        <Button onClick={handleSave} disabled={isSaving} className="gap-2">
          <Save className="h-4 w-4" />
          {isSaving ? "Saving..." : "Save Changes"}
        </Button>
      </div>

      {/* Content */}
      <div className="grid gap-6 lg:grid-cols-4">
        {/* Sidebar */}
        <Card className="h-fit">
          <CardContent className="p-2">
            <nav className="space-y-1">
              {sections.map((section) => {
                const Icon = section.icon;
                return (
                  <Button
                    key={section.id}
                    variant={
                      activeSection === section.id ? "secondary" : "ghost"
                    }
                    className="w-full justify-start gap-2"
                    onClick={() => setActiveSection(section.id)}
                  >
                    <Icon className="h-4 w-4" />
                    {section.title}
                  </Button>
                );
              })}
            </nav>
          </CardContent>
        </Card>

        {/* Main Content */}
        <div className="lg:col-span-3">{renderSection()}</div>
      </div>
    </div>
  );
}
