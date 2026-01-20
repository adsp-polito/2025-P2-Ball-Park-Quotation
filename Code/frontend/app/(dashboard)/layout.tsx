import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import { MainContent } from "@/components/layout/main-content";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-background">
      {/* Sidebar */}
      <Sidebar />

      {/* Main content - responds to sidebar collapse state */}
      <MainContent>
        <Header />
        <main className="p-4 xl:p-6">{children}</main>
      </MainContent>
    </div>
  );
}
