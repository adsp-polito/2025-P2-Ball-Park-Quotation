import { redirect } from "next/navigation";

export default function RootPage() {
  // Redirect to dashboard or login
  redirect("/dashboard");
}
