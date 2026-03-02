import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Dog Bone Blocks",
  description: "Deterministic block coding game for kids"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
