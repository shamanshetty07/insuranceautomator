import type { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary";

const styles: Record<Variant, string> = {
  primary: "bg-slate-900 text-white hover:bg-slate-700 disabled:bg-slate-400",
  secondary: "border border-slate-300 bg-white text-slate-800 hover:bg-slate-50 disabled:opacity-50",
};

export function Button({
  variant = "primary",
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      className={`rounded-md px-4 py-2 text-sm font-medium focus:outline-2 focus:outline-offset-2 focus:outline-slate-900 ${styles[variant]} ${className}`}
      {...props}
    />
  );
}
