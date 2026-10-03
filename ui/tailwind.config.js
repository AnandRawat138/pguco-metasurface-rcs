/** Colours are CSS variables (see src/index.css) so light and dark mode share one set of class names. */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        page: "var(--page)", surface: "var(--surface)", raised: "var(--raised)", line: "var(--line)",
        ink: "var(--ink)", ink2: "var(--ink2)", muted: "var(--muted)", accent: "var(--accent)",
        pred: "var(--pred)", valid: "var(--valid)", warn: "var(--warn)", bad: "var(--bad)",
      },
      fontFamily: { sans: ["Inter", "system-ui", "sans-serif"], mono: ["ui-monospace", "SFMono-Regular", "monospace"] },
    },
  },
};
