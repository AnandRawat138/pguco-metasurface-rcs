import React, { useEffect, useState } from "react";
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

const PlotlyComponent = createPlotlyComponent(Plotly);

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// Fixed categorical order: baseline, optimized/validated, predicted, further series.
export function seriesColors() {
  return [cssVar("--s1"), cssVar("--s2"), cssVar("--s3"), cssVar("--s4")];
}

export function Plot({ data, layout = {}, height = 320 }) {
  const ink2 = cssVar("--ink2");
  const line = cssVar("--line");
  const axis = { gridcolor: line, zerolinecolor: line, linecolor: ink2, tickfont: { color: ink2, size: 11 }, title: { font: { size: 12, color: ink2 } } };
  const merge = (k) => ({ ...axis, ...(layout[k] || {}), title: { ...axis.title, ...(typeof layout[k]?.title === "string" ? { text: layout[k].title } : layout[k]?.title || {}) } });
  return (
    <PlotlyComponent
      data={data}
      layout={{
        autosize: true, height, margin: { l: 56, r: 16, t: layout.title ? 36 : 12, b: 44 },
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: "Inter, system-ui, sans-serif", color: cssVar("--ink"), size: 12 },
        legend: { orientation: "h", y: -0.22, font: { size: 11 } },
        hovermode: "closest",
        ...layout,
        title: layout.title ? { text: layout.title, font: { size: 13 }, x: 0, xanchor: "left" } : undefined,
        xaxis: merge("xaxis"), yaxis: merge("yaxis"),
      }}
      config={{ displaylogo: false, responsive: true }}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}

export function Card({ title, subtitle, children, right, className = "" }) {
  return (
    <section className={`bg-surface border border-line rounded-lg p-4 min-w-0 ${className}`}>
      {(title || right) && (
        <header className="flex flex-wrap items-start justify-between gap-x-3 gap-y-2 mb-3">
          <div className="min-w-0 flex-1 basis-56">
            {title && <h2 className="text-sm font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="text-xs text-muted mt-0.5">{subtitle}</p>}
          </div>
          {right && <div className="flex flex-wrap gap-1 max-w-full">{right}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

// Fidelity badge: predicted values are always visibly distinct from validated / reported ones.
const FID = {
  PREDICTED: ["PREDICTED", "text-pred border-pred"],
  VALIDATED_SIMULATION: ["VALIDATED (simulation)", "text-valid border-valid"],
  MEASURED: ["MEASURED", "text-valid border-valid"],
  REPORTED: ["REPORTED", "text-warn border-warn"],
  REPORTED_PLOT_DIGITIZED: ["REPORTED (plot-digitized)", "text-warn border-warn"],
  SYNTHETIC_ANALYTIC_ORACLE: ["SYNTHETIC ORACLE (not full-wave)", "text-warn border-warn"],
};

export function Badge({ fidelity, children }) {
  const f = String(fidelity || "").toUpperCase();
  const key = f.includes("SYNTHETIC") ? "SYNTHETIC_ANALYTIC_ORACLE"
    : f.includes("PREDICTED") ? "PREDICTED"
    : f.includes("PLOT") ? "REPORTED_PLOT_DIGITIZED"
    : f.includes("REPORTED") ? "REPORTED"
    : f.includes("MEASURED") ? "MEASURED"
    : f.includes("VALIDATED") ? "VALIDATED_SIMULATION" : null;
  const [text, cls] = key ? FID[key] : [fidelity, "text-ink2 border-line"];
  return <span title={String(fidelity || "")} className={`inline-block text-[10px] font-semibold tracking-wide uppercase border rounded px-1.5 py-0.5 whitespace-nowrap ${cls}`}>{children || text}</span>;
}

export function Button({ children, onClick, kind = "primary", disabled, busy }) {
  const base = "text-sm rounded-md px-3 py-1.5 font-medium transition disabled:opacity-40 disabled:cursor-not-allowed";
  const cls = kind === "primary" ? "bg-accent text-white hover:opacity-90" : "border border-line text-ink hover:bg-raised";
  return (
    <button className={`${base} ${cls}`} onClick={onClick} disabled={disabled || busy}>
      {busy ? "Working…" : children}
    </button>
  );
}

export function Stat({ label, value, sub, badge }) {
  return (
    <div className="bg-surface border border-line rounded-lg p-4 min-w-0">
      <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1">
        <span className="text-xs text-muted">{label}</span>
        {badge}
      </div>
      <div className="text-2xl font-semibold mt-1 tabular-nums">{value}</div>
      {sub && <div className="text-xs text-ink2 mt-0.5">{sub}</div>}
    </div>
  );
}

export function Table({ columns, rows, max = 400 }) {
  if (!rows || !rows.length) return <Empty>No rows yet.</Empty>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-muted border-b border-line">
            {columns.map((c) => (
              <th key={c.key || c.label} className="py-1.5 pr-3 font-medium whitespace-nowrap">{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, max).map((r, i) => (
            <tr key={i} className="border-b border-line/60 hover:bg-raised">
              {columns.map((c) => (
                <td key={c.key || c.label} className="py-1.5 pr-3 align-top">{c.render ? c.render(r, i) : fmt(r[c.key])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function fmt(v, d = 3) {
  if (v === null || v === undefined) return "–";
  if (typeof v === "number" && Number.isInteger(v) && Math.abs(v) < 1e6) return String(v);
  if (typeof v === "number") return Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : v.toFixed(d);
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (Array.isArray(v)) return v.map((x) => fmt(x, d)).join(", ");
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

export function Empty({ children }) {
  return <p className="text-sm text-muted py-6 text-center">{children}</p>;
}

export function ErrorBox({ error }) {
  if (!error) return null;
  return <div className="text-sm text-bad border border-bad/40 rounded-md p-2 my-2 whitespace-pre-wrap break-words">{String(error)}</div>;
}

export function Field({ label, children, hint }) {
  return (
    <label className="block text-xs text-ink2 mb-2">
      <span className="block mb-1">{label}</span>
      {children}
      {hint && <span className="block text-muted mt-0.5">{hint}</span>}
    </label>
  );
}

export const inputCls = "w-full bg-raised border border-line rounded-md px-2 py-1 text-sm text-ink";

export function Json({ value }) {
  return <pre className="text-[11px] bg-raised border border-line rounded-md p-2 overflow-auto max-h-96">{JSON.stringify(value, null, 2)}</pre>;
}

// Runs an async action, tracking busy/error state for the caller.
export function useAction() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const run = async (fn) => {
    setBusy(true);
    setError(null);
    try {
      return await fn();
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setBusy(false);
    }
  };
  return { busy, error, run, setError };
}

export function useFetch(fn, deps) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let live = true;
    setError(null);
    fn()
      .then((d) => live && setData(d))
      .catch((e) => live && setError(e.message));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  return { data, error, reload: () => setTick((t) => t + 1) };
}

export function NeedSession({ session, children }) {
  if (!session) return <Empty>Create or open a study on the Dashboard first.</Empty>;
  return children;
}
