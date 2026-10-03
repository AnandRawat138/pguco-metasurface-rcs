import React, { useState } from "react";
import { get, post } from "../api.js";
import { Badge, Button, Card, Empty, ErrorBox, Field, Json, Plot, Table, fmt, inputCls, seriesColors, useAction, useFetch } from "../components.jsx";

const num = (s) => (s === "" || s === undefined ? s : isNaN(+s) ? s : +s);

function hexToRgb(h) {
  const m = /^#?([0-9a-f]{6})$/i.exec(h.trim());
  if (!m) return null;
  const n = parseInt(m[1], 16);
  return [(n >> 16) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

function Digitizer({ figure }) {
  const [cal, setCal] = useState({ x_pixels: "", x_values: "", y_pixels: "", y_values: "", x_log: false, y_log: false });
  const [color, setColor] = useState("");
  const [curve, setCurve] = useState(null);
  const act = useAction();
  const pair = (s) => s.split(",").map((v) => parseFloat(v));
  const run = () => act.run(async () => {
    const c = { x_pixels: pair(cal.x_pixels), x_values: pair(cal.x_values), y_pixels: pair(cal.y_pixels), y_values: pair(cal.y_values), x_log: cal.x_log, y_log: cal.y_log };
    if ([c.x_pixels, c.x_values, c.y_pixels, c.y_values].some((p) => p.length !== 2 || p.some(isNaN))) throw new Error("Each calibration entry needs two comma-separated numbers");
    setCurve(await post("/paper/digitize", { figure_path: figure.path, calibration_json: c, color_json: color ? hexToRgb(color) : null }));
  });
  const C = seriesColors();
  return (
    <div className="grid gap-3 md:grid-cols-2">
      <div>
        <img src={`/api/paper/figure?path=${encodeURIComponent(figure.path)}`} alt={figure.caption || "figure"} className="max-w-full border border-line rounded" />
        <p className="text-xs text-muted mt-1">Hover pixel coordinates are not shown here; read them in an image viewer (column = x pixel, row = y pixel).</p>
      </div>
      <div>
        <div className="grid grid-cols-2 gap-x-2">
          <Field label="x pixels (two columns)"><input className={inputCls} placeholder="80, 520" value={cal.x_pixels} onChange={(e) => setCal({ ...cal, x_pixels: e.target.value })} /></Field>
          <Field label="x values at those pixels"><input className={inputCls} placeholder="8, 12" value={cal.x_values} onChange={(e) => setCal({ ...cal, x_values: e.target.value })} /></Field>
          <Field label="y pixels (two rows)"><input className={inputCls} placeholder="400, 40" value={cal.y_pixels} onChange={(e) => setCal({ ...cal, y_pixels: e.target.value })} /></Field>
          <Field label="y values at those pixels"><input className={inputCls} placeholder="-25, -5" value={cal.y_values} onChange={(e) => setCal({ ...cal, y_values: e.target.value })} /></Field>
          <Field label="Trace colour (hex, blank = darkest)"><input className={inputCls} placeholder="#ff0000" value={color} onChange={(e) => setColor(e.target.value)} /></Field>
          <div className="flex items-center gap-3 text-xs mt-4">
            <label><input type="checkbox" checked={cal.x_log} onChange={(e) => setCal({ ...cal, x_log: e.target.checked })} /> log x</label>
            <label><input type="checkbox" checked={cal.y_log} onChange={(e) => setCal({ ...cal, y_log: e.target.checked })} /> log y</label>
          </div>
        </div>
        <Button kind="secondary" busy={act.busy} onClick={run}>Digitise curve</Button>
        <ErrorBox error={act.error} />
        {curve && (
          <>
            <div className="flex items-center gap-2 mt-2"><Badge fidelity="REPORTED_PLOT_DIGITIZED" /><span className="text-xs">confidence {curve.confidence}</span></div>
            <Plot height={240} data={[{ x: curve.x, y: curve.y, mode: "lines", name: "digitised", line: { color: C[3], width: 2 } }]} layout={{ showlegend: false }} />
            <p className="text-xs text-ink2">Uncertainty ±{fmt(curve.x_uncertainty)} in x, ±{fmt(curve.y_uncertainty)} in y. {curve.notes.join(" ")}</p>
          </>
        )}
      </div>
    </div>
  );
}

export function PaperEvaluation({ sid }) {
  const act = useAction();
  const [ex, setEx] = useState(null);
  const [fig, setFig] = useState(null);
  const cfg = useFetch(() => (sid ? get(`/sessions/${sid}/config`) : Promise.resolve(null)), [sid]);
  const [mapping, setMapping] = useState({});
  const [userVals, setUserVals] = useState({});
  const [rec, setRec] = useState(null);
  const recAct = useAction();
  const params = (cfg.data?.schema?.parameters || []).filter((p) => !p.fixed || true);
  const names = Array.from(new Set((ex?.values || []).map((v) => v.name)));

  return (
    <div className="grid gap-4">
      <Card title="Evaluate an existing published design" subtitle="An engineering aid for importing an existing design. It is not part of the optimisation method; the paper's optimised design is never used for training or as a target.">
        <input type="file" accept=".pdf" className="text-xs" onChange={(e) => e.target.files[0] && act.run(async () => { setEx(await post("/paper/extract", { file: e.target.files[0] })); setFig(null); setRec(null); })} />
        {act.busy && <p className="text-xs text-muted mt-2">Extracting text, tables and figures…</p>}
        <ErrorBox error={act.error} />
        {ex && <p className="text-xs text-warn mt-2">{ex.status}. {ex.n_pages} pages. {ex.warnings.join(" ")}</p>}
      </Card>
      {ex && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card title="Extracted values" subtitle="Each value keeps its source (text, table, caption) and a confidence level.">
            <Table columns={[{ label: "Name", key: "name" }, { label: "Value", render: (v) => fmt(v.value) }, { label: "Unit", key: "unit" }, { label: "Source", key: "source" }, { label: "Confidence", key: "confidence" }, { label: "Note", key: "note" }]} rows={ex.values} />
          </Card>
          <Card title="Figures" subtitle="Pick a plot to digitise. Kind is a heuristic guess.">
            {ex.figures.length ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {ex.figures.map((f) => (
                  <button key={f.path} onClick={() => setFig(f)} className={`border rounded p-1 text-left ${fig?.path === f.path ? "border-accent" : "border-line"}`}>
                    <img src={`/api/paper/figure?path=${encodeURIComponent(f.path)}`} alt="" className="w-full h-24 object-contain" />
                    <span className="block text-[10px] text-muted">p.{f.page} · {f.kind_guess}</span>
                  </button>
                ))}
              </div>
            ) : <Empty>No embedded figures found.</Empty>}
          </Card>
        </div>
      )}
      {fig && <Card title={`Digitise figure (page ${fig.page})`} subtitle={fig.caption}><Digitizer figure={fig} /></Card>}
      {ex && (
        <Card title="Reconstruct the design as a baseline" subtitle="Map schema parameters to extracted values or type them yourself. Missing values are never invented; the result carries a HIGH / MEDIUM / LOW reconstruction confidence.">
          {!sid ? <Empty>Open a study first; reconstruction targets its parameter schema.</Empty> : (
            <>
              <Table columns={[
                { label: "Parameter", key: "name" },
                { label: "Unit", key: "unit" },
                { label: "From extracted value", render: (p) => (
                  <select className={`${inputCls} w-auto`} value={mapping[p.name] || ""} onChange={(e) => setMapping({ ...mapping, [p.name]: e.target.value })}>
                    <option value="">–</option>{names.map((n) => <option key={n}>{n}</option>)}
                  </select>) },
                { label: "Or user value", render: (p) => <input className={`${inputCls} w-24`} value={userVals[p.name] ?? ""} onChange={(e) => setUserVals({ ...userVals, [p.name]: e.target.value })} /> },
              ]} rows={params} />
              <div className="mt-2"><Button kind="secondary" busy={recAct.busy} onClick={() => recAct.run(async () => {
                const m = Object.fromEntries(Object.entries(mapping).filter(([, v]) => v));
                const u = Object.fromEntries(Object.entries(userVals).filter(([, v]) => v !== "").map(([k, v]) => [k, num(v)]));
                setRec(await post("/paper/reconstruct", { sid, extracted_json: ex.values, mapping_json: m, user_values_json: u }));
              })}>Reconstruct</Button></div>
              <ErrorBox error={recAct.error || cfg.error} />
              {rec && (
                <div className="mt-3">
                  <p className="text-sm">Reconstruction confidence: <span className="font-semibold">{rec.reconstruction_confidence}</span>{rec.unresolved_parameters.length ? ` · unresolved: ${rec.unresolved_parameters.join(", ")}` : ""}</p>
                  <Json value={rec} />
                </div>
              )}
            </>
          )}
        </Card>
      )}
    </div>
  );
}

function useBenchmarks() {
  return useFetch(() => get("/benchmarks"), []);
}

export function ModelComparison() {
  const b = useBenchmarks();
  const C = [...seriesColors(), "#e87ba4", "#008300", "#4a3aa7", "#e34948"];
  const comps = (b.data || []).filter((x) => x.kind === "comparison");
  return (
    <div className="grid gap-4">
      <ErrorBox error={b.error} />
      {!comps.length && <Card><Empty>No benchmark yet. Run <code className="font-mono">python -m metasurface_optimizer.experiment &lt;config&gt; --benchmark</code>.</Empty></Card>}
      {comps.map((c) => {
        const names = Object.keys(c.curves || {});
        return (
          <React.Fragment key={c.path}>
            <Card title="Best composite vs. high-fidelity evaluations" subtitle={`${c.path} · mean ± 1 std over seeds · lower is better`}>
              <Plot height={380} data={names.flatMap((n, i) => {
                const cv = c.curves[n];
                if (!cv.mean) return [];
                const col = C[i % C.length];
                return [
                  { x: [...cv.n_hf, ...cv.n_hf.slice().reverse()], y: [...cv.mean.map((m, j) => m + cv.std[j]), ...cv.mean.map((m, j) => m - cv.std[j]).reverse()], fill: "toself", fillcolor: col + "22", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
                  { x: cv.n_hf, y: cv.mean, mode: "lines", name: n, line: { color: col, width: n.includes("PGUCO") ? 3 : 1.6 } },
                ];
              })} layout={{ xaxis: { title: "high-fidelity evaluations" }, yaxis: { title: "best composite J" } }} />
            </Card>
            <Card title="Comparison table" subtitle="All seeds are reported, including failures. 95% confidence intervals are bootstrap estimates.">
              <Table columns={Object.keys(c.rows[0] || {}).map((k) => ({ label: k, render: (r) => fmt(num(r[k])) }))} rows={c.rows} />
            </Card>
          </React.Fragment>
        );
      })}
    </div>
  );
}

export function Ablation() {
  const b = useBenchmarks();
  const abl = (b.data || []).filter((x) => x.kind === "ablation");
  const C = seriesColors();
  return (
    <div className="grid gap-4">
      <ErrorBox error={b.error} />
      {!abl.length && <Card><Empty>No ablation yet. Run <code className="font-mono">python -m metasurface_optimizer.experiment &lt;config&gt; --ablation</code>.</Empty></Card>}
      {abl.map((a) => (
        <React.Fragment key={a.path}>
          <Card title="Final best composite per variant" subtitle={`${a.path} · mean ± 1 std over seeds · lower is better`}>
            <Plot height={Math.max(260, 30 * a.rows.length + 80)} data={[{
              type: "bar", orientation: "h", y: a.rows.map((r) => r.variant).reverse(), x: a.rows.map((r) => +r.composite_mean).reverse(),
              error_x: { type: "data", array: a.rows.map((r) => +r.composite_std).reverse(), color: "#85837d" },
              marker: { color: a.rows.map((r) => (r.variant.toLowerCase().includes("reference") || r.variant.toLowerCase().includes("full") ? C[1] : C[0])).reverse() },
            }]} layout={{ xaxis: { title: "best composite J" }, margin: { l: 190, r: 16, t: 12, b: 44 } }} />
          </Card>
          <Card title="Ablation table">
            <Table columns={Object.keys(a.rows[0] || {}).map((k) => ({ label: k, render: (r) => fmt(num(r[k])) }))} rows={a.rows} />
          </Card>
        </React.Fragment>
      ))}
    </div>
  );
}
