import React, { useEffect, useState } from "react";
import { get, post } from "../api.js";
import { Badge, Button, Card, Empty, ErrorBox, Field, Json, NeedSession, Plot, Table, fmt, inputCls, seriesColors, useAction, useFetch } from "../components.jsx";

export function CellSvg({ geo, color = "var(--s1)", size = 260 }) {
  if (!geo) return null;
  const [px, py] = geo.period;
  const pad = 0.04 * Math.max(px, py);
  const pts = (poly) => poly.map(([x, y]) => `${x},${-y}`).join(" ");
  return (
    <svg viewBox={`${-px / 2 - pad} ${-py / 2 - pad} ${px + 2 * pad} ${py + 2 * pad}`} width={size} height={size} role="img" aria-label="unit cell top view">
      <rect x={-px / 2} y={-py / 2} width={px} height={py} fill="var(--raised)" stroke="var(--ink2)" strokeWidth={px / 300} />
      {geo.polygons.map((p, i) => <polygon key={i} points={pts(p)} fill={color} stroke="var(--ink)" strokeWidth={px / 400} />)}
    </svg>
  );
}

function flatten(obj, prefix = "", out = []) {
  if (obj && typeof obj === "object" && !Array.isArray(obj)) {
    for (const [k, v] of Object.entries(obj)) flatten(v, prefix ? `${prefix}.${k}` : k, out);
  } else out.push({ key: prefix, value: obj });
  return out;
}

export function Geometry({ sid, st }) {
  const cfg = useFetch(() => (sid ? get(`/sessions/${sid}/config`) : Promise.resolve(null)), [sid]);
  const [which, setWhich] = useState("baseline");
  const [vals, setVals] = useState(null);
  const [geo, setGeo] = useState(null);
  const act = useAction();
  const params = cfg.data?.schema?.parameters || [];

  const load = (w, v) => act.run(async () => {
    const g = await get(`/sessions/${sid}/geometry`, v ? { values_json: JSON.stringify(v) } : { which: w });
    setGeo(g);
    setVals(g.values);
  });
  useEffect(() => {
    if (sid) load(which);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sid, which, st?.n_hf]);

  const img = useAction();
  const [imgRes, setImgRes] = useState(null);
  const [imgFile, setImgFile] = useState(null);
  const [period, setPeriod] = useState("");
  const stl = useAction();
  const [stlRes, setStlRes] = useState(null);

  return (
    <div className="grid gap-4">
      <NeedSession session={sid}>
        <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
          <Card title="Unit cell (top metal)" right={
            <select className={`${inputCls} w-auto`} value={which} onChange={(e) => setWhich(e.target.value)}>
              <option value="baseline">existing design x0</option>
              <option value="best">best evaluated</option>
            </select>}>
            <div className="flex justify-center"><CellSvg geo={geo} color={which === "best" ? "var(--s2)" : "var(--s1)"} /></div>
            {geo && (
              <div className="mt-3 text-xs space-y-1">
                <div>Validation: {geo.validation.valid ? <span className="text-valid font-medium">valid</span> : <span className="text-bad font-medium">invalid</span>}</div>
                {(geo.validation.violations || []).map((v, i) => <div key={i} className="text-bad">{typeof v === "string" ? v : JSON.stringify(v)}</div>)}
                <Table columns={[{ label: "Layer", key: "name" }, { label: "z0", key: "z0" }, { label: "Thickness", key: "thickness" }, { label: "Material", key: "material" }]} rows={geo.layers} />
              </div>
            )}
          </Card>
          <Card title="Parameter schema" subtitle="Bounds, units and step sizes come from the YAML schema. Edit values to preview a design (no simulation is run).">
            <ErrorBox error={act.error || cfg.error} />
            <Table
              columns={[
                { label: "Parameter", key: "name" },
                { label: "Value", render: (p) => (p.fixed ? fmt(vals?.[p.name]) : (
                  <input type="number" className={`${inputCls} w-28`} step={p.resolution || p.step || "any"} min={p.min} max={p.max}
                    value={vals?.[p.name] ?? ""} onChange={(e) => setVals({ ...vals, [p.name]: parseFloat(e.target.value) })} />)) },
                { label: "Unit", key: "unit" },
                { label: "Min", key: "min" },
                { label: "Max", key: "max" },
                { label: "Step", key: "step" },
                { label: "Fixed", render: (p) => (p.fixed ? "fixed" : "") },
              ]}
              rows={params}
            />
            <div className="mt-3"><Button kind="secondary" onClick={() => load(which, vals)} busy={act.busy}>Preview geometry</Button></div>
          </Card>
        </div>
      </NeedSession>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Estimate parameters from an image or mask" subtitle="An approximate estimate with a confidence level; verify every value before using it.">
          <div className="flex flex-wrap items-end gap-2">
            <input type="file" accept=".png,.jpg,.jpeg,.bmp,.npy" className="text-xs" onChange={(e) => setImgFile(e.target.files[0])} />
            <Field label="Period (mm), sets the scale"><input className={`${inputCls} w-28`} value={period} onChange={(e) => setPeriod(e.target.value)} /></Field>
            <div className="mb-2"><Button kind="secondary" disabled={!imgFile} busy={img.busy} onClick={() => img.run(async () => setImgRes(await post("/geometry/image", { file: imgFile, period_mm: period || null })))}>Analyse</Button></div>
          </div>
          <ErrorBox error={img.error} />
          {imgRes && <Json value={imgRes} />}
        </Card>
        <Card title="Inspect an STL" subtitle="Mesh-to-parameter inference is approximate and never unique; the original file is preserved.">
          <input type="file" accept=".stl" className="text-xs" onChange={(e) => e.target.files[0] && stl.run(async () => setStlRes(await post("/geometry/stl", { file: e.target.files[0] })))} />
          <ErrorBox error={stl.error} />
          {stl.busy && <p className="text-xs text-muted mt-2">Analysing…</p>}
          {stlRes && <Json value={stlRes} />}
        </Card>
      </div>
    </div>
  );
}

const FIDS = ["VALIDATED_SIMULATION", "MEASURED", "REPORTED", "REPORTED_PLOT_DIGITIZED"];

export function ResponsePlots({ resp, label, other, otherLabel, predicted }) {
  const C = seriesColors();
  if (!resp) return null;
  const sets = [[resp, label || "baseline", C[0]], ...(other ? [[other, otherLabel || "optimized", C[1]]] : [])];
  const spKeys = Object.keys(resp.sparams || {}).filter((k) => k.startsWith("S11")).slice(0, 2);
  const traces = (field) => sets.flatMap(([r, lab, col]) => spKeys.filter((k) => r.sparams?.[k]).map((k, j) => ({
    x: r.frequency.map((f) => f / 1e9), y: r.sparams[k][field], mode: "lines", name: `${lab} ${k}`, line: { color: col, width: 2, dash: j ? "dot" : "solid" },
  })));
  const red = (r) => {
    const s = r.rcs?.specular_dbsm, ref = r.rcs?.reference_dbsm;
    if (!s || !ref) return null;
    return s[0][0].map((v, i) => (v === null || ref[0][0][i] === null ? null : v - ref[0][0][i]));
  };
  const redTraces = sets.map(([r, lab, col]) => ({ r, lab, col, y: red(r) })).filter((t) => t.y)
    .map(({ r, lab, col, y }) => ({ x: r.rcs.frequency.map((f) => f / 1e9), y, mode: "lines", name: lab, line: { color: col, width: 2 } }));
  if (predicted) {
    const [name, ys] = Object.entries(predicted.rcs_reduction_db)[0] || [];
    const sd = predicted.rcs_reduction_db_std?.[name];
    if (ys) {
      const f = predicted.frequency.map((v) => v / 1e9);
      if (sd) {
        redTraces.push({ x: [...f, ...f.slice().reverse()], y: [...ys.map((v, i) => v + 2 * sd[i]), ...ys.map((v, i) => v - 2 * sd[i]).reverse()], fill: "toself", fillcolor: "rgba(27,175,122,0.15)", line: { width: 0 }, name: "PREDICTED ±2σ", hoverinfo: "skip" });
      }
      redTraces.push({ x: f, y: ys, mode: "lines", name: `PREDICTED ${name.replace("RCSred_", "")}`, line: { color: C[2], width: 2, dash: "dash" } });
    }
  }
  const bi = resp.rcs?.bistatic_center;
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card title="Reflection magnitude |S11|" subtitle="solid: first state, dotted: second state">
        {spKeys.length ? <Plot data={traces("db")} layout={{ xaxis: { title: "frequency (GHz)" }, yaxis: { title: "|S11| (dB)" } }} /> : <Empty>No S-parameters.</Empty>}
      </Card>
      <Card title="Reflection phase">
        {spKeys.length ? <Plot data={traces("phase")} layout={{ xaxis: { title: "frequency (GHz)" }, yaxis: { title: "phase (deg)" } }} /> : <Empty>No S-parameters.</Empty>}
      </Card>
      <Card title="Specular RCS reduction vs. reference plate" subtitle="first polarization, first incidence angle">
        {redTraces.length ? <Plot data={redTraces} layout={{ xaxis: { title: "frequency (GHz)" }, yaxis: { title: "σ / σ_ref (dB)" } }} /> : <Empty>No specular RCS with a reference.</Empty>}
      </Card>
      <Card title={bi ? `Bistatic RCS at ${(bi.frequency / 1e9).toFixed(2)} GHz` : "Bistatic RCS"} subtitle={label}>
        {bi ? (
          <Plot data={[{ type: "heatmap", x: bi.phi_s, y: bi.theta_s, z: bi.dbsm, colorscale: "Blues", reversescale: true, colorbar: { title: { text: "dBsm", side: "right" } } }]}
            layout={{ xaxis: { title: "φ_s (deg)" }, yaxis: { title: "θ_s (deg)" } }} />
        ) : <Empty>No bistatic data.</Empty>}
      </Card>
    </div>
  );
}

export function EMResponse({ sid, st, refresh }) {
  const [fid, setFid] = useState("VALIDATED_SIMULATION");
  const [sp, setSp] = useState(null);
  const [rcs, setRcs] = useState(null);
  const [unit, setUnit] = useState("ghz");
  const act = useAction();
  const base = useFetch(() => (sid && st?.baseline ? get(`/sessions/${sid}/baseline_analysis`) : Promise.resolve(null)), [sid, !!st?.baseline]);
  const hasData = !!st?.baseline;
  const set = (fields) => act.run(async () => {
    await post(`/sessions/${sid}/baseline`, fields);
    await refresh();
    base.reload();
  });
  return (
    <NeedSession session={sid}>
      <div className="grid gap-4">
        <Card title="Response of the existing design (y0)" subtitle="Upload Touchstone (.sNp) or CSV S-parameters and/or an RCS CSV with its true fidelity, or use the synthetic oracle in demo mode.">
          {hasData ? <p className="text-sm text-ink2">The baseline is set ({st.baseline.label}). Create a new study to change it.</p> : (
            <div className="flex flex-wrap items-center gap-2">
              <select className={`${inputCls} w-auto`} value={fid} onChange={(e) => setFid(e.target.value)}>{FIDS.map((f) => <option key={f}>{f}</option>)}</select>
              <label className="text-xs">S-params <input type="file" className="text-xs" onChange={(e) => setSp(e.target.files[0])} /></label>
              <label className="text-xs">RCS <input type="file" className="text-xs" onChange={(e) => setRcs(e.target.files[0])} /></label>
              <select className={`${inputCls} w-auto`} value={unit} onChange={(e) => setUnit(e.target.value)}>{["ghz", "mhz", "hz"].map((u) => <option key={u}>{u}</option>)}</select>
              <Button onClick={() => set({ fidelity: fid, sparams: sp, rcs, freq_unit: unit })} disabled={!sp && !rcs} busy={act.busy}>Load response</Button>
              <span className="text-xs text-muted">or</span>
              <Button kind="secondary" onClick={() => set({ use_oracle: "true" })} busy={act.busy}>Use synthetic oracle (demo)</Button>
            </div>
          )}
          <ErrorBox error={act.error || base.error} />
        </Card>
        {base.data && (
          <>
            <div className="flex items-center gap-2"><span className="text-sm text-ink2">Data shown:</span><Badge fidelity={base.data.response.fidelity} /></div>
            <ResponsePlots resp={base.data.response} label={`baseline (${base.data.response.label})`} />
          </>
        )}
      </div>
    </NeedSession>
  );
}

export function SensitivityChart({ sens, title }) {
  const C = seriesColors();
  if (!sens) return <Empty>Not computed yet. It is computed after the initial design.</Empty>;
  const rows = sens.ranking.slice().reverse();
  return (
    <Plot height={Math.max(220, 28 * rows.length + 60)}
      data={[{ type: "bar", orientation: "h", x: rows.map((r) => r.sensitivity), y: rows.map((r) => r.parameter), marker: { color: C[0] }, hovertemplate: "%{y}: %{x:.3f}<extra></extra>" }]}
      layout={{ title, xaxis: { title: "normalised local response sensitivity" }, margin: { l: 130, r: 16, t: 36, b: 44 } }} />
  );
}

export function BaselineAnalysis({ sid, st }) {
  const base = useFetch(() => (sid && st?.baseline ? get(`/sessions/${sid}/baseline_analysis`) : Promise.resolve(null)), [sid, st?.n_hf, !!st?.baseline]);
  const d = base.data;
  return (
    <NeedSession session={sid}>
      {!d ? <Empty>Load the baseline response first (EM Response tab).</Empty> : (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card title="Objectives at the existing design" subtitle="Raw objective values J0 (lower is better after sign convention).">
            <Table columns={[{ label: "Objective", key: "name" }, { label: "J0", render: (r) => fmt(r.v) }]} rows={d.objective_names.map((n, i) => ({ name: n, v: d.J0[i] }))} />
            <div className="mt-2"><Badge fidelity={d.response.fidelity} /></div>
          </Card>
          <Card title="Response features" subtitle="Bandwidths, resonances and RCS statistics extracted from y0.">
            <div className="max-h-80 overflow-auto">
              <Table columns={[{ label: "Feature", key: "key" }, { label: "Value", render: (r) => fmt(r.value) }]} rows={flatten(d.summary)} max={300} />
            </div>
          </Card>
          <Card title="Which parameters matter here?" subtitle="Central-difference sensitivities from evaluated designs. They are local and data-associated, not causal effects.">
            <SensitivityChart sens={d.hf_sensitivity} title="from high-fidelity evaluations" />
          </Card>
          <Card title="Surrogate sensitivities" subtitle="PREDICTED: computed on the surrogate around the current centre.">
            <SensitivityChart sens={d.surrogate_sensitivity} title="from the surrogate (PREDICTED)" />
          </Card>
        </div>
      )}
      <ErrorBox error={base.error} />
    </NeedSession>
  );
}
