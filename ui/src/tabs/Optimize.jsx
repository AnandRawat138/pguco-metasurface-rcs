import React, { useEffect, useState } from "react";
import { fileUrl, get, post } from "../api.js";
import { Badge, Button, Card, Empty, ErrorBox, Field, Json, NeedSession, Plot, Stat, Table, fmt, inputCls, seriesColors, useAction, useFetch } from "../components.jsx";
import { CellSvg, ResponsePlots } from "./Design.jsx";

function ConfigEditor({ sid, locked, onSaved }) {
  const cfg = useFetch(() => get(`/sessions/${sid}/config`), [sid]);
  const [text, setText] = useState("");
  const act = useAction();
  useEffect(() => {
    if (cfg.data) {
      const c = cfg.data.config;
      setText(JSON.stringify({ objectives: c.objectives, normalization: c.normalization, physics: c.physics, optimizer: c.optimizer, transfer: c.transfer, high_fidelity: c.high_fidelity }, null, 2));
    }
  }, [cfg.data]);
  return (
    <Card title="Objectives, physics and optimiser settings" subtitle={locked ? "Locked: the study already has data. Create a new study to change the configuration." : "Edit before loading the baseline. Weights, bands, normalisation, acquisition and stopping rules are all here."}>
      <textarea className={`${inputCls} font-mono text-[11px] h-72`} value={text} onChange={(e) => setText(e.target.value)} disabled={locked} spellCheck={false} />
      <ErrorBox error={act.error || cfg.error} />
      {!locked && <div className="mt-2"><Button kind="secondary" busy={act.busy} onClick={() => act.run(async () => { await post(`/sessions/${sid}/config`, { config_json: JSON.parse(text) }, "PUT"); onSaved(); })}>Save configuration</Button></div>}
    </Card>
  );
}

export function Optimization({ sid, st, refresh, go }) {
  const act = useAction();
  const [k, setK] = useState(4);
  const [msg, setMsg] = useState(null);
  const call = (path, fields, after) => act.run(async () => {
    const r = await post(`/sessions/${sid}${path}`, fields);
    setMsg(after ? after(r) : null);
    await refresh();
  });
  const state = st?.status?.state;
  const pendingOpen = (st?.pending || []).filter((d) => !d.done).length;
  const hasBaseline = !!st?.baseline;
  const isOracle = st?.baseline?.label?.includes("SYNTHETIC");
  const hist = st?.history || [];
  const C = seriesColors();

  return (
    <NeedSession session={sid}>
      <div className="grid gap-4">
        <div className="grid gap-4 grid-cols-2 lg:grid-cols-4">
          <Stat label="State" value={<span className="text-lg">{state || "–"}</span>} sub={st?.status?.stop_reason ? `stopped: ${st.status.stop_reason}` : ""} />
          <Stat label="High-fidelity evaluations" value={`${st?.n_hf ?? 0} / ${st?.budget ?? "–"}`} />
          <Stat label="Trust-region radius" value={fmt(st?.trust_region_radius)} sub="normalised parameter space" />
          <Stat label="Designs awaiting results" value={pendingOpen} />
        </div>
        <Card title="Mode B: human-in-the-loop optimisation" subtitle="Each step proposes a few designs. You validate them in HFSS/CST and upload the results; the surrogate learns from validated data only.">
          {!hasBaseline && <p className="text-sm text-warn mb-2">Load the baseline response first (<button className="underline" onClick={() => go("EM Response")}>EM Response</button>).</p>}
          <div className="flex flex-wrap items-end gap-2">
            <Button disabled={!hasBaseline || state !== "baseline_set"} busy={act.busy} onClick={() => call("/initial_design", {}, (r) => `${r.designs.length} initial designs planned. ${r.instruction}`)}>1. Plan initial designs</Button>
            <Button kind="secondary" disabled={!pendingOpen} busy={act.busy} onClick={() => go("Dataset")}>Upload results</Button>
            {isOracle && <Button kind="secondary" disabled={!pendingOpen} busy={act.busy} onClick={() => call("/evaluate_pending_with_oracle", {}, (r) => `${r.evaluated} designs evaluated with the SYNTHETIC ORACLE (not full-wave).`)}>Evaluate with synthetic oracle (demo)</Button>}
            <Button disabled={state !== "awaiting_initial_results" || pendingOpen > 0} busy={act.busy} onClick={() => call("/finish_initial", {}, () => "Initial design complete; local sensitivities computed.")}>2. Finish initial design</Button>
            <Field label="Batch size"><input type="number" min={1} max={16} className={`${inputCls} w-20`} value={k} onChange={(e) => setK(e.target.value)} /></Field>
            <Button disabled={!["ready", "awaiting_results"].includes(state) || (state === "awaiting_results" && pendingOpen > 0)} busy={act.busy} onClick={() => call("/propose", { k }, (r) => `${r.candidates.length} counterfactual candidates proposed (${r.label}).`)}>3. Propose candidates</Button>
            <Button disabled={state !== "awaiting_results" || pendingOpen > 0} busy={act.busy} onClick={() => call("/complete_batch", {}, (r) => (r.stop_reason ? `Stopping rule met: ${r.stop_reason}` : "Batch recorded; the trust region was updated."))}>4. Complete batch</Button>
          </div>
          {state === "awaiting_results" && pendingOpen > 0 && <p className="text-xs text-ink2 mt-2">Proposed designs are listed on the <button className="underline text-accent" onClick={() => go("Candidate Designs")}>Candidate Designs</button> tab.</p>}
          {msg && <p className="text-sm text-ink2 mt-3">{msg}</p>}
          <ErrorBox error={act.error} />
        </Card>
        <div className="grid gap-4 lg:grid-cols-2">
          <Card title="Automatic run (configured high-fidelity source)" subtitle="Runs the whole loop against the configured evaluator until a stopping rule fires. In the demo that is the synthetic oracle; HFSS/CST automation is not implemented.">
            <Button kind="secondary" disabled={!hasBaseline || state === "running"} busy={act.busy} onClick={() => call("/run_auto", {}, () => "Automatic run started; this page refreshes while it runs.")}>Run automatically</Button>
            {state === "running" && <p className="text-xs text-muted mt-2">Running…</p>}
            {st?.status?.result_summary && <Json value={st.status.result_summary} />}
          </Card>
          <Card title="Batch history">
            {hist.length ? (
              <Plot height={260} data={[
                { x: hist.map((h) => h.n_hf), y: hist.map((h) => h.best_composite), mode: "lines+markers", name: "best evaluated composite", line: { color: C[1], width: 2 }, marker: { size: 8 } },
              ]} layout={{ xaxis: { title: "high-fidelity evaluations" }, yaxis: { title: "composite J" }, showlegend: false }} />
            ) : <Empty>No completed batches yet.</Empty>}
            <Table columns={[{ label: "Iter", key: "iteration" }, { label: "N_HF", key: "n_hf" }, { label: "Best", render: (h) => fmt(h.best_composite) }, { label: "HV", render: (h) => fmt(h.hypervolume) }, { label: "Radius", render: (h) => fmt(h.radius) }, { label: "Status", key: "status" }]} rows={hist.slice().reverse()} />
          </Card>
        </div>
        <ConfigEditor sid={sid} locked={hasBaseline} onSaved={refresh} />
      </div>
    </NeedSession>
  );
}

function CandidateTable({ cands, names }) {
  const [open, setOpen] = useState(null);
  return (
    <>
      <Table
        columns={[
          { label: "#", render: (_, i) => i + 1 },
          { label: "Kind", key: "kind" },
          { label: "Modified parameters", render: (c) => c.modifications.map((m) => `${m.parameter} ${m.change > 0 ? "+" : ""}${fmt(m.change, 3)}`).join(", ") || "–" },
          { label: "Predicted J (μ ± σ)", render: (c) => `${fmt(c.mu)} ± ${fmt(c.sigma)}` },
          { label: "vs. incumbent", render: (c) => `${fmt(c.predicted_composite_change_pct_vs_incumbent, 1)}%` },
          { label: "Acquisition", render: (c) => fmt(c.acquisition?.score) },
          { label: "Why", key: "reason" },
          { label: "", render: (c, i) => <button className="text-accent underline" onClick={() => setOpen(open === i ? null : i)}>{open === i ? "hide" : "details"}</button> },
        ]}
        rows={cands}
      />
      {open !== null && cands[open] && (
        <div className="grid gap-3 md:grid-cols-2 mt-3 border-t border-line pt-3">
          <div className="text-xs space-y-2">
            <div><Badge fidelity="PREDICTED" /> <span className="text-ink2">{cands[open].label}</span></div>
            <p>{cands[open].predicted_effect}</p>
            <Table columns={[{ label: "Objective", key: "n" }, { label: "μ", render: (r) => fmt(r.m) }, { label: "σ", render: (r) => fmt(r.s) }]}
              rows={names.map((n, i) => ({ n, m: cands[open].mu_J[i], s: cands[open].sigma_J[i] }))} />
            <Json value={{ values: cands[open].values, modifications: cands[open].modifications, acquisition: cands[open].acquisition, physics_audit: cands[open].physics_audit, detail: cands[open].detail }} />
          </div>
          <CandidateGeometry values={cands[open].values} />
        </div>
      )}
    </>
  );
}

function CandidateGeometry({ values }) {
  const sid = (() => {
    try {
      return localStorage.getItem("msopt.sid");
    } catch (_) {
      return null;
    }
  })();
  const g = useFetch(() => (sid ? get(`/sessions/${sid}/geometry`, { values_json: JSON.stringify(values) }) : Promise.resolve(null)), [JSON.stringify(values)]);
  return <div className="flex flex-col items-center"><CellSvg geo={g.data} color="var(--s3)" /><span className="text-xs text-muted">candidate geometry (not yet validated)</span></div>;
}

export function Candidates({ sid, st }) {
  const act = useAction();
  const [modeA, setModeA] = useState(null);
  const [k, setK] = useState(10);
  const names = st?.objective_names || [];
  return (
    <NeedSession session={sid}>
      <div className="grid gap-4">
        <Card title="Latest counterfactual candidates" subtitle="Every number here is a surrogate prediction. A candidate is only an improvement once it has been simulated and uploaded." right={<Badge fidelity="PREDICTED" />}>
          {st?.proposals?.length ? <CandidateTable cands={st.proposals} names={names} /> : <Empty>No proposals yet. Use “Propose candidates” on the Optimization tab.</Empty>}
        </Card>
        <Card title="Mode A: prediction only" subtitle="Rank candidates from the data you already have, without running new simulations. Requires a trained surrogate (at least one proposal step)." right={<Badge fidelity="PREDICTED" />}>
          <div className="flex items-end gap-2">
            <Field label="Number of candidates"><input type="number" className={`${inputCls} w-20`} value={k} onChange={(e) => setK(e.target.value)} /></Field>
            <div className="mb-2"><Button kind="secondary" busy={act.busy} onClick={() => act.run(async () => setModeA(await post(`/sessions/${sid}/predict_only`, { k })))}>Rank candidates</Button></div>
          </div>
          <ErrorBox error={act.error} />
          {modeA && <CandidateTable cands={modeA.candidates} names={names} />}
        </Card>
      </div>
    </NeedSession>
  );
}

export function Results({ sid, st }) {
  const cmp = useFetch(() => (sid && st?.n_hf > 1 ? get(`/sessions/${sid}/compare`) : Promise.resolve(null)), [sid, st?.n_hf]);
  const names = st?.objective_names || [];
  const [pair, setPair] = useState([0, 1]);
  const C = seriesColors();
  const recs = st?.records || [];
  const log = st?.prediction_log || [];
  const d = cmp.data;
  return (
    <NeedSession session={sid}>
      {!d ? <Empty>Results appear once designs beyond the baseline have been evaluated.</Empty> : (
        <div className="grid gap-4">
          <Card title="Baseline vs. best evaluated design" right={<div className="flex gap-1"><Badge fidelity={d.baseline.fidelity} /><Badge fidelity={d.optimized.fidelity} />{d.predicted && <Badge fidelity="PREDICTED" />}</div>}
            subtitle="Both curves come from evaluated data with their own labels. The dashed curve is the surrogate's prediction for the same design.">
            <Table columns={[{ label: "Objective", key: "n" }, { label: "Baseline J0", render: (r) => fmt(r.a) }, { label: "Best evaluated", render: (r) => fmt(r.b) }, { label: "Change", render: (r) => fmt(r.b - r.a) }]}
              rows={names.map((n, i) => ({ n, a: d.J0[i], b: d.J_best[i] }))} />
          </Card>
          <ResponsePlots resp={d.baseline} label="baseline" other={d.optimized} otherLabel="best evaluated" predicted={d.predicted} />
          <div className="grid gap-4 lg:grid-cols-2">
            <Card title="Trade-offs between objectives" subtitle="Normalised objectives, lower is better. The candidate pool is PREDICTED." right={
              <div className="flex gap-1">
                {[0, 1].map((a) => (
                  <select key={a} className={`${inputCls} w-auto text-xs`} value={pair[a]} onChange={(e) => setPair(a ? [pair[0], +e.target.value] : [+e.target.value, pair[1]])}>
                    {names.map((n, i) => <option key={n} value={i}>{n}</option>)}
                  </select>))}
              </div>}>
              <Plot data={[
                ...(st.pool_pred_norm ? [{ x: st.pool_pred_norm.map((r) => r[pair[0]]), y: st.pool_pred_norm.map((r) => r[pair[1]]), mode: "markers", name: "candidate pool (PREDICTED)", marker: { color: C[2], size: 6, opacity: 0.35 } }] : []),
                { x: recs.map((r) => r.J_norm[pair[0]]), y: recs.map((r) => r.J_norm[pair[1]]), text: recs.map((r) => r.tag), mode: "markers", name: "evaluated", marker: { color: C[0], size: 9, line: { color: "white", width: 1 } } },
                { x: recs.filter((r) => r.pareto).map((r) => r.J_norm[pair[0]]), y: recs.filter((r) => r.pareto).map((r) => r.J_norm[pair[1]]), mode: "markers", name: "evaluated Pareto set", marker: { color: C[1], size: 11, line: { color: "white", width: 1 } } },
                { x: [recs[0]?.J_norm[pair[0]]], y: [recs[0]?.J_norm[pair[1]]], mode: "markers", name: "baseline x0", marker: { symbol: "star", size: 14, color: "#52514e" } },
              ]} layout={{ xaxis: { title: names[pair[0]] }, yaxis: { title: names[pair[1]] } }} />
            </Card>
            <Card title="Were the predictions right?" subtitle="Out-of-sample: each prediction was logged before its design was evaluated. Error bars are ±2σ.">
              {log.length ? (() => {
                const v = log.map((p) => p.validated_composite), p = log.map((p) => p.predicted_composite);
                const lo = Math.min(...v, ...p), hi = Math.max(...v, ...p);
                const rmse = Math.sqrt(v.reduce((s, x, i) => s + (x - p[i]) ** 2, 0) / v.length);
                return (
                  <>
                    <Plot data={[
                      { x: [lo, hi], y: [lo, hi], mode: "lines", name: "perfect prediction", line: { color: "#85837d", dash: "dot", width: 1.5 } },
                      { x: v, y: p, mode: "markers", name: "candidates", marker: { color: C[0], size: 9 }, error_y: { type: "data", array: log.map((q) => 2 * (q.sigma_composite ?? 0)), color: "#85837d", thickness: 1 } },
                    ]} layout={{ xaxis: { title: "evaluated composite" }, yaxis: { title: "PREDICTED composite" } }} />
                    <p className="text-xs text-ink2">RMSE {fmt(rmse)} over {log.length} predictions.</p>
                  </>
                );
              })() : <Empty>No out-of-sample predictions yet.</Empty>}
            </Card>
          </div>
        </div>
      )}
      <ErrorBox error={cmp.error} />
    </NeedSession>
  );
}

export function Export({ sid, st }) {
  const act = useAction();
  const [res, setRes] = useState(null);
  const groups = {};
  for (const f of res?.files || []) {
    const parts = f.split("/");
    const g = parts.length > 2 ? parts[1] : "";
    (groups[g] = groups[g] || []).push(f);
  }
  return (
    <NeedSession session={sid}>
      <Card title="Export the study" subtitle="Writes baseline and optimised parameters, per-layer STL files with JSON sidecars, candidate tables, plots, metrics, the report (JSON, Markdown, PDF) and the reproducibility record.">
        <Button busy={act.busy} disabled={!st?.n_hf} onClick={() => act.run(async () => setRes(await post(`/sessions/${sid}/export`)))}>Export now</Button>
        <ErrorBox error={act.error} />
        {res && (
          <div className="mt-4 grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            {Object.entries(groups).map(([g, fs]) => (
              <div key={g} className="border border-line rounded-md p-3">
                <h3 className="text-xs font-semibold mb-1">{g || "top level"}</h3>
                <ul className="text-xs space-y-0.5">
                  {fs.map((f) => <li key={f}><a className="text-accent underline break-all" href={fileUrl(f)} target="_blank" rel="noreferrer">{f.split("/").pop()}</a></li>)}
                </ul>
              </div>
            ))}
          </div>
        )}
        <p className="text-xs text-muted mt-3">STL files describe geometry only. Import them into HFSS or CST, assign the materials listed in the sidecar JSON, and simulate; the exported design is PREDICTED until you do.</p>
      </Card>
    </NeedSession>
  );
}
