import React, { useEffect, useState } from "react";
import { get, post } from "../api.js";
import { Badge, Button, Card, Empty, ErrorBox, Field, Json, NeedSession, Plot, Stat, Table, fmt, inputCls, seriesColors, useAction, useFetch } from "../components.jsx";

// High-Fidelity Simulation tab: two-PC HFSS automation (templates, jobs, queue, budget, predicted vs simulated).

const STATE_CLS = {
  COMPLETED: "text-valid", FAILED: "text-bad", FAILED_VALIDATION: "text-bad", TIMEOUT: "text-bad", CANCELLED: "text-muted",
  PENDING: "text-ink2", CLAIMED: "text-warn", RUNNING: "text-warn", SOLVING: "text-warn", EXTRACTING: "text-warn",
};
const TSTATUS = { VERIFIED: "text-valid border-valid", UNVERIFIED: "text-warn border-warn", VERIFICATION_FAILED: "text-bad border-bad" };

function TemplateBadge({ status }) {
  const cls = TSTATUS[status] || "text-bad border-bad";
  return <span className={`inline-block text-[10px] font-semibold uppercase border rounded px-1.5 py-0.5 whitespace-nowrap ${cls}`}>{status}</span>;
}

function upload(path, file) {
  const fd = new FormData();
  fd.append("results_zip", file);
  return fetch(`/api${path}`, { method: "POST", body: fd }).then(async (r) => {
    if (!r.ok) throw new Error(`${r.status}: ${(await r.json().catch(() => ({}))).detail || r.statusText}`);
    return r.json();
  });
}

function Templates({ sid, templates, reload }) {
  const ex = useFetch(() => get("/sim/templates/examples"), []);
  const [text, setText] = useState("");
  const [msg, setMsg] = useState(null);
  const [vf, setVf] = useState({ f_start_ghz: 8, f_stop_ghz: 12, points: 101, theta_deg: "0" });
  const act = useAction();
  return (
    <Card title="HFSS templates" subtitle="A template is a project on the HFSS PC that you built and checked. The framework only changes the mapped variables. A template is usable after its 12-step verification passed on the HFSS PC.">
      <Table columns={[
        { label: "Template", render: (t) => <span className="font-mono">{t.template_id}</span> },
        { label: "Type", key: "simulation_type" },
        { label: "Project (on HFSS PC)", render: (t) => <span className="break-all">{t.aedt_project}</span> },
        { label: "Status", render: (t) => <TemplateBadge status={t.status} /> },
        { label: "Checks", render: (t) => (t.verification?.rows || []).map((r) => r.status === "PASS" ? "✓" : r.status === "SKIP" ? "–" : "✗").join("") || "–" },
        { label: "", render: (t) => (
          <div className="flex flex-wrap gap-2">
            <button className="text-accent underline" onClick={() => act.run(async () => {
              const r = await post(`/sim/templates/${t.template_id}/verification_job`, { sid, ...vf });
              setMsg(<span>Verification package ready: <a className="underline text-accent" href={r.download}>download {r.batch_id}.zip</a>. {r.instruction}</span>);
            })}>Verify template</button>
            <label className="text-accent underline cursor-pointer">Import verification
              <input type="file" accept=".zip" className="hidden" onChange={(e) => e.target.files[0] && act.run(async () => {
                const r = await upload(`/sim/templates/${t.template_id}/import_verification`, e.target.files[0]);
                setMsg(`Template ${r.template_id}: ${r.status}`);
                reload();
              })} />
            </label>
          </div>) },
      ]} rows={templates} />
      <div className="grid gap-2 grid-cols-2 md:grid-cols-4 mt-3">
        <Field label="Verify: start (GHz)"><input className={inputCls} type="number" value={vf.f_start_ghz} onChange={(e) => setVf({ ...vf, f_start_ghz: e.target.value })} /></Field>
        <Field label="stop (GHz)"><input className={inputCls} type="number" value={vf.f_stop_ghz} onChange={(e) => setVf({ ...vf, f_stop_ghz: e.target.value })} /></Field>
        <Field label="points"><input className={inputCls} type="number" value={vf.points} onChange={(e) => setVf({ ...vf, points: e.target.value })} /></Field>
        <Field label="θ (deg, comma list)"><input className={inputCls} value={vf.theta_deg} onChange={(e) => setVf({ ...vf, theta_deg: e.target.value })} /></Field>
      </div>
      <details className="mt-2">
        <summary className="text-sm text-accent cursor-pointer">Register a template</summary>
        <p className="text-xs text-ink2 my-2">Name, project path on the HFSS PC, design, setup, sweep, parameter mapping (schema parameter → AEDT variable + unit), excitation names, angle/polarization variables, boundaries, materials and extraction expressions. Start from an example and edit every name to match your project.</p>
        <div className="flex flex-wrap gap-2 mb-2">
          {Object.entries(ex.data || {}).map(([k, v]) => <Button key={k} kind="secondary" onClick={() => setText(JSON.stringify(v, null, 2))}>{k.replace("example_", "").replace("_template", "")} example</Button>)}
        </div>
        <textarea className={`${inputCls} font-mono text-[11px] h-64`} value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} placeholder="template registration JSON" />
        <div className="mt-2"><Button busy={act.busy} disabled={!text} onClick={() => act.run(async () => {
          const r = await post("/sim/templates", { template_json: text, sid, replace: true });
          setMsg(`Registered ${r.template_id} (${r.status}).`);
          reload();
        })}>Register</Button></div>
      </details>
      {msg && <p className="text-sm text-ink2 mt-3">{msg}</p>}
      <ErrorBox error={act.error || ex.error} />
    </Card>
  );
}

function Setup({ sid, templates, onDone }) {
  const [f, setF] = useState({ template_id: "", mode: "MOCK", transport: "mock", queue_path: "", max_simulations: 20, batch_size: 4, max_iterations: "",
    f_start_ghz: 8, f_stop_ghz: 12, points: 101, polarizations: "TE,TM", theta_deg: "0", outputs: "sparameters", aedt_version: "2024.2" });
  useEffect(() => { if (!f.template_id && templates.length) setF((v) => ({ ...v, template_id: templates[0].template_id })); }, [templates]);
  const act = useAction();
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });
  const t = templates.find((x) => x.template_id === f.template_id);
  const transports = f.mode === "MOCK" ? ["mock", "manual", "shared_folder"] : ["manual", "shared_folder"];
  return (
    <Card title="Study setup" subtitle="Choose the template, the mode and the simulation budget. The budget is never exceeded.">
      <div className="grid gap-x-3 grid-cols-2 md:grid-cols-4">
        <Field label="Template"><select className={inputCls} value={f.template_id} onChange={set("template_id")}>{templates.map((x) => <option key={x.template_id} value={x.template_id}>{x.template_id} ({x.status})</option>)}</select></Field>
        <Field label="Mode" hint={f.mode === "MOCK" ? "software test only, results are not physics" : f.mode === "DRY_RUN" ? "packages only, nothing is simulated" : "real HFSS on the HFSS PC"}>
          <select className={inputCls} value={f.mode} onChange={(e) => setF({ ...f, mode: e.target.value, transport: e.target.value === "MOCK" ? "mock" : "manual" })}>
            <option>DRY_RUN</option><option>MOCK</option><option>HFSS</option>
          </select></Field>
        <Field label="Transfer"><select className={inputCls} value={f.transport} onChange={set("transport")}>{transports.map((x) => <option key={x}>{x}</option>)}</select></Field>
        {f.transport === "shared_folder" && <Field label="Queue folder (seen by both PCs)"><input className={inputCls} value={f.queue_path} onChange={set("queue_path")} placeholder="\\\\server\\simulation_queue" /></Field>}
        <Field label="Max HFSS sims"><input className={inputCls} type="number" min={1} value={f.max_simulations} onChange={set("max_simulations")} /></Field>
        <Field label="Batch size"><input className={inputCls} type="number" min={1} value={f.batch_size} onChange={set("batch_size")} /></Field>
        <Field label="Max iterations" hint="empty = no limit"><input className={inputCls} type="number" min={1} value={f.max_iterations} onChange={set("max_iterations")} /></Field>
        <Field label="AEDT version"><input className={inputCls} value={f.aedt_version} onChange={set("aedt_version")} /></Field>
        <Field label="Start (GHz)"><input className={inputCls} type="number" value={f.f_start_ghz} onChange={set("f_start_ghz")} /></Field>
        <Field label="Stop (GHz)"><input className={inputCls} type="number" value={f.f_stop_ghz} onChange={set("f_stop_ghz")} /></Field>
        <Field label="Points"><input className={inputCls} type="number" value={f.points} onChange={set("points")} /></Field>
        <Field label="Polarizations"><input className={inputCls} value={f.polarizations} onChange={set("polarizations")} /></Field>
        <Field label="θ incidence (deg)"><input className={inputCls} value={f.theta_deg} onChange={set("theta_deg")} /></Field>
        <Field label="Outputs" hint="sparameters, rcs, farfield"><input className={inputCls} value={f.outputs} onChange={set("outputs")} /></Field>
      </div>
      {f.mode === "HFSS" && t && t.status !== "VERIFIED" && <p className="text-sm text-warn">This template is {t.status}. HFSS mode needs a verified template.</p>}
      <Button disabled={!f.template_id} busy={act.busy} onClick={() => act.run(async () => {
        const fields = { ...f };
        if (!fields.max_iterations) delete fields.max_iterations;
        if (fields.transport !== "shared_folder") delete fields.queue_path;
        await post(`/sessions/${sid}/sim/setup`, fields);
        onDone();
      })}>Set up simulation</Button>
      <ErrorBox error={act.error} />
    </Card>
  );
}

function Jobs({ sid, st, reload }) {
  const act = useAction();
  const [log, setLog] = useState(null);
  const shared = st.transport === "shared_folder";
  const call = (path) => act.run(async () => { await post(`/sessions/${sid}/sim/jobs/${path}`); reload(); });
  return (
    <Card title="Jobs" subtitle="Failed jobs are never overwritten and never count as results. Each attempt keeps its own folder.">
      <Table columns={[
        { label: "Job", render: (j) => <span className="font-mono">{j.job_id}</span> },
        { label: "Candidate", key: "candidate_id" },
        { label: "Kind", key: "kind" },
        { label: "State", render: (j) => <span className={`font-semibold ${STATE_CLS[j.state] || ""}`}>{j.state}</span> },
        { label: "Attempts", key: "attempts" },
        { label: "Result", render: (j) => j.state === "COMPLETED" ? <Badge fidelity={j.simulation_type === "MOCK" ? "MOCK_SOLVER" : "VALIDATED_SIMULATION"} /> : j.error ? <span className="text-bad" title={j.error}>{j.error.slice(0, 60)}</span> : "–" },
        { label: "Predicted J", render: (j) => j.prediction ? <span>{fmt(j.prediction.mu)} ± {fmt(j.prediction.sigma)} <Badge fidelity="PREDICTED" /></span> : "–" },
        { label: "", render: (j) => (
          <div className="flex flex-wrap gap-2 whitespace-nowrap">
            {shared && ["PENDING", "CLAIMED", "RUNNING", "SOLVING", "EXTRACTING"].includes(j.state) && <button className="text-accent underline" onClick={() => call(`${j.job_id}/cancel`)}>cancel</button>}
            {["FAILED", "FAILED_VALIDATION", "TIMEOUT", "CANCELLED"].includes(j.state) && j.imported && <button className="text-accent underline" onClick={() => call(`${j.job_id}/retry`)}>retry</button>}
            <a className="text-accent underline" href={`/api/sessions/${sid}/sim/jobs/${j.job_id}/download`}>package</a>
            {j.result_dir && <a className="text-accent underline" href={`/api/sessions/${sid}/sim/jobs/${j.job_id}/download?what=result`}>result</a>}
            <button className="text-accent underline" onClick={() => act.run(async () => setLog({ id: j.job_id, text: await fetch(`/api/sessions/${sid}/sim/jobs/${j.job_id}/logs`).then((r) => r.text()) }))}>logs</button>
            {shared && j.imported && <button className="text-accent underline" onClick={() => call(`${j.job_id}/archive`)}>archive</button>}
          </div>) },
      ]} rows={st.jobs.slice().reverse()} />
      {log && <div className="mt-3"><div className="text-xs text-ink2 mb-1">Log of {log.id} <button className="underline" onClick={() => setLog(null)}>close</button></div><pre className="text-[11px] bg-raised border border-line rounded-md p-2 overflow-auto max-h-80 whitespace-pre-wrap">{log.text}</pre></div>}
      {st.rejected?.length > 0 && <>
        <h4 className="text-xs font-semibold text-ink2 mt-4 mb-1">Candidates rejected before simulation (no budget used)</h4>
        <Table columns={[{ label: "Candidate", key: "candidate_id" }, { label: "Phase", key: "phase" }, { label: "Why", render: (r) => (r.errors || []).join("; ") }]} rows={st.rejected} />
      </>}
      <ErrorBox error={act.error} />
    </Card>
  );
}

function Loop({ sid, st, reload }) {
  const act = useAction();
  const [msg, setMsg] = useState(null);
  const [reason, setReason] = useState("");
  const ob = st.open_batch;
  const manual = st.transport === "manual";
  const bc = st.summary.baseline_check;
  const run = (fn) => act.run(async () => { setMsg(await fn()); reload(); });
  return (
    <Card title="Optimization loop" subtitle={`Phase: ${st.phase}. generate candidates → simulate → collect results → update the optimizer.`}>
      <div className="flex flex-wrap items-end gap-2">
        <Button disabled={!!ob || !!st.summary.stop_reason} busy={act.busy} onClick={() => run(async () => {
          const r = await post(`/sessions/${sid}/sim/next_batch`);
          return `Batch ${r.batch.batch_no} (${r.batch.phase}): ${r.batch.job_ids.length} job(s).`;
        })}>Next batch</Button>
        {ob && manual && ob.transport?.batch_id && <a className="text-sm text-accent underline" href={`/api/sessions/${sid}/sim/batches/${ob.transport.batch_id}/download`}>Download batch {ob.transport.batch_id}.zip</a>}
        {ob && manual && <label className="text-sm rounded-md px-3 py-1.5 font-medium border border-line cursor-pointer hover:bg-raised">Import results (.zip)
          <input type="file" accept=".zip" className="hidden" onChange={(e) => e.target.files[0] && run(async () => {
            const r = await upload(`/sessions/${sid}/sim/import`, e.target.files[0]);
            return `Imported ${r.imported.length} result(s).`;
          })} /></label>}
        {ob && !manual && <Button kind="secondary" busy={act.busy} onClick={() => run(async () => `Collected ${(await post(`/sessions/${sid}/sim/collect`)).collected.length} result(s).`)}>Collect results</Button>}
        {(st.transport === "mock" || st.transport === "shared_folder") && <Button kind="secondary" disabled={st.running || !!st.summary.stop_reason} busy={act.busy} onClick={() => run(async () => { await post(`/sessions/${sid}/sim/run`); return "Running until a stopping rule or the budget ends it."; })}>Run to the end</Button>}
        <a className="text-sm text-accent underline" href="/api/sim/script" target="_blank" rel="noreferrer">Generate PyAEDT script</a>
        {st.phase === "iterate" && <Button kind="secondary" busy={act.busy} onClick={() => run(async () => { const r = await post(`/sessions/${sid}/sim/export`); return `Exported ${Object.keys(r.written).length} files (${r.fidelity}).`; })}>Export final outputs</Button>}
      </div>
      {manual && ob && <p className="text-xs text-ink2 mt-2">On the HFSS PC: unzip, run <span className="font-mono">python hfss_worker/worker.py --input {ob.transport?.batch_id}</span>, then zip <span className="font-mono">{ob.transport?.batch_id}_results</span> and import it here.</p>}
      {st.running && <p className="text-xs text-muted mt-2">Running…</p>}
      {st.summary.stop_reason && <p className="text-sm text-ink2 mt-2">Stopped: {st.summary.stop_reason}</p>}
      {bc && (
        <div className={`mt-3 text-sm border rounded-md p-2 ${bc.passed ? "border-valid/50" : "border-bad/50"}`}>
          <div className="font-medium">{bc.passed ? "Baseline reproduced" : "Baseline NOT reproduced"}</div>
          <p className="text-xs text-ink2">{bc.message}</p>
          {bc.worst && <p className="text-xs">RMSE {fmt(bc.worst.response_rmse_db)} dB · resonance shift {fmt(100 * bc.worst.resonance_shift_rel, 2)}% · peak difference {fmt(bc.worst.peak_difference_db)} dB</p>}
          {!bc.passed && !st.summary.override && st.mode === "HFSS" && <div className="flex flex-wrap gap-2 items-end mt-2">
            <Field label="Override reason (recorded)"><input className={inputCls} value={reason} onChange={(e) => setReason(e.target.value)} /></Field>
            <Button kind="secondary" disabled={!reason.trim()} onClick={() => run(async () => { await post(`/sessions/${sid}/sim/override_baseline`, { reason }); return "Override recorded."; })}>Override and continue</Button>
          </div>}
        </div>
      )}
      {msg && <p className="text-sm text-ink2 mt-2">{msg}</p>}
      <ErrorBox error={act.error || st.error} />
    </Card>
  );
}

function Convergence({ st }) {
  const C = seriesColors();
  const rows = (st.summary.convergence || []).filter((r) => r.J_best !== null && r.J_best !== undefined);
  const er = (st.summary.convergence || []).filter((r) => r.batch_composite_mae !== null && r.batch_composite_mae !== undefined);
  const fs = st.summary.feedback_summary;
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card title="Best simulated design vs simulations used" right={<Badge fidelity={st.mode === "MOCK" ? "MOCK_SOLVER" : "VALIDATED_SIMULATION"} />}>
        {rows.length ? <Plot height={260} data={[{ x: rows.map((r) => r.n_hfss), y: rows.map((r) => r.J_best), mode: "lines+markers", line: { shape: "hv", color: C[0], width: 2 }, marker: { size: 8 } }]}
          layout={{ xaxis: { title: "simulations used (N_HFSS)" }, yaxis: { title: "best composite J" }, showlegend: false }} /> : <Empty>No simulated candidates yet.</Empty>}
      </Card>
      <Card title="Surrogate error vs simulations used" subtitle="|J_simulated − J_PREDICTED|, out of sample (predicted before simulating).">
        {er.length ? <Plot height={260} data={[{ x: er.map((r) => r.n_hfss), y: er.map((r) => r.batch_composite_mae), mode: "lines+markers", line: { color: C[1], width: 2 }, marker: { size: 8 } }]}
          layout={{ xaxis: { title: "simulations used (N_HFSS)" }, yaxis: { title: "composite MAE" }, showlegend: false }} /> : <Empty>Appears after the first proposed batch is simulated.</Empty>}
        {fs && <Table columns={[{ label: "Objective", key: "n" }, { label: "MAE", render: (r) => fmt(r.mae) }, { label: "RMSE", render: (r) => fmt(r.rmse) }, { label: "Mean rel. error", render: (r) => `${fmt(100 * r.rel, 1)}%` }]}
          rows={fs.objective_names.map((n, i) => ({ n, mae: fs.mae[i], rmse: fs.rmse[i], rel: fs.mean_rel_error[i] }))} />}
      </Card>
    </div>
  );
}

export function Simulation({ sid }) {
  const s = useFetch(() => (sid ? get(`/sessions/${sid}/sim/status`) : Promise.resolve(null)), [sid]);
  const st = s.data;
  useEffect(() => {
    if (!st?.configured || !(st.running || (st.open_batch && st.transport !== "manual"))) return;
    const t = setInterval(s.reload, 4000);
    return () => clearInterval(t);
  }, [st?.running, st?.open_batch, st?.transport]);
  const templates = st?.templates || [];
  const b = st?.summary?.budget;
  return (
    <NeedSession session={sid}>
      <div className="grid gap-4">
        <ErrorBox error={s.error} />
        {st?.configured && (
          <div className="grid gap-4 grid-cols-2 lg:grid-cols-5">
            <Stat label="Backend" value={<span className="text-lg">{st.mode}</span>} sub={`${st.transport}${st.mode === "MOCK" ? " · not physics" : ""}`} />
            <Stat label="Template" value={<span className="text-sm font-mono break-all">{st.template.template_id}</span>} sub={<TemplateBadge status={st.template.status} />} />
            <Stat label="Versions" value={<span className="text-sm">{st.versions?.aedt_version ? `AEDT ${st.versions.aedt_version}` : "–"}</span>} sub={st.versions?.pyaedt_version ? `PyAEDT ${st.versions.pyaedt_version}` : "from the first HFSS result"} />
            <Stat label="Budget" value={`${b.used} / ${b.max_simulations}`} sub={`${b.remaining} left · batch ${b.batch_size}`} />
            <Stat label="Queue" value={st.queue ? `${st.queue.counts.pending} pending` : `${b.completed} done`} sub={st.queue ? `${st.queue.counts.running} running · ${st.queue.counts.completed} done · ${st.queue.counts.failed} failed · ${st.queue.workers.length} worker(s)` : `${b.failed} failed`} />
          </div>
        )}
        {st && <Templates sid={sid} templates={templates} reload={s.reload} />}
        {st && !st.configured && <Setup sid={sid} templates={templates} onDone={s.reload} />}
        {st?.configured && <Loop sid={sid} st={st} reload={s.reload} />}
        {st?.configured && <Jobs sid={sid} st={st} reload={s.reload} />}
        {st?.configured && <Convergence st={st} />}
        {st?.configured && st.queue && st.queue.workers.length > 0 && <Card title="Workers"><Json value={st.queue.workers} /></Card>}
      </div>
    </NeedSession>
  );
}
