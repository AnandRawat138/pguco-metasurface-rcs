import React, { useCallback, useEffect, useState } from "react";
import { get } from "./api.js";
import { Dashboard, Dataset, ExperimentLog } from "./tabs/Study.jsx";
import { Geometry, EMResponse, BaselineAnalysis } from "./tabs/Design.jsx";
import { Optimization, Candidates, Results, Export } from "./tabs/Optimize.jsx";
import { PaperEvaluation, ModelComparison, Ablation } from "./tabs/Evaluate.jsx";
import { Simulation } from "./tabs/Simulation.jsx";

const TABS = [
  ["Dashboard", Dashboard],
  ["Dataset", Dataset],
  ["Geometry", Geometry],
  ["EM Response", EMResponse],
  ["Baseline Analysis", BaselineAnalysis],
  ["Optimization", Optimization],
  ["Candidate Designs", Candidates],
  ["High-Fidelity Simulation", Simulation],
  ["Existing Paper", PaperEvaluation],
  ["Model Comparison", ModelComparison],
  ["Ablation Study", Ablation],
  ["Results", Results],
  ["Export", Export],
  ["Experiment Log", ExperimentLog],
];

const load = (k, d) => {
  try {
    return localStorage.getItem(k) || d;
  } catch (_) {
    return d;
  }
};
const save = (k, v) => {
  try {
    localStorage.setItem(k, v);
  } catch (_) {
    /* storage unavailable */
  }
};

export default function App() {
  const [tab, setTab] = useState(load("msopt.tab", "Dashboard"));
  const [sid, setSid] = useState(load("msopt.sid", ""));
  const [st, setSt] = useState(null);
  const [err, setErr] = useState(null);

  const refresh = useCallback(async () => {
    if (!sid) return setSt(null);
    try {
      setSt(await get(`/sessions/${sid}/state`));
      setErr(null);
    } catch (e) {
      setErr(e.message);
      setSt(null);
    }
  }, [sid]);

  useEffect(() => {
    refresh();
  }, [refresh]);
  // Poll while an automatic run is in progress.
  useEffect(() => {
    if (st?.status?.state !== "running") return;
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, [st?.status?.state, refresh]);

  const choose = (t) => {
    setTab(t);
    save("msopt.tab", t);
  };
  const open = (id) => {
    setSid(id);
    save("msopt.sid", id);
  };

  const Cur = (TABS.find(([n]) => n === tab) || TABS[0])[1];
  return (
    <div className="min-h-screen">
      <header className="border-b border-line bg-surface">
        <div className="max-w-[1400px] mx-auto px-4 py-3 flex flex-wrap items-center gap-x-4 gap-y-1">
          <h1 className="text-base font-semibold">Counterfactual metasurface optimizer</h1>
          <span className="text-xs text-muted">physics-guided, simulation-efficient</span>
          <span className="ml-auto text-xs text-ink2 truncate">
            {sid ? <>study <span className="font-mono">{sid}</span>{st?.name ? ` · ${st.name}` : ""}{st?.status?.state ? ` · ${st.status.state}` : ""}</> : "no study open"}
          </span>
        </div>
        <nav className="max-w-[1400px] mx-auto px-4 flex gap-1 overflow-x-auto">
          {TABS.map(([n]) => (
            <button key={n} onClick={() => choose(n)}
              className={`text-sm whitespace-nowrap px-3 py-2 border-b-2 ${tab === n ? "border-accent text-ink font-medium" : "border-transparent text-ink2 hover:text-ink"}`}>
              {n}
            </button>
          ))}
        </nav>
      </header>
      <main className="max-w-[1400px] mx-auto px-4 py-4">
        {err && sid && <p className="text-sm text-bad mb-3">Could not load study {sid}: {err}</p>}
        <Cur sid={sid} st={st} refresh={refresh} open={open} go={choose} />
      </main>
    </div>
  );
}
