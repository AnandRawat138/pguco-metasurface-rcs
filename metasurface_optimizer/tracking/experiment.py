"""Experiment records (Sec. 31) and a pluggable experiment repository (SQLite default)."""
from __future__ import annotations

import datetime as _dt
import json
import platform
import sqlite3
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

FRAMEWORK_VERSION = "0.1.0"


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if hasattr(o, "value"):
        return o.value
    if hasattr(o, "__dict__"):
        return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
    return str(o)


def dumps(obj: Any, **kw) -> str:
    return json.dumps(obj, default=_jsonable, **kw)


@dataclass
class ExperimentRecord:
    """Everything needed to reproduce and audit one run."""

    name: str
    random_seed: int
    parameter_schema: Dict[str, Any]
    objective_definition: List[Dict[str, Any]]
    normalization: Dict[str, Any]
    optimizer: Dict[str, Any]
    acquisition_function: Dict[str, Any]
    physics_constraints: Dict[str, Any]
    training_configuration: Dict[str, Any]
    candidate_pool_size: Optional[int] = None
    number_of_HFSS_evaluations: int = 0
    high_fidelity_source: Dict[str, Any] = field(default_factory=dict)
    dataset_version: str = "none"
    model_version: str = FRAMEWORK_VERSION
    stopping_rule: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    experiment_id: str = field(default_factory=lambda: _dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
    timestamp: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat())
    environment: Dict[str, str] = field(default_factory=lambda: {"python": platform.python_version(), "platform": platform.platform()})
    notes: str = ""

    def __post_init__(self) -> None:
        try:
            import numpy, scipy, sklearn, torch, xgboost

            self.environment.update({"numpy": numpy.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                                     "torch": torch.__version__, "xgboost": xgboost.__version__})
        except Exception:  # pragma: no cover
            pass

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(dumps(asdict(self)))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentRecord":
        d = json.loads(Path(path).read_text())
        return cls(**d)


class ExperimentRepository(ABC):
    """Storage interface so that SQLite can be replaced (e.g. PostgreSQL) without touching callers."""

    @abstractmethod
    def add(self, rec: ExperimentRecord, artifacts_dir: Optional[str] = None) -> None: ...

    @abstractmethod
    def list(self, limit: int = 100) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def get(self, experiment_id: str) -> Optional[Dict[str, Any]]: ...


class SQLiteRepository(ExperimentRepository):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY, name TEXT, timestamp TEXT, seed INTEGER, optimizer TEXT,
                n_hf INTEGER, best_composite REAL, artifacts_dir TEXT, record_json TEXT)""")

    def _conn(self):
        return sqlite3.connect(self.path)

    def add(self, rec: ExperimentRecord, artifacts_dir: Optional[str] = None) -> None:
        best = rec.results.get("best_validated", {}).get("composite") if isinstance(rec.results, dict) else None
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?,?,?,?,?)",
                      (rec.experiment_id, rec.name, rec.timestamp, rec.random_seed, rec.optimizer.get("name", ""),
                       rec.number_of_HFSS_evaluations, best, artifacts_dir, dumps(asdict(rec))))

    def list(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT experiment_id,name,timestamp,seed,optimizer,n_hf,best_composite,artifacts_dir FROM experiments "
                             "ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        keys = ["experiment_id", "name", "timestamp", "seed", "optimizer", "n_hf", "best_composite", "artifacts_dir"]
        return [dict(zip(keys, r)) for r in rows]

    def get(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT record_json FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
        return json.loads(r[0]) if r else None
