"""FSSDataBase mask prior on tiny synthetic data (no real FSSDataBase files are committed)."""
import copy
import os

import numpy as np
import pytest

from test_fssdatabase import fss_root  # noqa: F401  (fixture: tiny on-disk FSSDataBase layout)
from metasurface_optimizer.models.mask_prior import (MaskPriorFeatureMap, MaskPriorModel, build_fssdb_cache, evaluate_split, load_cache,
                                                     scalar_inputs, spectrum_errors)


def _toy(n=24, grid=50, seed=0):
    """Square patches of random size; |S11| grows with fill factor, S21 is its passive complement."""
    rng = np.random.default_rng(seed)
    f = np.linspace(10, 20, 101)
    masks = np.zeros((n, 3, grid, grid), np.uint8)
    S = np.zeros((n, 101, 4), np.complex64)
    for i in range(n):
        a = rng.integers(5, grid // 2)
        masks[i, 0, grid // 2 - a:grid // 2 + a, grid // 2 - a:grid // 2 + a] = 1
        r = 0.9 * masks[i, 0].mean() * np.exp(1j * np.radians(10 * f))
        t = np.sqrt(1 - np.abs(r) ** 2) * np.exp(1j * np.radians(10 * f + 90))
        S[i] = np.stack([r, r, t, t], -1)
    sc = np.column_stack([rng.choice([1.0, 2.0], n), np.ones(n), np.full(n, 0.5)]).astype(np.float32)
    return masks, sc, S


def test_fit_predict_save_load(tmp_path):
    masks, sc, S = _toy()
    m = MaskPriorModel(width=8, emb=16).fit(masks[:20], sc[:20], S[:20], (masks[20:], sc[20:], S[20:]), epochs=3, threads=1)
    p = m.predict(masks, sc)
    assert p.shape == (24, 101, 4) and np.iscomplexobj(p) and np.isfinite(p).all()
    assert len(m.history) == 3 and all("val_loss" in h for h in m.history)
    m.save(tmp_path / "m.pt")
    m2 = MaskPriorModel.load(tmp_path / "m.pt")
    np.testing.assert_allclose(m2.predict(masks, sc), p, rtol=1e-5, atol=1e-6)
    assert m2.embed(masks[:2], sc[:2]).shape == (2, 16)


def test_passivity_term_penalises_gain():
    import torch

    m = MaskPriorModel(lambda_passive=1.0)
    y = torch.zeros(2, 101, 4, 2)
    loud = torch.zeros(2, 101, 4, 2)
    loud[..., 0] = 1.0  # |S11| = |S21| = 1 -> power 2 > 1
    assert float(m._loss(loud, y)) > float(MaskPriorModel(lambda_passive=0.0)._loss(loud, y))


def test_spectrum_errors_exact_and_wrapped_phase():
    S = np.exp(1j * np.radians(np.linspace(-179, 179, 101)))[None, :, None].repeat(4, 2) * 0.5
    e = spectrum_errors(S, S)
    assert all(v["mag_db_mae"] == 0 and v["phase_deg_mae"] == pytest.approx(0, abs=1e-6) for v in e.values())
    e = spectrum_errors(S * np.exp(1j * np.radians(10)), S)  # 10 deg everywhere, including across the +-180 wrap
    assert e["S11_TE"]["phase_deg_mae"] == pytest.approx(10, abs=1e-4)


def test_cache_split_is_disjoint_and_complete(fss_root, tmp_path):
    root, ids = fss_root
    d = build_fssdb_cache(root, tmp_path / "c.npz", seed=3, fractions=(0.5, 0.25, 0.25))
    d2 = load_cache(tmp_path / "c.npz")
    assert list(d2["ids"]) == list(d["ids"]) and d2["masks"].shape[1:] == (3, 50, 50)
    groups = [set(d2["ids"][d2["split"] == k]) for k in ("train", "val", "test")]
    assert sum(map(len, groups)) == len(d2["ids"]) and not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert scalar_inputs(d2).shape == (len(d2["ids"]), 3)


def test_evaluate_split_reports_model_and_baselines():
    masks, sc, S = _toy(30)
    data = {"masks": masks, "S": S, "h_total_mm": sc[:, 0], "n_layers": np.ones(30, np.uint8), "wire_width_mm": sc[:, 2],
            "split": np.array(["train"] * 20 + ["val"] * 5 + ["test"] * 5)}
    m = MaskPriorModel(width=8, emb=16).fit(masks[:20], sc[:20], S[:20], epochs=2, threads=1)
    r = evaluate_split(m, data, "test")
    assert r["n"] == 5 and r["n_train"] == 20
    assert set(r) >= {"model", "baseline_train_mean", "baseline_nn_fill_factor"} and set(r["model"]) == {"S11_TE", "S11_TM", "S21_TE", "S21_TM"}


@pytest.fixture
def demo_builder():
    from metasurface_optimizer.experiment import ExperimentBuilder, load_config

    root = os.path.join(os.path.dirname(__file__), "..")
    return ExperimentBuilder(load_config(os.path.join(root, "configs/experiments/demo_checkerboard.yaml")))


def test_feature_map_and_experiment_wiring(tmp_path, demo_builder):
    from metasurface_optimizer.experiment import ExperimentBuilder

    masks, sc, S = _toy()
    MaskPriorModel(width=8, emb=16).fit(masks, sc, S, epochs=1, threads=1).save(tmp_path / "m.pt")
    cfg = copy.deepcopy(demo_builder.cfg)
    cfg["transfer"] = {"enabled": True, "prior": "fssdatabase_mask_prior", "model_path": str(tmp_path / "m.pt"), "n_freq": 5}
    b = ExperimentBuilder(cfg, demo_builder.root)
    fm = b.prior_feature_fn()
    assert isinstance(fm, MaskPriorFeatureMap)
    X = np.vstack([b.x0()] * 2)
    X[1, b.schema.index("length_a")] = 3.0
    F = fm(X)
    assert F.shape == (2, 2 * 5 + 2) and np.isfinite(F).all()
    assert 0.0 <= fm.out_of_range_fraction <= 1.0
    m, k = fm.cell_masks(X, fm.cells[0])
    assert m.shape == (2, 3, 50, 50) and m[:, 1:].sum() == 0
    assert m[0, 0].mean() == pytest.approx((8.5 / 10) ** 2, abs=0.05) and m[1, 0].mean() < m[0, 0].mean()
    cfg["transfer"]["features"] = "embedding"
    assert ExperimentBuilder(cfg, demo_builder.root).prior_feature_fn()(X).shape == (2, 8)
    cfg["transfer"] = {"enabled": True, "prior": "nope"}
    with pytest.raises(ValueError):
        ExperimentBuilder(cfg, demo_builder.root).prior_feature_fn()
    assert ExperimentBuilder(demo_builder.cfg, demo_builder.root).prior_feature_fn(True) is not None  # synthetic path still works


@pytest.mark.skipif("FSSDATABASE_ROOT" not in os.environ, reason="set FSSDATABASE_ROOT to an FSSDataBase copy")
def test_real_data_small_fit(tmp_path):
    d = build_fssdb_cache(os.environ["FSSDATABASE_ROOT"], tmp_path / "c.npz", max_samples=120)
    tr = d["split"] == "train"
    m = MaskPriorModel(width=8, emb=16).fit(d["masks"][tr], scalar_inputs(d)[tr], d["S"][tr], epochs=2, threads=2)
    r = evaluate_split(m, d, "test")
    assert r["n"] > 0 and all(np.isfinite(v["mag_db_mae"]) for v in r["model"].values())
