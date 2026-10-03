"""FSSDataBase adapter on a small fixture written in the dataset's documented on-disk format.

The fixture is generated here (no real FSSDataBase data is committed); its layout follows the
README and a partial local copy: headerless 10-column raw_result.csv with 0 and 30 deg blocks,
500x500 mask_<n>.csv per layer, data.json, and a SQLite DataBase.db with TEXT ids."""
import json
import sqlite3

import numpy as np
import pytest

from metasurface_optimizer.data.public_datasets.fssdatabase import FSSDataBaseAdapter


def _write_sample(root, sid, n_layers, nan_30=False):
    d = root / "datasets" / sid
    d.mkdir(parents=True)
    f = np.round(np.linspace(10, 20, 101), 1)
    rows = []
    for ang in (0.0, 30.0):
        for k, fk in enumerate(f):
            db = [-4.5 - 0.1 * k, -4.7, -1.9, -1.8]
            ph = [22.6 + k, 24.9, 92.8, 93.5]
            if nan_30 and ang == 30.0 and k > 30:
                db = [np.nan] * 4; ph = [np.nan] * 4
            rows.append([ang, fk] + db + ph)
    with open(d / "raw_result.csv", "w") as fh:
        for r in rows:
            fh.write(",".join("nan" if np.isnan(v) else repr(float(v)) for v in r) + "\n")
    for n in range(n_layers):
        m = np.zeros((500, 500), int); m[100:400, 200 + 10 * n:300] = 1
        np.savetxt(d / f"mask_{n}.csv", m, fmt="%d", delimiter=",")
    (d / "data.json").write_text('{"idx": %s, "unit": [0.4255, 5], "sub": {"sub1": {"material": "F4BM348", "h": 1.0}}}' % sid)


@pytest.fixture
def fss_root(tmp_path):
    ids = ["1786866648.2374983", "1786870719.0583885", "1787441664.8019676"]
    _write_sample(tmp_path, ids[0], 2)
    _write_sample(tmp_path, ids[1], 1, nan_30=True)
    (tmp_path / "datasets" / ids[2]).mkdir(parents=True)  # interrupted download: empty folder
    (tmp_path / "database").mkdir()
    con = sqlite3.connect(tmp_path / "database" / "DataBase.db")
    con.execute("CREATE TABLE samples(id TEXT, sample_path TEXT, size REAL, height REAL)")
    cols = ",".join(f"p{i:03d} REAL" for i in range(300))
    con.execute(f"CREATE TABLE responses(id TEXT, direct INTEGER, angle REAL, {cols})")
    grid = np.linspace(1, 30, 300)
    band = (grid >= 10 - 1e-9) & (grid <= 20 + 1e-9)
    for sid in ids:
        con.execute("INSERT INTO samples VALUES (?,?,?,?)", (sid, f"datasets/{sid}", 5.0, 1.0))
        for direct in (0, 1):
            for ang in (0.0, 29.999999999999996):
                con.execute(f"INSERT INTO responses VALUES (?,?,?,{','.join('?' * 300)})",
                            (sid, direct, ang, *np.where(band, np.where(grid < 15, 1.0, -1.0), 0.0)))
    con.commit(); con.close()
    return tmp_path, ids


def test_load_zero_degrees(fss_root):
    root, ids = fss_root
    ad = FSSDataBaseAdapter(root, angle=0.0, mask_size=100)
    s = ad.load()
    assert s.masks.shape == (2, 3, 100, 100) and s.responses.shape == (2, 101, 4)
    assert s.frequency[0] == pytest.approx(10e9) and s.frequency[-1] == pytest.approx(20e9)
    np.testing.assert_allclose(20 * np.log10(np.abs(s.responses[0, 0, 0])), -4.5, atol=1e-9)
    np.testing.assert_allclose(np.degrees(np.angle(s.responses[0, 0, 2])), 92.8, atol=1e-9)
    assert [m["id"] for m in s.meta] == ids[:2]  # ids stay exact strings
    assert s.meta[0]["n_layers"] == 2 and s.masks[0, 2].sum() == 0 and s.meta[0]["period_mm"] == 10.0
    assert s.meta[0]["idx"] == ids[0]
    assert len(ad.failures) == 1 and ad.failures[0]["id"] == ids[2]  # empty folder reported, not fatal


def test_nan_policy_at_30_degrees(fss_root):
    root, ids = fss_root
    ad = FSSDataBaseAdapter(root, angle=30.0)
    s = ad.load()
    assert ad.skipped_nan == [ids[1]] and len(s.meta) == 1
    s2 = FSSDataBaseAdapter(root, angle=30.0, nan_policy="keep").load()
    assert np.isnan(s2.responses).any()


def test_database_labels_cover_undownloaded_samples(fss_root):
    root, ids = fss_root
    lab = FSSDataBaseAdapter(root).database_labels(direct=1, angle=30.0)
    assert lab["ids"] == sorted(ids)  # includes the sample whose folder is empty
    assert lab["frequency"].min() >= 10e9 - 1 and lab["frequency"].max() <= 20e9 + 1
    assert set(np.unique(lab["labels"])) <= {-1.0, 1.0}


@pytest.mark.skipif("FSSDATABASE_ROOT" not in __import__("os").environ, reason="set FSSDATABASE_ROOT to a (partial) FSSDataBase copy")
def test_real_copy():
    """Runs against real FSSDataBase files when available (not committed). Checks the label table agrees with raw data."""
    import os

    root = os.environ["FSSDATABASE_ROOT"]
    ad = FSSDataBaseAdapter(root, angle=0.0, max_samples=50)
    s = ad.load()
    assert s.responses.shape[1:] == (101, 4) and s.masks.shape[1] == 3
    lab = ad.database_labels(direct=1, angle=0.0)
    row = {i: k for k, i in enumerate(lab["ids"])}
    agree = []
    for i, m in enumerate(s.meta):
        idx = [int(np.argmin(np.abs(s.frequency - f))) for f in lab["frequency"]]
        raw = np.where(20 * np.log10(np.abs(s.responses[i, idx, 2])) > -5, 1, -1)
        agree.append(np.mean(raw == lab["labels"][row[m["id"]]]))
    assert np.mean(agree) > 0.95  # residual disagreement comes from the DB's coarser 1-30 GHz grid near the -5 dB edge
