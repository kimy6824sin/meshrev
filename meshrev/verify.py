"""Deviation analysis: scan mesh vs reconstructed solid."""
import numpy as np
import trimesh


def deviation(scan, model, n=40000, seed=0):
    """Signed distance of scan surface samples to the model (+ outside model). Returns (points, d)."""
    pts, _ = trimesh.sample.sample_surface_even(scan, n, seed=seed) if len(scan.faces) else (np.zeros((0, 3)), None)
    q = trimesh.proximity.ProximityQuery(model)
    d = -q.signed_distance(pts)  # trimesh: positive inside -> flip so + = material missing in model
    return pts, d


def stats(d, tol):
    a = np.abs(d)
    return dict(rms=float(np.sqrt((d ** 2).mean())), p95=float(np.percentile(a, 95)),
                max=float(a.max()), within=float((a <= tol).mean()))
