"""Deviation analysis: scan mesh vs reconstructed solid (bounded memory, point-to-plane)."""
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def distance(model, pts, n=400000, seed=0):
    """Signed distance of pts to the model surface: + = outside the model (model lacks material)."""
    q, fi = trimesh.sample.sample_surface(model, n, seed=seed)
    nq = model.face_normals[fi]
    _, j = cKDTree(q).query(pts, k=1)
    return np.einsum('ij,ij->i', pts - q[j], nq[j])


def deviation(scan, model, n=40000, seed=0):
    pts, _ = trimesh.sample.sample_surface_even(scan, n, seed=seed)
    return pts, distance(model, pts)


def stats(d, tol):
    a = np.abs(d)
    return dict(rms=float(np.sqrt((d ** 2).mean())), p95=float(np.percentile(a, 95)),
                max=float(a.max()), within=float((a <= tol).mean()))


def tiers(m, feats, pts, sigma, zplanes=()):
    """Precision tier per point: 0=A on a precise primitive (hole, turned cylinder/face, datum plane),
    2=C rough (cast) surface, 1=B otherwise."""
    lab = np.ones(len(pts), int)
    _, nb = cKDTree(m.vertices).query(pts, 16)
    nrm = m.vertex_normals[nb[:, 0]]
    nz = np.abs(nrm[:, 2]) > 0.95  # faces normal to the main axis
    rad = np.abs(nrm[:, 0] * pts[:, 0] + nrm[:, 1] * pts[:, 1]) / (np.hypot(pts[:, 0], pts[:, 1]) + 1e-9) > 0.95
    P = m.vertices[nb] - m.vertices[nb].mean(1, keepdims=True)
    rough = np.sqrt(np.maximum(np.linalg.eigvalsh(np.einsum('nki,nkj->nij', P, P) / 16)[:, 0], 0))
    lab[rough > max(4 * sigma, 0.04)] = 2
    band = max(4 * sigma, 0.08)
    A = np.zeros(len(pts), bool)
    for f in feats:
        if f['op'] in ('hole', 'holes'):
            from .residual import hole_points
            a = np.asarray(f['ax'], float)
            for p in hole_points(f):
                d = pts - p
                t = d @ a
                rr = np.linalg.norm(d - np.outer(t, a), axis=1)
                A |= (t > -band) & (t < f['h'] + band) & (np.abs(rr - f['d'] / 2) < band + 0.03 * f['d'])
        elif f['op'] == 'revolve':
            rz = np.c_[np.hypot(pts[:, 0], pts[:, 1]), pts[:, 2]]
            for e in f['profile']:
                if e[0] != 'L':
                    continue
                a, b = np.asarray(e[1]), np.asarray(e[2])
                if min(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1e-6:
                    continue  # only cylinders and faces normal to the axis
                u = b - a
                L = np.linalg.norm(u)
                t = np.clip((rz - a) @ u / max(L * L, 1e-12), 0, 1)
                ok = nz if abs(u[1]) < 1e-9 else rad
                A |= ok & (np.linalg.norm(rz - (a + np.outer(t, u)), axis=1) < band)
    for z, _, _ in zplanes:
        A |= nz & (np.abs(pts[:, 2] - z) < band)
    lab[A] = 0
    return lab


def tier_stats(d, lab, tol):
    out = {}
    for i, k in enumerate('ABC'):
        s = lab == i
        if s.sum() > 20:
            out[k] = dict(stats(d[s], tol[k]), frac=float(s.mean()))
    return out
