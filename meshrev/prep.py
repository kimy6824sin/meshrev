"""Mesh loading, cleaning, noise estimate and canonical frame detection."""
import numpy as np
import trimesh
from scipy.spatial import cKDTree

rng = np.random.default_rng(0)


def load(path, min_frac=0.01):
    """Load mesh, drop scan fragments, repair small holes."""
    m = trimesh.load(path, force='mesh')
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    cc = trimesh.graph.connected_components(m.face_adjacency, min_len=1)
    keep = [c for c in cc if len(c) >= min_frac * len(m.faces)]
    if len(keep) < len(cc):
        m = m.submesh(keep, append=True)
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
    trimesh.repair.fix_normals(m)
    return m


def smooth_normals(m, it=2):
    """Face normals from Laplacian-smoothed vertex normals (noise suppression)."""
    vn = m.vertex_normals.copy()
    A = m.vertex_adjacency_graph
    import networkx as nx
    L = nx.to_scipy_sparse_array(A, nodelist=range(len(m.vertices)), format='csr')
    for _ in range(it):
        vn = vn + L @ vn
        vn /= np.linalg.norm(vn, axis=1)[:, None] + 1e-12
    fn = vn[m.faces].sum(1)
    return fn / (np.linalg.norm(fn, axis=1)[:, None] + 1e-12)


def noise(m, k=16, n=5000):
    """Robust scan noise sigma: median smallest PCA std over local vertex neighborhoods."""
    v = m.vertices
    idx = rng.choice(len(v), min(n, len(v)), replace=False)
    _, nb = cKDTree(v).query(v[idx], k)
    P = v[nb] - v[nb].mean(1, keepdims=True)
    ev = np.linalg.eigvalsh(np.einsum('nki,nkj->nij', P, P) / k)
    return float(np.median(np.sqrt(np.maximum(ev[:, 0], 0))))


def dir_clusters(D, w, ang=3.0, minfrac=0.02, k=10):
    """Greedy antipodal direction clustering on the unit sphere -> [(dir, weight_frac)]."""
    D = D / (np.linalg.norm(D, axis=1)[:, None] + 1e-12)
    tot, c, out = w.sum(), np.cos(np.radians(ang)), []
    alive = np.ones(len(D), bool)
    for _ in range(k):
        idx = np.flatnonzero(alive)
        if len(idx) < 3:
            break
        sub = rng.choice(idx, min(30000, len(idx)), replace=False)
        cand = rng.choice(idx, min(200, len(idx)), p=w[idx] / w[idx].sum())
        s = (w[sub, None] * (np.abs(D[sub] @ D[cand].T) > c)).sum(0)
        a = D[cand[s.argmax()]]
        for _ in range(3):
            d = D[idx] @ a
            mk = np.abs(d) > c
            a = (D[idx][mk] * (np.sign(d[mk]) * w[idx][mk])[:, None]).sum(0)
            a /= np.linalg.norm(a)
        mk = alive & (np.abs(D @ a) > c)
        if w[mk].sum() < minfrac * tot:
            break
        out.append((a, w[mk].sum() / tot))
        alive &= np.abs(D @ a) < np.cos(np.radians(2 * ang))
    return out


def cyl_axes(m, fn, k=24, n=20000):
    """Candidate cylinder/cone axis directions from local normal PCA (normals of a cylinder span a great circle)."""
    c = m.triangles_center
    idx = rng.choice(len(c), min(n, len(c)), replace=False)
    _, nb = cKDTree(c).query(c[idx], k)
    N = fn[nb]
    ev, evec = np.linalg.eigh(np.einsum('nki,nkj->nij', N, N) / k)
    # second moment of unit normals: plane -> (0,0,1); cylinder -> (0,a,1-a); noise -> all >0
    ok = (ev[:, 1] > 0.01) & (ev[:, 0] < 0.15 * ev[:, 1])
    return evec[ok, :, 0], m.area_faces[idx][ok] * len(c) / len(idx)


def frame(m, fn=None):
    """Canonical frame: Z = dominant plane normal / cylinder axis, X = next orthogonal direction.
    Returns 4x4 transform world->canonical and info."""
    fn = smooth_normals(m) if fn is None else fn
    w = m.area_faces
    planes = dir_clusters(fn, w, 3, 0.02)
    ad, aw = cyl_axes(m, fn)
    axes = dir_clusters(ad, aw, 3, 0.03) if len(ad) else []
    cand = [(a, f, 'plane') for a, f in planes] + [(a, f * aw.sum() / w.sum(), 'axis') for a, f in axes]

    def score(a):  # plane area normal to a + cylinder area around a
        return sum(f for b, f, _ in cand if abs(a @ b) > 0.998)
    Z = max((c[0] for c in cand), key=score)
    perp = [c for c in cand if abs(c[0] @ Z) < 0.087]
    if perp:
        X = max(perp, key=lambda c: score(c[0]))[0]
    else:  # principal in-plane direction of the vertex cloud
        v = m.vertices - m.vertices.mean(0)
        v = v - np.outer(v @ Z, Z)
        X = np.linalg.eigh(v.T @ v)[1][:, -1]
    X = X - (X @ Z) * Z
    X /= np.linalg.norm(X)
    # heavy end down: volume centroid below bbox middle
    R = np.array([X, np.cross(Z, X), Z])
    p = m.vertices @ R.T
    cz = (m.center_mass if m.is_watertight else m.centroid) @ Z
    if cz > (p[:, 2].min() + p[:, 2].max()) / 2:
        R = np.array([X, -R[1], -Z])
    T = np.eye(4)
    T[:3, :3] = R
    p = m.vertices @ R.T
    lo, hi = np.percentile(p, 0.02, axis=0), np.percentile(p, 99.98, axis=0)  # robust to scan spikes
    T[:3, 3] = -np.array([*(lo[:2] + hi[:2]) / 2, lo[2]])
    return T, dict(planes=planes, axes=axes)


def prepare(path):
    m = load(path)
    fn = smooth_normals(m)
    T, info = frame(m, fn)
    m.apply_transform(T)
    fn = fn @ T[:3, :3].T
    info.update(T=T, sigma=noise(m), n_faces=len(m.faces), extents=m.extents.tolist())
    return m, fn, info
