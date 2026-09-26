"""Planar sections, voxel/occupancy grids and 2D contour -> sketch entity (line/arc/circle) fitting."""
import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import contourpy


def sections(m, zs):
    """XY cross-sections of a canonical-frame mesh at heights zs -> list of shapely geometries."""
    out = []
    for p in m.section_multiplane([0, 0, 0], [0, 0, 1], list(zs)):
        if p is None or not len(p.polygons_full):
            out.append(Polygon())
            continue
        T = p.metadata.get('to_3D', np.eye(4)) if hasattr(p, 'metadata') else np.eye(4)
        g = unary_union([q.buffer(0) for q in p.polygons_full])
        out.append(affinity.affine_transform(g, [T[0, 0], T[0, 1], T[1, 0], T[1, 1], T[0, 3], T[1, 3]]))
    return out


def grid_xy(bounds, h):
    xs = np.arange(bounds[0][0] + h / 2, bounds[1][0], h)
    ys = np.arange(bounds[0][1] + h / 2, bounds[1][1], h)
    return xs, ys


def voxels(polys, xs, ys):
    """Occupancy (nz, ny, nx) from per-slice polygons."""
    X, Y = np.meshgrid(xs, ys)
    return np.stack([shapely.contains_xy(g, X, Y) if not g.is_empty else np.zeros(X.shape, bool) for g in polys])


def rz_occupancy(polys, rs, nth=360):
    """Angular occupancy fraction (nz, nr) about the Z axis."""
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    X, Y = np.outer(rs, np.cos(th)), np.outer(rs, np.sin(th))
    return np.stack([shapely.contains_xy(g, X, Y).mean(1) if not g.is_empty else np.zeros(len(rs)) for g in polys])


def contours(img, level, x0, y0, h, min_area=0.0):
    """Iso-contours of a 2D grid (rows=y) as shapely polygons (with holes)."""
    pad = np.pad(img.astype(float), 1)
    gen = contourpy.contour_generator(z=pad, name='serial', fill_type=contourpy.FillType.OuterOffset)
    polys = []
    for pts, offs in zip(*gen.filled(level, 1e9)):
        rings = [pts[a:b] for a, b in zip(offs[:-1], offs[1:])]
        rings = [np.c_[x0 + (r[:, 0] - 1) * h, y0 + (r[:, 1] - 1) * h] for r in rings]
        p = Polygon(rings[0], rings[1:]).buffer(0)
        if p.area > min_area:
            polys.append(p)
    return polys


# ---------------- 2D entity fitting ----------------

def resample(P, ds):
    """Uniformly resample a closed ring (drop duplicate end point)."""
    P = np.asarray(P)[:, :2]
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    Q = np.vstack([P, P[:1]])
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    n = max(int(L[-1] / ds), 8)
    t = np.linspace(0, L[-1], n, endpoint=False)
    return np.c_[np.interp(t, L, Q[:, 0]), np.interp(t, L, Q[:, 1])]


def circle3(a, b, c):
    """Circle through 3 points -> (center, r) or None if collinear."""
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-12:
        return None
    s = [p @ p for p in (a, b, c)]
    ux = (s[0] * (b[1] - c[1]) + s[1] * (c[1] - a[1]) + s[2] * (a[1] - b[1])) / d
    uy = (s[0] * (c[0] - b[0]) + s[1] * (a[0] - c[0]) + s[2] * (b[0] - a[0])) / d
    cen = np.array([ux, uy])
    return cen, np.linalg.norm(a - cen)


def fit_circle(P, w=None, it=5):
    """Robust algebraic circle fit (Kasa + Tukey IRLS) -> center, r, rms."""
    w = np.ones(len(P)) if w is None else w
    for _ in range(it):
        A = np.c_[2 * P, np.ones(len(P))]
        b = (P ** 2).sum(1)
        sol = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)[0]
        c = sol[:2]
        r = np.sqrt(max(sol[2] + c @ c, 1e-12))
        res = np.linalg.norm(P - c, axis=1) - r
        s = 4.685 * max(1.4826 * np.median(np.abs(res)), 1e-4)
        w = np.clip(1 - (res / s) ** 2, 0, 1) ** 2
    return c, r, float(np.sqrt(np.mean(res[w > 0] ** 2))) if (w > 0).any() else 1e9


def _fit_line(Q):
    c = Q.mean(0)
    d = np.linalg.eigh((Q - c).T @ (Q - c))[1][:, -1]
    return c, d, np.abs(np.cross(d, Q - c))


def _fit_arc(Q):
    A = np.c_[2 * Q, np.ones(len(Q))]
    sol = np.linalg.lstsq(A, (Q ** 2).sum(1), rcond=None)[0]
    c = sol[:2]
    r = np.sqrt(max(sol[2] + c @ c, 1e-12))
    return c, r, np.abs(np.linalg.norm(Q - c, axis=1) - r)


def _ok(e, t):
    return np.percentile(e - t, 90) <= 0 and (e <= 2 * t).all()


def _line_ok(P, i, j, T):
    return j - i >= 2 and _ok(_fit_line(P[i:j + 1])[2], T[i:j + 1])


def _arc_ok(P, i, j, T, rmax):
    if j - i < 6:
        return False
    Q = P[i:j + 1]
    c, r, e = _fit_arc(Q)
    if r > rmax or not _ok(e, T[i:j + 1]):
        return False
    ang = np.diff(np.unwrap(np.arctan2(*(Q - c).T[::-1])))
    return (ang > 0).mean() > 0.9 or (ang < 0).mean() > 0.9


def _extend(ok, i, n, k0=2):
    """Largest j (i<j<i+n) with ok(i, j); exponential + binary search."""
    j, step = i + k0, 1
    if j >= i + n or not ok(i, j):
        return i
    while j + step < i + n and ok(i, j + step):
        j += step
        step *= 2
    lo, hi = j, min(j + step, i + n - 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ok(i, mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _meet(a, b, p):
    """Junction vertex of consecutive primitives a, b nearest to p."""
    if a[0] == 'L' and b[0] == 'L':
        a, b = a[:3], b[:3]
        den = np.cross(a[2], b[2])
        if abs(den) > 0.02:
            return a[1] + np.cross(b[1] - a[1], b[2]) / den * a[2]
        return p
    if a[0] == 'L' or b[0] == 'L':
        (_, c0, d), (_, cc, r) = [q[:3] for q in ((a, b) if a[0] == 'L' else (b, a))]
        f = c0 - cc
        B, C = f @ d, f @ f - r * r
        disc = B * B - C
        if disc < 0:
            return p
        xs = [c0 + (-B + s * np.sqrt(disc)) * d for s in (1, -1)]
    else:
        (_, c1, r1), (_, c2, r2) = a[:3], b[:3]
        D = np.linalg.norm(c2 - c1)
        if D < 1e-9 or D > r1 + r2 or D < abs(r1 - r2):
            return p
        t = (r1 * r1 - r2 * r2 + D * D) / (2 * D)
        u = (c2 - c1) / D
        hgt = np.sqrt(max(r1 * r1 - t * t, 0))
        xs = [c1 + t * u + s * hgt * np.array([-u[1], u[0]]) for s in (1, -1)]
    x = min(xs, key=lambda q: np.linalg.norm(q - p))
    return x if np.linalg.norm(x - p) < 1.0 else p


def fit_ring(P, tol, ds=None, hole=False):
    """Closed ring -> entities: [('C', cx, cy, r)] or list of ('L', p0, p1) / ('A', p0, pm, p1).
    tol: scalar or callable(points)->per-point tolerance. Primitives are least-squares fitted and
    joined at their exact intersections (noise-free vertices)."""
    t0 = float(np.median(tol(np.asarray(P)[:, :2]))) if callable(tol) else float(tol)
    P = resample(P, ds or min(max(t0 * 0.7, 0.05), 0.5))
    T = tol(P) if callable(tol) else np.full(len(P), tol)
    c, r, rms = fit_circle(P)
    e = np.abs(np.linalg.norm(P - c, axis=1) - r)
    if len(P) > 12 and (_ok(e, T * 1.5) or (hole and np.percentile(e, 80) <= max(2 * T.max(), 0.12 * r)
                                             and np.percentile(e, 98) <= max(6 * T.max(), 0.2 * r))):
        return [('C', float(c[0]), float(c[1]), float(r))]
    d1, d2 = np.roll(P, -3, 0) - P, P - np.roll(P, 3, 0)
    s = int(np.abs(np.arctan2(np.cross(d2, d1), (d1 * d2).sum(1))).argmax())
    P, T = np.roll(P, -s, 0), np.roll(T, -s)
    P, T = np.vstack([P, P[:1]]), np.r_[T, T[:1]]
    n, rmax = len(P), 3 * np.ptp(P, 0).max()
    prims, i = [], 0
    while i < n - 1:
        jl = _extend(lambda a, b: _line_ok(P, a, b, T), i, n - i)
        ja = _extend(lambda a, b: _arc_ok(P, a, b, T, rmax), i, n - i, 6)
        if ja - i > (jl - i) * 1.3 + 3:
            j = ja
            cc, rr, _ = _fit_arc(P[i:j + 1])
            prims.append(['A', cc, rr, i, j])
        else:
            j = max(jl, min(i + 2, n - 1))
            cc, dd, _ = _fit_line(P[i:j + 1])
            prims.append(['L', cc, dd, i, j])
        i = j
    # merge collinear lines / concentric arcs
    out = []
    for q in prims:
        if out and q[0] == out[-1][0] == 'L' and abs(np.cross(q[2], out[-1][2])) < 0.02 \
                and abs(np.cross(out[-1][2], q[1] - out[-1][1])) < t0:
            o = out[-1]
            o[1], o[2], _ = _fit_line(P[o[3]:q[4] + 1]); o[4] = q[4]
            continue
        if out and q[0] == out[-1][0] == 'A' and np.linalg.norm(q[1] - out[-1][1]) < t0 and abs(q[2] - out[-1][2]) < t0:
            o = out[-1]
            o[1], o[2], _ = _fit_arc(P[o[3]:q[4] + 1]); o[4] = q[4]
            continue
        out.append(q)
    k = len(out)
    V = [_meet(out[m - 1], out[m], P[out[m][3]]) for m in range(k)]
    ents = []
    for m, q in enumerate(out):
        a, b = V[m], V[(m + 1) % k]
        if q[0] == 'L':
            ents.append(('L', a, b))
        else:
            mid = P[(q[3] + q[4]) // 2] - q[1]
            ents.append(('A', a, q[1] + q[2] * mid / np.linalg.norm(mid), b))
    return ents


def merge(ents, ang=1.0):
    """Merge collinear consecutive lines."""
    out = []
    for e in ents:
        if out and e[0] == 'L' and out[-1][0] == 'L':
            a, b = out[-1][1], out[-1][2]
            u, v = b - a, e[2] - e[1]
            if abs(np.degrees(np.arctan2(np.cross(u, v), u @ v))) < ang:
                out[-1] = ('L', a, e[2])
                continue
        out.append(e)
    return out


def snap_axes(ents, ang=1.5):
    """Mechanical regularization: near-horizontal/vertical lines -> exact; recompute line-line corners."""
    ents = [list(e) for e in ents]
    n = len(ents)
    if n < 3 or ents[0][0] == 'C':
        return ents
    fixed = []
    for e in ents:
        if e[0] != 'L':
            fixed.append(None)
            continue
        d = e[2] - e[1]
        a = np.degrees(np.arctan2(d[1], d[0])) % 180
        k = round(a / 90) * 90
        fixed.append(k % 180 if abs(a - k) < ang else None)
    for i, e in enumerate(ents):
        if fixed[i] is None:
            continue
        m = (e[1] + e[2]) / 2
        horiz = fixed[i] == 0
        for k in (1, 2):
            if horiz:
                e[k] = np.array([e[k][0], m[1]])
            else:
                e[k] = np.array([m[0], e[k][1]])
    # re-join: shared vertex = intersection for line-line, else midpoint
    for i in range(n):
        a, b = ents[i], ents[(i + 1) % n]
        p = _isect(a, b) if a[0] == 'L' and b[0] == 'L' else None
        if p is None:
            p = (a[-1] + b[1]) / 2 if fixed[i] is None and fixed[(i + 1) % n] is None else (
                a[-1] if fixed[i] is not None else b[1])
        a[-1], b[1] = p, p
    return [tuple(e) for e in ents]


def _isect(a, b):
    p, r = a[1], a[2] - a[1]
    q, s = b[1], b[2] - b[1]
    den = np.cross(r, s)
    if abs(den) < 1e-9 * np.linalg.norm(r) * np.linalg.norm(s):
        return None
    t = np.cross(q - p, s) / den
    x = p + t * r
    return x if np.linalg.norm(x - a[2]) < 3 * max(np.linalg.norm(r), np.linalg.norm(s)) else None


def polygon_entities(poly, tol, min_hole=0.0):
    """shapely Polygon -> (exterior_entities, [interior_entities...])."""
    ext = snap_axes(fit_ring(np.asarray(poly.exterior.coords), tol))
    ints = [snap_axes(fit_ring(np.asarray(r.coords), tol, hole=True)) for r in poly.interiors
            if Polygon(r).area > min_hole]
    return ext, ints


def ents_polygon(ents, n_arc=16):
    """Entities -> shapely Polygon (for comparison / voxelization)."""
    if ents[0][0] == 'C':
        _, x, y, r = ents[0]
        return shapely.Point(x, y).buffer(r, 64)
    pts = []
    for e in ents:
        cr = circle3(e[1], e[2], e[3]) if e[0] == 'A' else None
        if cr is None:
            pts.append(e[1])
        else:
            c, r = cr
            a = [np.arctan2(*(p - c)[::-1]) for p in (e[1], e[2], e[3])]
            a1, a2 = np.unwrap([a[0], a[1], a[2]])[[1, 2]]
            t = np.r_[np.linspace(a[0], a1, n_arc // 2, endpoint=False), np.linspace(a1, a2, n_arc // 2, endpoint=False)]
            pts.extend(c + r * np.c_[np.cos(t), np.sin(t)])
    return Polygon(pts).buffer(0) if len(pts) >= 3 else Polygon()
