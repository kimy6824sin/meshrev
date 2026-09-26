"""Base body recognition: layered extrusion (prismatic) or revolve profile (rotational), plus holes."""
import numpy as np
import shapely
from shapely.geometry import Polygon
from . import slicer as S


def _irls_mean(x, w, it=5):
    m = np.average(x, weights=w)
    for _ in range(it):
        r = x - m
        s = 4.685 * max(1.4826 * np.median(np.abs(r)), 1e-4)
        ww = w * np.clip(1 - (r / s) ** 2, 0, 1) ** 2
        m = np.average(x, weights=ww) if ww.sum() > 0 else m
    return m


def zplanes(m, fn, ang=2.0, minfrac=0.003, gap=0.3):
    """Heights of planar faces normal to Z (up or down facing) -> [(z, area_frac, dir)]."""
    c, a = m.triangles_center[:, 2], m.area_faces
    out = []
    for sgn in (1, -1):
        sel = sgn * fn[:, 2] > np.cos(np.radians(ang))
        z, w = c[sel], a[sel]
        if not len(z):
            continue
        o = np.argsort(z)
        z, w = z[o], w[o]
        cut = np.flatnonzero(np.diff(z) > gap) + 1
        for zz, ww in zip(np.split(z, cut), np.split(w, cut)):
            if ww.sum() > minfrac * a.sum():
                out.append((float(_irls_mean(zz, ww)), float(ww.sum() / a.sum()), sgn))
    return sorted(out)


def _dist(p, q):
    """Mean offset distance between two sections."""
    L = p.length + q.length
    return p.symmetric_difference(q).area / (L / 2) if L > 0 else 0.0


def levels(m, fn, tol, dz=None):
    """Split height into levels of (near) constant cross-section. Boundaries snap to planar faces."""
    ez = m.bounds[1][2]
    dz = dz or float(np.clip(ez / 200, 0.05, 0.5))
    zs = np.arange(dz / 2, ez, dz)
    P = S.sections(m, zs)
    zp = zplanes(m, fn)
    bnd = sorted({0.0, ez, *[z for z, _, _ in zp if dz < z < ez - dz]})
    out = []

    def split(a, b):
        idx = np.flatnonzero((zs > a + dz) & (zs < b - dz))
        if len(idx) < 3:
            idx = np.flatnonzero((zs > a) & (zs < b))
        if not len(idx):
            return
        d = [_dist(P[idx[0]], P[i]) for i in idx]
        if max(d) <= tol or b - a < 4 * dz:
            out.append((a, b, P[idx[len(idx) // 2]]))
            return
        # split where consecutive change is largest
        dd = [_dist(P[i], P[i + 1]) for i in idx[:-1]]
        k = int(np.argmax(dd))
        zc = (zs[idx[k]] + zs[idx[k + 1]]) / 2
        split(a, zc)
        split(zc, b)
    for a, b in zip(bnd[:-1], bnd[1:]):
        split(a, b)
    # merge thin transition levels (fillets/chamfers) into the more similar neighbour
    minh = max(3 * dz, 0.5)

    def thin(i):  # too thin, or a thin transition (chamfer/fillet) close to a neighbour
        a, b, p = out[i]
        nb = [out[j][2] for j in (i - 1, i + 1) if 0 <= j < len(out)]
        return b - a < minh or (b - a < 1.5 and min(_dist(p, q) for q in nb) < b - a)
    while len(out) > 1:
        cand = [i for i in range(len(out)) if thin(i)]
        if not cand:
            break
        i = min(cand, key=lambda j: out[j][1] - out[j][0])
        a, b, p = out[i]
        if i == 0 or (i < len(out) - 1 and _dist(p, out[i + 1][2]) < _dist(p, out[i - 1][2])):
            j = i + 1
            out[j] = (a, out[j][1], out[j][2])
        else:
            j = i - 1
            out[j] = (out[j][0], b, out[j][2])
        out.pop(i)
    # merge equal neighbours
    res = [out[0]]
    for a, b, p in out[1:]:
        if _dist(res[-1][2], p) <= tol:
            res[-1] = (res[-1][0], b, res[-1][2] if res[-1][1] - res[-1][0] > b - a else p)
        else:
            res.append((a, b, p))
    err = sum(P[k].symmetric_difference(g).area * dz for a, b, g in res for k in np.flatnonzero((zs > a) & (zs < b)))
    return res, zp, err


def refine_hole(m, p, k, r, h, band=0.6):
    """Least-squares circle on mesh vertices of a cylinder along principal axis k (0/1/2) starting at p.
    Returns refined (p, d, h, tilt_deg); the axial range is re-measured from the cylinder's vertices."""
    v = m.vertices
    u, w, s = v[:, (k + 1) % 3], v[:, (k + 2) % 3], v[:, k]
    cu, cw, s0 = p[(k + 1) % 3], p[(k + 2) % 3], p[k]
    rho = np.hypot(u - cu, w - cw)
    sel = (np.abs(rho - r) < band) & (s > s0 + 0.1 * h) & (s < s0 + 0.9 * h)
    if sel.sum() < 30:
        return None
    c, rr, rms = S.fit_circle(np.c_[u[sel], w[sel]])
    ang = np.arctan2(w[sel] - c[1], u[sel] - c[0])
    if rms > 0.1 * rr or len(np.unique((ang // 0.3).astype(int))) < 0.6 * 21:  # needs >60% angular coverage
        return None
    sm = np.median(s[sel])
    lo, hi = sel & (s < sm), sel & (s >= sm)
    tilt = 0.0
    if lo.sum() > 8 and hi.sum() > 8:
        c0, c1 = S.fit_circle(np.c_[u[lo], w[lo]])[0], S.fit_circle(np.c_[u[hi], w[hi]])[0]
        tilt = float(np.degrees(np.arctan2(np.linalg.norm(c1 - c0), np.median(s[hi]) - np.median(s[lo]))))
    q = list(p)
    q[(k + 1) % 3], q[(k + 2) % 3] = float(c[0]), float(c[1])
    return q, float(2 * rr), h, tilt, float(rms)


def stack(m, fn, tol):
    """Prismatic body: list of Z-prisms + vertical holes."""
    lv, zp, err = levels(m, fn, tol)
    feats, holes = [], []
    for a, b, g in lv:
        for p in getattr(g, 'geoms', [g]):
            if p.area < 4 * tol * tol:
                continue
            ext, ints = S.polygon_entities(p, tol, min_hole=tol * tol)
            if ext[0][0] == 'C' or len(ext) >= 2:
                cuts = []
                for e in ints:
                    if e[0][0] == 'C':
                        holes.append([e[0][1], e[0][2], e[0][3], a, b])
                    else:
                        cuts.append(e)
                feats.append(dict(op='prism', ax=2, s0=a, s1=b, outer=ext, inner=cuts))
    return feats, merge_holes(m, holes, tol), dict(levels=[(a, b) for a, b, _ in lv], zplanes=zp, err=err)


def merge_holes(m, holes, tol):
    """Merge coaxial equal hole segments of adjacent levels, refine on mesh."""
    holes.sort(key=lambda h: h[3])
    out = []
    for h in holes:
        for o in out:
            if np.hypot(o[0] - h[0], o[1] - h[1]) < 3 * tol and abs(o[2] - h[2]) < 3 * tol and abs(o[4] - h[3]) < 1e-6:
                o[4] = h[4]
                break
        else:
            out.append(list(h))
    res = []
    for cx, cy, r, z0, z1 in out:
        rh = refine_hole(m, [cx, cy, z0], 2, r, z1 - z0)
        p, d, h, tilt, rough = rh or ([cx, cy, z0], 2 * r, z1 - z0, 0.0, 0.0)
        if tilt > 5:  # not a straight hole (taper / cast core): keep as a cut prism
            res.append(dict(op='prism', ax=2, s0=z0, s1=z1, outer=[('C', cx, cy, r)], inner=[], mode='cut', tier='B'))
            continue
        res.append(dict(op='hole', p=p, ax=[0, 0, 1], d=d, h=h, tilt=tilt, k=2, rough=rough))
    return res


# ---------------- rotational ----------------

def axis_center(m, fn):
    """Least-squares point where normal lines of faces parallel to Z meet (the revolve axis)."""
    sel = np.abs(fn[:, 2]) < 0.2
    p, n = m.triangles_center[sel, :2], fn[sel, :2]
    n = n / np.linalg.norm(n, axis=1)[:, None]
    w = m.area_faces[sel]
    c = np.zeros(2)
    for _ in range(8):
        P = np.eye(2)[None] - n[:, :, None] * n[:, None, :]  # projectors orthogonal to normal
        A = (w[:, None, None] * P).sum(0)
        b = (w[:, None, None] * P @ p[:, :, None]).sum(0)[:, 0]
        c = np.linalg.solve(A, b)
        d = p - c
        r = np.abs(np.cross(n, d))
        s = 4.685 * max(1.4826 * np.median(r), 1e-3)
        w = m.area_faces[sel] * np.clip(1 - (r / s) ** 2, 0, 1) ** 2
    return c


def rz_profile(m, h=None, nth=360):
    """Angular occupancy image in (r, z) half plane."""
    ez, R = m.bounds[1][2], np.hypot(m.vertices[:, 0], m.vertices[:, 1]).max()
    h = h or float(np.clip(max(ez, R) / 300, 0.05, 0.4))
    zs, rs = np.arange(h / 2, ez, h), np.arange(h / 2, R + h, h)
    occ = S.rz_occupancy(S.sections(m, zs), rs, nth)
    return occ, rs, zs, h


def rev_error(occ, rs, h):
    """Volume mis-modelled by the best revolve (occupancy thresholded at 0.5)."""
    return float((np.minimum(occ, 1 - occ) * 2 * np.pi * rs[None, :] * h * h).sum())


def revolve(m, occ, rs, zs, h, tol, zp):
    """Revolve profile (r, z) from angular occupancy >= 0.5, snapped to planar heights and refined radii."""
    polys = S.contours(occ, 0.5, rs[0], zs[0], h, min_area=4 * h * h)
    feats = []
    zv = [z for z, _, _ in zp]
    for p in polys:
        # clip r >= 0 and close on the axis
        p = p.intersection(shapely.box(0, -1, rs[-1] + 1, zs[-1] + 1))
        for q in getattr(p, 'geoms', [p]):
            if q.area < 4 * h * h:
                continue
            ents = S.snap_axes(S.fit_ring(np.asarray(q.exterior.coords), max(tol, h / 2)))
            ents = _refine_profile(m, ents, zv, h)
            feats.append(dict(op='revolve', profile=ents))
    return feats


def _refine_profile(m, ents, zv, h):
    """Snap horizontal lines to planar face heights, refine vertical lines' radius on mesh vertices."""
    v = m.vertices
    rho = np.hypot(v[:, 0], v[:, 1])
    ents = [list(e) for e in ents]
    for e in ents:
        if e[0] != 'L':
            continue
        (r0, z0), (r1, z1) = e[1], e[2]
        if abs(z0 - z1) < 1e-9 and zv:  # horizontal -> planar face
            zz = min(zv, key=lambda z: abs(z - z0))
            if abs(zz - z0) < 2 * h:
                e[1], e[2] = np.array([r0, zz]), np.array([r1, zz])
        elif abs(r0 - r1) < 1e-9:  # vertical -> cylinder
            lo, hi = min(z0, z1), max(z0, z1)
            sel = (np.abs(rho - r0) < 2 * h) & (v[:, 2] > lo + 0.15 * (hi - lo)) & (v[:, 2] < hi - 0.15 * (hi - lo))
            if sel.sum() > 20:
                rr = _irls_mean(rho[sel], np.ones(sel.sum()))
                e[1], e[2] = np.array([rr, z0]), np.array([rr, z1])
    n = len(ents)
    for i in range(n):  # re-join
        a, b = ents[i], ents[(i + 1) % n]
        if a[0] == 'L' and b[0] == 'L':
            p = S._isect(a, b)
            if p is not None:
                a[-1], b[1] = p, p
                continue
        b[1] = a[-1]
    return [tuple(e) for e in ents]
