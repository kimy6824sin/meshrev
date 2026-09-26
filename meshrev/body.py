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
    ez = float(np.percentile(m.vertices[:, 2], 99.98))
    dz = dz or float(np.clip(ez / 200, 0.05, 0.5))
    zs = np.arange(dz / 2, ez, dz)
    P = S.sections(m, zs)
    zp = zplanes(m, fn)
    if zp and abs(zp[-1][0] - ez) < 0.5:  # datum faces define the ends, not noisy extreme vertices
        ez = zp[-1][0]
    z0 = zp[0][0] if zp and abs(zp[0][0]) < 0.5 else 0.0
    bnd = sorted({z0, ez, *[z for z, _, _ in zp if z0 + dz < z < ez - dz]})
    out = []

    def split(a, b):
        idx = np.flatnonzero((zs > a + dz) & (zs < b - dz))
        if len(idx) < 3:
            idx = np.flatnonzero((zs > a) & (zs < b))
        if not len(idx):
            return
        d = [_dist(P[idx[0]], P[i]) for i in idx]
        if max(d) <= tol or b - a < 4 * dz:
            out.append((a, b, P[idx[len(idx) // 2]], 0.0))
            return
        # casting draft: section offsets uniformly with height -> one tapered extrusion
        pa, pb = P[idx[0]], P[idx[-1]]
        za, zb = zs[idx[0]], zs[idx[-1]]
        delta = (pb.area - pa.area) / max((pa.length + pb.length) / 2, 1e-9)
        if b - a > 1.0 and _dist(pa.buffer(delta, join_style=2), pb) <= tol and \
                all(_dist(pa.buffer(delta * (zs[i] - za) / (zb - za), join_style=2), P[i]) <= 1.5 * tol for i in idx):
            ang = float(np.degrees(np.arctan(-delta / (zb - za))))
            if 0.2 < abs(ang) < 15:
                out.append((a, b, pa.buffer(-delta * (za - a) / (zb - za), join_style=2), ang))
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
        a, b, p, _ = out[i]
        nb = [out[j][2] for j in (i - 1, i + 1) if 0 <= j < len(out)]
        return b - a < minh or (b - a < 1.5 and min(_dist(p, q) for q in nb) < b - a)
    while len(out) > 1:
        cand = [i for i in range(len(out)) if thin(i)]
        if not cand:
            break
        i = min(cand, key=lambda j: out[j][1] - out[j][0])
        a, b, p, _ = out[i]
        if i == 0 or (i < len(out) - 1 and _dist(p, out[i + 1][2]) < _dist(p, out[i - 1][2])):
            j = i + 1
            q = out[j]
            out[j] = (a, q[1], q[2].buffer(np.tan(np.radians(q[3])) * (q[0] - a), join_style=2) if q[3] else q[2], q[3])
        else:
            j = i - 1
            out[j] = (out[j][0], b, out[j][2], out[j][3])
        out.pop(i)
    # merge equal neighbours
    res = [out[0]]
    for a, b, p, t in out[1:]:
        r = res[-1]
        if not t and not r[3] and _dist(r[2], p) <= tol:
            res[-1] = (r[0], b, r[2] if r[1] - r[0] > b - a else p, 0.0)
        else:
            res.append((a, b, p, t))
    err = sum(P[k].symmetric_difference(g).area * dz for a, b, g, _ in res for k in np.flatnonzero((zs > a) & (zs < b)))
    return res, zp, err


def refine_hole(m, p, k, r, h, band=0.6, measure=True):
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
    x = np.vstack([v, m.triangles_center])  # centres help on long CAD-like triangles
    xu, xw, xs = x[:, (k + 1) % 3], x[:, (k + 2) % 3], x[:, k]
    ins = (np.abs(np.hypot(xu - c[0], xw - c[1]) - rr) < max(3 * rms, 0.1) + 0.02 * rr) & (xs > s0 - 0.5 * h) & (xs < s0 + 1.5 * h)
    ss = np.sort(xs[ins])
    if measure and len(ss) > 20:  # contiguous run overlapping the seed range
        runs = np.split(ss, np.flatnonzero(np.diff(ss) > max(0.3 * h, 2.0)) + 1)
        run = max(runs, key=lambda r: min(r[-1], s0 + h) - max(r[0], s0))
        q[k], h = float(run[0]), float(run[-1] - run[0])
    return q, float(2 * rr), h, tilt, float(rms)


def stack(m, fn, tol):
    """Prismatic body: list of Z-prisms + vertical holes."""
    lv, zp, err = levels(m, fn, tol)
    feats, holes = [], []
    for a, b, g, taper in lv:
        for p in getattr(g, 'geoms', [g]):
            if p.area < 4 * tol * tol:
                continue
            c = np.array(p.centroid.coords[0])
            ok, k, F, t, rho = _spectrum(p, c)
            gf = gear_feature([(t, rho)], k, c, a, b) if ok and not taper else None
            if gf:
                gf['r_in'] = gf['r_root'] - max(1.0, 2 * tol)
                feats.append(gf)
                p = Polygon(shapely.Point(*c).buffer(gf['r_root'] + tol, 256).exterior, [r.coords for r in p.interiors])
            ext, ints = S.polygon_entities(p, tol, min_hole=tol * tol)
            if ext[0][0] == 'C' or len(ext) >= 2:
                cuts = []
                for e in ints:
                    if e[0][0] == 'C':
                        holes.append([e[0][1], e[0][2], e[0][3], a, b])
                    else:
                        cuts.append(e)
                feats.append(dict(op='prism', ax=2, s0=a, s1=b, outer=ext, inner=cuts, taper=taper))
    return feats, merge_holes(m, holes, tol), dict(levels=[(a, b) for a, b, _, _ in lv], zplanes=zp, err=err)


def merge_holes(m, holes, tol):
    """Merge coaxial equal hole segments of adjacent levels, refine on mesh."""
    holes.sort(key=lambda h: h[3])
    out = []
    for h in holes:
        for o in out:
            if np.hypot(o[0] - h[0], o[1] - h[1]) < tol and abs(o[2] - h[2]) < max(0.05, tol / 2) and abs(o[4] - h[3]) < 1e-6:
                o[4] = h[4]
                break
        else:
            out.append(list(h))
    res = []
    for cx, cy, r, z0, z1 in out:
        rh = refine_hole(m, [cx, cy, z0], 2, r, z1 - z0, measure=False)
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
    # refine with robust circle fits of near-circular outer contours (immune to tangential flank normals)
    ez = m.bounds[1][2]
    cs = []
    for g in S.sections(m, np.linspace(0.1 * ez, 0.9 * ez, 9)):
        if g.is_empty:
            continue
        p = max(getattr(g, 'geoms', [g]), key=lambda q: q.area)
        cc, rr, rms = S.fit_circle(S.resample(np.asarray(p.exterior.coords), 0.2))
        if rms < 0.05 * rr and np.linalg.norm(cc - c) < 0.05 * rr:
            cs.append(cc)
    return np.median(cs, axis=0) if cs else c


def rz_profile(m, h=None, nth=360):
    """Angular occupancy image in (r, z) half plane."""
    ez, R = m.bounds[1][2], np.hypot(m.vertices[:, 0], m.vertices[:, 1]).max()
    h = h or float(np.clip(max(ez, R) / 300, 0.05, 0.4))
    zs, rs = np.arange(h / 2, ez, h), np.arange(h / 2, R + h, h)
    occ = S.rz_occupancy(S.sections(m, zs), rs, nth)
    return occ, rs, zs, h


def revolve(m, occ, rs, zs, h, tol, zp):
    """Revolve profile (r, z) from angular occupancy >= 0.5, snapped to planar heights and refined radii."""
    polys = S.contours(occ, 0.5, rs[0], zs[0], h, min_area=4 * h * h)
    amax = max((p.area for p in polys), default=0)
    polys = [p for p in polys if p.area > 0.02 * amax]
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
        elif abs(r0 - r1) < 1e-9 and r0 < max(2 * h, 0.2):  # on the axis: exactly r = 0
            e[1], e[2] = np.array([0.0, z0]), np.array([0.0, z1])
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
    return S.despike([tuple(e) for e in ents])


def axis_holes(m, k, tol, exclude_axis=False):
    """Straight holes along principal axis k: circular inner loops consistent over several sections,
    refined by a least-squares cylinder on the mesh. Returns hole features."""
    u, v = (k + 1) % 3, (k + 2) % 3
    mm = m.copy()
    mm.vertices = m.vertices[:, [u, v, k]]  # cyclic permutation keeps orientation
    lo, hi = mm.bounds[:, 2]
    step = float(np.clip((hi - lo) / 60, 0.3, 2.0))
    ss = np.arange(lo + step / 2, hi, step)
    size = float(np.max(mm.extents[:2]))
    circ = []
    for s, g in zip(ss, S.sections(mm, ss)):
        for p in getattr(g, 'geoms', [g]):
            for r in p.interiors:
                if Polygon(r).area > (0.4 * size) ** 2 * np.pi:
                    continue
                e = S.fit_ring(np.asarray(r.coords), tol, hole=True)
                if e[0][0] == 'C' and not (exclude_axis and np.hypot(e[0][1], e[0][2]) < 3 * tol):
                    circ.append([e[0][1], e[0][2], e[0][3], s])
    groups = []
    for c in circ:
        for gr in groups:
            q = gr[-1]
            if abs(c[3] - q[3]) < 1.6 * step and np.hypot(c[0] - q[0], c[1] - q[1]) < 3 * tol + 0.05 * q[2] \
                    and abs(c[2] - q[2]) < 3 * tol + 0.05 * q[2]:
                gr.append(c)
                break
        else:
            groups.append([c])
    out = []
    for gr in groups:
        if len(gr) < 2:
            continue
        a = np.array(gr)
        p = [0.0, 0.0, 0.0]
        p[u], p[v], p[k] = np.median(a[:, 0]), np.median(a[:, 1]), a[0, 3] - step
        rh = refine_hole(m, p, k, float(np.median(a[:, 2])), a[-1, 3] - a[0, 3] + 2 * step)
        if rh is None or rh[3] > 5:
            continue
        p, d, h, tilt, rms = rh
        ax = [0, 0, 0]
        ax[k] = 1
        out.append(dict(op='hole', p=p, ax=ax, d=d, h=h, tilt=tilt, k=k, rough=rms))
    return out


def _polar_profile(poly, c, n=4096):
    P = np.asarray(poly.exterior.coords)[:-1, :2] - c
    th, rho = np.arctan2(P[:, 1], P[:, 0]), np.hypot(P[:, 0], P[:, 1])
    o = np.argsort(th)
    t = np.linspace(-np.pi, np.pi, n, endpoint=False)
    return t, np.interp(t, th[o], rho[o], period=2 * np.pi)


def _spectrum(p, c, kmin=8):
    t, rho = _polar_profile(p, np.asarray(c))
    F = np.abs(np.fft.rfft(rho - rho.mean())) * 2 / len(rho)
    k = kmin + int(np.argmax(F[kmin:400]))
    ok = F[k] > 0.3 and F[k] > 8 * np.median(F[kmin:400])
    return ok, k, F, t, rho


def _fold(profiles, n, bins=20):
    """Average tooth over all pitches of all slices (per-slice phase alignment)."""
    phs, rhs, ref = [], [], None
    for t, rho in profiles:
        a = np.angle(np.fft.rfft(rho)[n])
        ref = a if ref is None else ref
        phs.append((t * n + a - ref) % (2 * np.pi))
        rhs.append(rho)
    ph, rh = np.concatenate(phs), np.concatenate(rhs)
    b = (ph / (2 * np.pi) * bins).astype(int) % bins
    prof = np.array([np.median(rh[b == i]) for i in range(bins)])
    explained = 1 - np.var(rh - prof[b]) / max(np.var(rh), 1e-12)
    if n < 12 or np.ptp(prof) < max(1.0, 0.02 * prof.max()) or explained < 0.7:
        return None
    shift = int(np.argmax(prof)) - bins // 2  # tooth centred in the pitch
    phi = (np.arange(bins) + 0.5 + shift) / bins * 2 * np.pi / n
    return np.roll(prof, -shift), phi


def gear_feature(profiles, n, c, s0, s1):
    r = _fold(profiles, n)
    if r is None:
        return None
    prof, phi = r
    return dict(op='gear', n=int(n), c=[float(v) for v in c], s0=float(s0), s1=float(s1), phi=phi.tolist(),
                rho=prof.tolist(), r_root=float(prof.min()), r_tip=float(prof.max()), tier='B')


def detect_gear(m, c=(0.0, 0.0), nz=16):
    """Toothed outer band of a turned part: dominant angular frequency of the section radius r(theta)."""
    ez = m.bounds[1][2]
    zs = np.linspace(0.03 * ez, 0.97 * ez, nz)
    big = lambda g: max(getattr(g, 'geoms', [g]), key=lambda q: q.area)
    hits = []
    for z, g in zip(zs, S.sections(m, zs)):
        if not g.is_empty:
            ok, k, F, t, rho = _spectrum(big(g), c)
            if ok:
                hits.append((z, k, F[k], t, rho))
    if len(hits) < 2:
        return None
    n = int(np.bincount([h[1] for h in hits]).argmax())
    band = [h for h in hits if h[1] == n]
    if len(band) < 2:
        return None
    dz = zs[1] - zs[0]
    fine = np.arange(max(band[0][0] - 1.5 * dz, 0), min(band[-1][0] + 1.5 * dz, ez), dz / 8)
    ok = np.array([not g.is_empty and _spectrum(big(g), c)[2][n] > 0.5 * band[0][2] for g in S.sections(m, fine)])
    s0, s1 = fine[ok.argmax()] - dz / 16, fine[len(ok) - 1 - ok[::-1].argmax()] + dz / 16
    mid = band[len(band) // 5: len(band) - len(band) // 5] or band  # skip tip chamfers at the band ends
    return gear_feature([(h[3], h[4]) for h in mid], n, c, s0, s1)


def outer_shell(occ, t_low=0.35):
    """Hysteresis: extend the >=0.5 region by the contiguous outer band with occupancy >= t_low
    (turned outer surface interrupted by windows/slots)."""
    out = occ.copy()
    for j, row in enumerate(occ):
        idx = np.flatnonzero(row >= t_low)
        if not len(idx):
            continue
        i = idx[-1]
        while i >= 0 and row[i] >= t_low:
            out[j, i] = max(out[j, i], 0.75)
            i -= 1
    return out
