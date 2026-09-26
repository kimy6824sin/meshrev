"""Mechanical design knowledge: standard sizes, design-intent regularization, pattern recognition."""
import numpy as np

# ISO 273 clearance holes (fine, medium, coarse) and ISO 261 tap drills, by thread size
CLEARANCE = {3: (3.2, 3.4, 3.6), 4: (4.3, 4.5, 4.8), 5: (5.3, 5.5, 5.8), 6: (6.4, 6.6, 7.0), 8: (8.4, 9.0, 10.0),
             10: (10.5, 11.0, 12.0), 12: (13.0, 13.5, 14.5), 14: (15.0, 15.5, 16.5), 16: (17.0, 17.5, 18.5),
             20: (21.0, 22.0, 24.0), 24: (25.0, 26.0, 28.0)}
TAP_DRILL = {3: 2.5, 4: 3.3, 5: 4.2, 6: 5.0, 8: 6.8, 10: 8.5, 12: 10.2, 14: 12.0, 16: 14.0, 20: 17.5, 24: 21.0}
CBORE = {3: 6.5, 4: 8.0, 5: 10.0, 6: 11.0, 8: 15.0, 10: 18.0, 12: 20.0, 16: 26.0}  # ISO 4762 socket head
BEARING_OD = [16, 19, 22, 24, 26, 28, 30, 32, 35, 37, 40, 42, 47, 52, 55, 62, 68, 72, 75, 80, 85, 90, 100, 110, 120,
              125, 130, 140, 150, 160, 170, 180]
GEAR_MODULE = [0.5, 0.6, 0.8, 1, 1.125, 1.25, 1.375, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8]
RING_GROOVE = [0.8, 1.0, 1.2, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]


def std_holes():
    t = [(d, f'M{m} 通孔({k})') for m, ds in CLEARANCE.items() for d, k in zip(ds, ('精', '中', '粗'))]
    t += [(d, f'M{m} 螺纹底孔') for m, d in TAP_DRILL.items()]
    t += [(d, f'M{m} 沉头孔') for m, d in CBORE.items()]
    t += [(float(d), f'轴承座孔 Ø{d}') for d in BEARING_OD]
    return t


def snap_value(x, tol, table=()):
    """Snap to a standard value if within tol, else to the roundest grid (1, 0.5, 0.1, 0.05) within tol/2."""
    best = min(table, key=lambda t: abs(t[0] - x), default=None)
    if best is not None and abs(best[0] - x) <= tol:
        return best[0], best[1]
    for q in (1.0, 0.5, 0.1, 0.05):
        r = round(x / q) * q
        if abs(r - x) <= tol / 2:
            return float(round(r, 3)), None
    return float(x), None


def _circle_pattern(H, tol):
    """Holes with equal size/axis/span -> polar pattern params or None."""
    n = len(H)
    k = H[0]['k']
    P = np.array([[h['p'][(k + 1) % 3], h['p'][(k + 2) % 3]] for h in H])
    if n == 2:
        c = P.mean(0)
    else:
        from .slicer import fit_circle
        c, r, rms = fit_circle(P, it=1)
        if rms > tol:
            return None
    r = np.linalg.norm(P - c, axis=1)
    if np.ptp(r) > 2 * tol:
        return None
    a = np.sort(np.degrees(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])) % 360)
    gaps = np.diff(np.r_[a, a[0] + 360])
    if np.abs(gaps - 360 / n).max() > 1.5:
        return None
    step = 360 / n
    a0 = float(np.mean((a - a[0] + step / 2) % step - step / 2) + a[0])
    return c, 2 * r.mean(), a0


def regularize(feats, holes, tol):
    """Design-intent recovery: standard hole sizes, axis snapping, patterns, round dimensions."""
    notes, table = [], std_holes()
    # equal-size intent: holes of the same axis with nearly equal diameter share one diameter
    for h in holes:
        same = [g for g in holes if g['k'] == h['k'] and abs(g['d'] - h['d']) < max(2 * tol['A'], 0.02 * h['d'])]
        h['d_mean'] = float(np.mean([g['d'] for g in same]))
    out_h = []
    for h in holes:
        d0 = h['d_mean']
        thread = [(m, t) for m, t in TAP_DRILL.items() if t - 0.2 <= d0 <= m + 0.1] if h.get('rough', 0) > max(0.015 * d0, 0.8 * tol['A']) else []
        if thread:  # rough bore between tap drill and major diameter -> scanned thread
            mm, t = thread[0]
            h['d'], lab = t, f'M{mm} 螺纹孔(建模为底孔Ø{t})'
            notes.append(f"孔实测Ø{h['d']:.2f}、表面呈螺纹状起伏 -> 判定为 M{mm} 螺纹孔")
        else:
            h['d'], lab = snap_value(d0, max(tol['A'], 0.008 * d0), [t for t in table if '底孔' not in t[1]])
        h['note'] = (lab or '孔') + f' (实测Ø{d0:.3f})'
        s0, s1 = snap_value(h['p'][h['k']], tol['A'])[0], snap_value(h['p'][h['k']] + h['h'], tol['A'])[0]
        h['p'][h['k']], h['h'] = s0, s1 - s0
        if h.get('tilt', 0) > 0.5:
            notes.append(f"孔Ø{h['d']} 实测轴线偏斜 {h['tilt']:.2f}°，按设计意图取与基准面垂直")
        out_h.append(h)
    # group equal holes (same axis, diameter, span) -> polar patterns
    used, res = set(), []
    for i, h in enumerate(out_h):
        if i in used:
            continue
        grp = [j for j, g in enumerate(out_h) if j not in used and g['k'] == h['k'] and abs(g['d'] - h['d']) < 1e-6
               and abs(g['p'][h['k']] - h['p'][h['k']]) < 3 * tol['A'] and abs(g['h'] - h['h']) < 3 * tol['A']]
        pat = _circle_pattern([out_h[j] for j in grp], tol['B']) if len(grp) >= 2 else None
        if pat:
            c, pcd, a0 = pat
            pcd, _ = snap_value(pcd, tol['B'])
            k = h['k']
            cc = [0.0, 0.0]
            cc[0], cc[1] = [snap_value(v, tol['A'])[0] for v in c]
            res.append(dict(op='holes', k=k, ax=h['ax'], d=h['d'], n=len(grp), pcd=pcd, a0=round(a0, 2),
                            c=cc, z0=h['p'][k], h=h['h'], note=h['note']))
            notes.append(f"识别圆周阵列: {len(grp)}×Ø{h['d']}，节圆Ø{pcd}")
            used.update(grp)
        else:
            res.append(h)
            used.add(i)
    for f in feats:  # round level heights of precise (base) prisms
        if f['op'] == 'prism' and f.get('tier') != 'C':
            for key in ('s0', 's1'):
                f[key] = snap_value(f[key], tol['A'])[0]
    return feats, res, notes


def summary(feats):
    out = []
    for f in feats:
        d = {'op': f['op'], 'mode': f.get('mode', 'add'), 'tier': f.get('tier', 'A')}
        if f['op'] == 'prism':
            d.update(axis='XYZ'[f['ax']], range=[round(f['s0'], 3), round(f['s1'], 3)],
                     sketch=''.join(e[0] for e in f['outer']), inner=len(f.get('inner', [])))
        elif f['op'] == 'revolve':
            d.update(profile=''.join(e[0] for e in f['profile']))
        elif f['op'] in ('hole', 'holes'):
            d.update({k: f[k] for k in ('d', 'h', 'n', 'pcd', 'note') if k in f})
        elif f['op'] == 'gear':
            d.update({k: f[k] for k in ('n', 'module', 'note') if k in f})
        out.append(d)
    return out
