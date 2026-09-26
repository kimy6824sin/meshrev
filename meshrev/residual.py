"""Analysis-by-synthesis: voxel difference between scan and current model -> extra prism / hole features.
Handles cast bodies, bosses, windows, cross holes along any principal axis with coarse (tier C) tolerance."""
import numpy as np
import shapely
from scipy import ndimage
from . import slicer as S


class Grid:
    def __init__(self, m, h):
        lo, hi = m.bounds[0] - 2 * h, m.bounds[1] + 2 * h
        self.h = h
        self.ax = [np.arange(lo[k] + h / 2, hi[k], h) for k in range(3)]  # x, y, z centers
        self.shape = tuple(len(a) for a in self.ax[::-1])  # (nz, ny, nx)

    def coords(self):
        Z, Y, X = np.meshgrid(self.ax[2], self.ax[1], self.ax[0], indexing='ij')
        return X, Y, Z


def mesh_voxels(m, g):
    return S.voxels(S.sections(m, g.ax[2]), g.ax[0], g.ax[1])


def _uv(k, X, Y, Z):
    P = (X, Y, Z)
    return P[(k + 1) % 3], P[(k + 2) % 3], P[k]


def rasterize(feats, g):
    """Voxelize a feature list (same semantics as the generated CAD script)."""
    X, Y, Z = g.coords()
    V = np.zeros(g.shape, bool)
    for f in feats:
        op = f['op']
        if op == 'prism':
            u, v, s = _uv(f['ax'], X, Y, Z)
            poly = S.ents_polygon(f['outer'])
            for e in f.get('inner', []):
                poly = poly.difference(S.ents_polygon(e))
            mk = (s >= f['s0']) & (s <= f['s1']) & shapely.contains_xy(poly, u, v)
        elif op == 'revolve':
            mk = shapely.contains_xy(S.ents_polygon(f['profile']), np.hypot(X, Y), Z)
        elif op == 'gear':
            mk = (Z >= f['s0']) & (Z <= f['s1']) & shapely.contains_xy(gear_polygon(f), X, Y)
        elif op in ('hole', 'holes'):
            mk = np.zeros(g.shape, bool)
            for p in hole_points(f):
                a = np.asarray(f['ax'], float)
                d = np.stack([X - p[0], Y - p[1], Z - p[2]], -1)
                t = d @ a
                mk |= (t >= 0) & (t <= f['h']) & (np.linalg.norm(d - t[..., None] * a, axis=-1) <= f['d'] / 2)
            V &= ~mk
            continue
        else:
            continue
        V = V | mk if f.get('mode', 'add') == 'add' else V & ~mk
    return V


def hole_points(f):
    if f['op'] == 'hole':
        return [f['p']]
    c, r, k, out = f['c'], f['pcd'] / 2, f['k'], []
    for i in range(f['n']):
        a = np.radians(f['a0'] + 360 * i / f['n'])
        p = [0.0, 0.0, 0.0]
        p[(k + 1) % 3], p[(k + 2) % 3], p[k] = c[0] + r * np.cos(a), c[1] + r * np.sin(a), f['z0']
        out.append(p)
    return out


def gear_polygon(f):
    phi, rho = np.asarray(f['phi']), np.asarray(f['rho'])
    th = np.concatenate([phi + 2 * np.pi * i / f['n'] for i in range(f['n'])])
    rr = np.tile(rho, f['n'])
    return shapely.Polygon(np.c_[f['c'][0] + rr * np.cos(th), f['c'][1] + rr * np.sin(th)]).buffer(0)


def _prism_fit(C, k, g):
    """Best layered prism decomposition of component mask C along axis k -> [(s0, s1, mask2d)], score."""
    Ck = np.moveaxis(C, 2 - k, 0)  # axis k first; remaining axes ordered to match (v, u)
    if k == 1:
        Ck = Ck.transpose(0, 2, 1)
    lay = np.flatnonzero(Ck.any((1, 2)))
    groups, cur = [], [lay[0]]
    for i in lay[1:]:
        a, b = Ck[cur[0]], Ck[i]
        jac = (a & b).sum() / max((a | b).sum(), 1)
        if i == cur[-1] + 1 and jac > 0.7:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    merged = []  # layers thinner than 3 voxels join the previous group
    for gr in groups:
        if merged and (len(gr) < 3 or len(merged[-1]) < 3) and gr[0] == merged[-1][-1] + 1:
            merged[-1] += gr
        else:
            merged.append(gr)
    groups = merged
    out, hit, vol = [], 0, 0
    for gr in groups:
        rep = Ck[gr].mean(0) >= 0.5
        if not rep.any():
            continue
        hit += (Ck[gr] & rep).sum()
        vol += rep.sum() * len(gr)
        s = g.ax[k]
        out.append((s[gr[0]] - g.h / 2, s[gr[-1]] + g.h / 2, rep))
    score = hit / max(C.sum() + vol - hit, 1) - 0.02 * len(out)
    return out, score


def residual_features(m, feats, g, M, tol, min_vol=None, mode_rp=True):
    """Extract add/cut prisms from the voxel difference."""
    B = rasterize(feats, g)
    st = ndimage.generate_binary_structure(3, 1)
    min_vol = min_vol or max(2.0, (2 * tol) ** 3)
    new = []
    for mode, R in (('add', M & ~B), ('cut', B & ~M)):
        R = ndimage.binary_opening(R, st, iterations=1)
        lab, n = ndimage.label(R, st)
        if not n:
            continue
        sizes = ndimage.sum(R, lab, range(1, n + 1)) * g.h ** 3
        for ci in np.flatnonzero(sizes >= min_vol) + 1:
            C = lab == ci
            best = max(((k, *_prism_fit(C, k, g)) for k in range(3)), key=lambda t: t[2])
            k, layers, sc = best
            if sc < 0.4:
                continue
            ku, kv = (k + 1) % 3, (k + 2) % 3
            for s0, s1, rep in layers:
                for p in S.contours(rep, 0.5, g.ax[ku][0], g.ax[kv][0], g.h, min_area=4 * g.h ** 2):
                    ext, ints = S.polygon_entities(p, max(tol, g.h))
                    if S.ents_polygon(ext).area < 4 * g.h ** 2:
                        continue
                    new.append(dict(op='prism', ax=k, s0=float(s0), s1=float(s1), outer=ext, inner=ints,
                                    mode=mode, tier='C', note=f'residual {mode}'))
    return new
