"""Feature list -> readable parametric CadQuery script -> solid / STEP."""
import numpy as np

HELPERS = '''# ---------------- helpers ----------------
AX = [((0, 1, 0), (1, 0, 0)), ((0, 0, 1), (0, 1, 0)), ((1, 0, 0), (0, 0, 1))]  # sketch xDir, normal for X/Y/Z


def wire(wp, ents):
    """Closed sketch from entities: ('C',x,y,r) | ('L',x0,y0,x1,y1) | ('A',x0,y0,xm,ym,x1,y1)."""
    if ents[0][0] == 'C':
        return wp.moveTo(ents[0][1], ents[0][2]).circle(ents[0][3])
    wp = wp.moveTo(ents[0][1], ents[0][2])
    for e in ents:
        wp = wp.lineTo(e[3], e[4]) if e[0] == 'L' else wp.threePointArc((e[3], e[4]), (e[5], e[6]))
    return wp.close()


def prism(ax, s0, s1, outer, inner=(), taper=0.0):
    """Extrude a sketch along axis ax (0=X,1=Y,2=Z) from s0 to s1; taper = draft angle in deg (+ shrinks)."""
    xd, nd = AX[ax]
    org = [0, 0, 0]
    org[ax] = s0
    pl = cq.Plane(origin=tuple(org), xDir=xd, normal=nd)
    s = wire(cq.Workplane(pl), outer).extrude(s1 - s0, taper=taper) if taper else wire(cq.Workplane(pl), outer).extrude(s1 - s0)
    for e in inner:
        t = wire(cq.Workplane(pl), e)
        s = s.cut(t.extrude(s1 - s0, taper=-taper) if taper else t.extrude(s1 - s0))
    return s


def revolve(profile):
    """Revolve an (r, z) profile 360 deg about Z."""
    pl = cq.Plane(origin=(0, 0, 0), xDir=(1, 0, 0), normal=(0, -1, 0))
    return wire(cq.Workplane(pl), profile).revolve(360, (0, 0, 0), (0, 1, 0))


def hole(d, p, ax, h):
    """Cylindrical cut tool of diameter d from point p along direction ax, length h."""
    return cq.Workplane().add(cq.Solid.makeCylinder(d / 2, h, cq.Vector(*p), cq.Vector(*ax)))


def gear(n, phi, rho, s0, s1, r_in, c=(0, 0)):
    """Toothed ring: one tooth outline (phi rad, rho) repeated n times, extruded along Z from s0 to s1."""
    pts = [(c[0] + r * math.cos(p + 2 * math.pi * i / n), c[1] + r * math.sin(p + 2 * math.pi * i / n))
           for i in range(n) for p, r in zip(phi, rho)]
    wp = cq.Workplane(cq.Plane(origin=(0, 0, s0), xDir=(1, 0, 0), normal=(0, 0, 1)))
    return wp.polyline(pts).close().extrude(s1 - s0).cut(wp.moveTo(*c).circle(r_in).extrude(s1 - s0))


def polar_pt(k, cu, cv, s, r, a):
    """Point on a pitch circle of radius r, angle a (deg), in the plane normal to axis k at height s."""
    p = [0.0, 0.0, 0.0]
    p[(k + 1) % 3], p[(k + 2) % 3], p[k] = cu + r * math.cos(math.radians(a)), cv + r * math.sin(math.radians(a)), s
    return p


'''

HEAD = '''"""Parametric model reconstructed by meshrev. Edit the PARAMETERS and re-run: python {name}"""
import cadquery as cq

# ---------------- PARAMETERS (mm) ----------------
{params}

{helpers}# ---------------- FEATURE TREE ----------------
{body}

# back to the scan's original placement
result = result.val().moved(cq.Location(cq.Vector{t}, cq.Vector{ax}, {ang}))
if __name__ == "__main__":
    cq.exporters.export(cq.Workplane().add(result), __file__.replace(".py", ".step"))
'''


def _f(x):
    return float(np.round(x, 4))


def _ents(ents):
    out = []
    for e in ents:
        if e[0] == 'C':
            out.append(('C', _f(e[1]), _f(e[2]), _f(e[3])))
        else:
            out.append((e[0], *[_f(v) for p in e[1:] for v in p]))
    return repr(out)


def script(feats, T, name='part.py'):
    """Generate the parametric script. Named parameters for holes, patterns and level heights."""
    P, B, used = [], [], set()

    def add(k, v, c=''):
        k = k if k not in used else f'{k}_{len(used)}'
        used.add(k)
        P.append(f'{k} = {_f(v)!r}' + (f'  # {c}' if c else ''))
        return k
    first = True
    for i, f in enumerate(feats):
        op = f['op']
        cl = '' if f.get('clean', True) else ', clean=False'
        if op in ('prism', 'revolve'):
            if op == 'prism':
                ax = 'XYZ'[f['ax']]
                s0, s1 = add(f'{ax}{i}_0', f['s0'], f.get('note', '')), add(f'{ax}{i}_1', f['s1'])
                tp = f", {add(f'DRAFT{i}', f['taper'], 'casting draft deg')}" if f.get('taper') else ''
                expr = f"prism({f['ax']}, {s0}, {s1}, {_ents(f['outer'])}, {[eval(_ents(e)) for e in f.get('inner', [])]!r}{tp})"
            else:
                expr = f"revolve({_ents(f['profile'])})"
            mode = f.get('mode', 'add')
            if first:
                B.append(f'result = {expr}')
                first = False
            else:
                B.append(f"result = result.{'union' if mode == 'add' else 'cut'}({expr}{cl})")
        elif op == 'gear':
            n = f'Z{i}'
            P.append(f"{n} = {int(f['n'])}  # {f.get('note', 'teeth')}")
            s0, s1 = add(f'Z{i}_0', f['s0']), add(f'Z{i}_1', f['s1'])
            B.append(f"TOOTH{i} = {[_f(v) for v in f['phi']]}, {[_f(v) for v in f['rho']]}  # one tooth: angle rad, radius")
            expr = f"gear({n}, *TOOTH{i}, {s0}, {s1}, {_f(f['r_in'])}, {[_f(v) for v in f['c']]})"
            B.append(f'result = {expr}' if first else f'result = result.union({expr}{cl})')
            first = False
        elif op == 'hole':
            d = add(f.get('name', f'D{i}'), f['d'], f.get('note', 'hole'))
            B.append(f"result = result.cut(hole({d}, {[_f(v) for v in f['p']]}, {[_f(v) for v in f['ax']]}, {_f(f['h'])}){cl})")
        elif op == 'holes':  # polar pattern
            d = add(f.get('name', f'D{i}'), f['d'], f.get('note', 'pattern hole'))
            n = f'N{i}'
            P.append(f"{n} = {int(f['n'])}  # count")
            pcd = add(f'PCD{i}', f['pcd'], 'pitch circle dia')
            a0 = add(f'A{i}', f['a0'], 'start angle deg')
            c = f['c']
            B.append(f"for i in range({n}):\n    result = result.cut(hole({d}, polar_pt({f['k']}, {_f(c[0])}, {_f(c[1])}, "
                     f"{_f(f['z0'])}, {pcd} / 2, {a0} + 360 * i / {n}), {f['ax']}, {_f(f['h'])}){cl})")
    from scipy.spatial.transform import Rotation
    Ti = np.linalg.inv(T)
    rv = Rotation.from_matrix(Ti[:3, :3]).as_rotvec()
    ang = float(np.linalg.norm(rv))
    ax = tuple(float(v) for v in (rv / ang if ang > 1e-12 else [0, 0, 1]))
    s = HEAD.format(helpers=HELPERS, name=name, params='\n'.join(P), body='\n'.join(B),
                    t=tuple(float(v) for v in Ti[:3, 3]), ax=ax, ang=float(np.degrees(ang)))
    return s.replace('import cadquery as cq', 'import math\nimport cadquery as cq')


def validate(feats, log=print):
    """Build every solid feature alone; drop the ones OCC cannot build (degenerate sketches)."""
    g = {}
    exec('import math\nimport cadquery as cq\n' + HELPERS, g)
    def solid(f):
        try:
            if f['op'] == 'prism':
                sol = g['prism'](f['ax'], f['s0'], f['s1'], eval(_ents(f['outer'])), [eval(_ents(e)) for e in f.get('inner', [])],
                                 f.get('taper', 0.0))
            else:
                sol = g['revolve'](eval(_ents(f['profile'])))
            v = sol.val()
            return v.isValid() and v.Volume() > 1e-6
        except Exception:
            return False
    ok = []
    for f in feats:
        if f['op'] not in ('prism', 'revolve') or solid(f):
            ok.append(f)
            continue
        f = _polyline_fallback(f)  # arcs that OCC rejects -> simplified straight-line sketch
        if f and solid(f):
            ok.append(f)
        else:
            log(f"  dropped unbuildable feature ({f and f.get('note', '')})")
    return ok


def _apply(res, f, g, check=False):
    """One feature of the tree applied with the script helpers (same semantics as the generated script)."""
    op = f['op']
    if op == 'prism':
        s = g['prism'](f['ax'], f['s0'], f['s1'], eval(_ents(f['outer'])), [eval(_ents(e)) for e in f.get('inner', [])],
                       f.get('taper', 0.0))
    elif op == 'revolve':
        s = g['revolve'](eval(_ents(f['profile'])))
    elif op == 'gear':
        s = g['gear'](f['n'], f['phi'], f['rho'], f['s0'], f['s1'], f['r_in'], f['c'])
    else:
        from .residual import hole_points
        for p in hole_points(f):
            res = res.cut(g['hole'](f['d'], p, f['ax'], f['h']), clean=f.get('clean', True))
        return res
    if res is None:
        return s
    cl = f.get('clean', True)
    out = res.union(s, clean=cl) if f.get('mode', 'add') == 'add' else res.cut(s, clean=cl)
    if check:  # silent kernel failures: volume must behave like a union / cut
        v0, v1, vt = res.val().Volume(), out.val().Volume(), s.val().Volume()
        ok = v1 >= v0 - 1e-4 * v0 if f.get('mode', 'add') == 'add' else v0 - vt - 1e-3 * v0 <= v1 <= v0 + 1e-4 * v0
        if not ok or not out.val().isValid():
            raise RuntimeError('boolean failed')
    return out


def safe_sequence(feats, log=print, mem_gb=6, timeout=300):
    """Run the boolean sequence in a fresh, memory/time-limited interpreter; a feature that crashes or hangs
    the OCC kernel is dropped and the sequence retried. Returns the surviving features."""
    import pickle
    import resource
    import subprocess
    import sys
    import tempfile
    lim = lambda: resource.setrlimit(resource.RLIMIT_AS, (mem_gb << 30, mem_gb << 30))
    bad = set()
    with tempfile.NamedTemporaryFile(suffix='.pkl') as tf:
        pickle.dump(feats, tf)
        tf.flush()
        while True:
            try:
                p = subprocess.run([sys.executable, '-m', 'meshrev.cad', tf.name, *map(str, bad)], capture_output=True,
                                   text=True, timeout=timeout, preexec_fn=lim)
                done, ok = {int(x) for x in p.stdout.split()}, p.returncode == 0
            except subprocess.TimeoutExpired as e:
                done, ok = {int(x) for x in (e.stdout or b'').decode().split()}, False
            nxt = None if ok else next((i for i in range(len(feats)) if i not in bad and i not in done), None)
            if nxt is None:
                return [f for i, f in enumerate(feats) if i not in bad]
            if feats[nxt].get('clean', True) and feats[nxt]['op'] != 'revolve':
                feats[nxt]['clean'] = False  # retry without face merging (UnifySameDomain can blow up)
            else:
                log(f"  dropped feature #{nxt} ({feats[nxt]['op']} {feats[nxt].get('note', '')}): CAD kernel failure")
                bad.add(nxt)
            tf.seek(0)
            tf.truncate()
            pickle.dump(feats, tf)
            tf.flush()


if __name__ == '__main__':  # child of safe_sequence: python -m meshrev.cad feats.pkl [skip indices...]
    import pickle
    import sys
    todo, skip = pickle.load(open(sys.argv[1], 'rb')), {int(x) for x in sys.argv[2:]}
    env = {}
    exec('import math\nimport cadquery as cq\n' + HELPERS, env)
    solid = None
    for n_, feat in enumerate(todo):
        if n_ not in skip:
            solid = _apply(solid, feat, env, check=True)
            print(n_, flush=True)


def _polyline_fallback(f, tol=0.05):
    from .slicer import ents_polygon
    key = 'outer' if f['op'] == 'prism' else 'profile'
    poly = ents_polygon(f[key])
    for e in f.get('inner', []):
        poly = poly.difference(ents_polygon(e))
    poly = poly.buffer(0).simplify(tol)
    if poly.is_empty:
        return None
    poly = max(getattr(poly, 'geoms', [poly]), key=lambda p: p.area)
    ring = lambda r: [('L', np.array(a), np.array(b)) for a, b in zip(r.coords[:-1], r.coords[1:])]
    f = dict(f, **{key: ring(poly.exterior)})
    if f['op'] == 'prism':
        f['inner'] = [ring(r) for r in poly.interiors]
    return f


def build(src):
    """Execute script -> cq.Shape (in original scan placement)."""
    g = {'__name__': 'meshrev_build'}
    exec(compile(src, 'part', 'exec'), g)
    return g['result']


def export(shape, path):
    import cadquery as cq
    cq.exporters.export(cq.Workplane().add(shape), str(path))


def to_mesh(shape, tol=0.02):
    import trimesh
    v, f = shape.tessellate(tol, 0.2)
    return trimesh.Trimesh(np.array([p.toTuple() for p in v]), np.array(f))
