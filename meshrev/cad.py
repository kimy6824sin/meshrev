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


def prism(ax, s0, s1, outer, inner=()):
    """Extrude a sketch along axis ax (0=X,1=Y,2=Z) from s0 to s1."""
    xd, nd = AX[ax]
    org = [0, 0, 0]
    org[ax] = s0
    pl = cq.Plane(origin=tuple(org), xDir=xd, normal=nd)
    s = wire(cq.Workplane(pl), outer).extrude(s1 - s0)
    for e in inner:
        s = s.cut(wire(cq.Workplane(pl), e).extrude(s1 - s0))
    return s


def revolve(profile):
    """Revolve an (r, z) profile 360 deg about Z."""
    pl = cq.Plane(origin=(0, 0, 0), xDir=(1, 0, 0), normal=(0, -1, 0))
    return wire(cq.Workplane(pl), profile).revolve(360, (0, 0, 0), (0, 1, 0))


def hole(d, p, ax, h):
    """Cylindrical cut tool of diameter d from point p along direction ax, length h."""
    return cq.Workplane().add(cq.Solid.makeCylinder(d / 2, h, cq.Vector(*p), cq.Vector(*ax)))


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
    cq.exporters.export(cq.Workplane().add(result), "{name}".replace(".py", ".step"))
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
    P, B = [], []
    add = lambda k, v, c='': P.append(f'{k} = {_f(v)!r}' + (f'  # {c}' if c else '')) or k
    first = True
    for i, f in enumerate(feats):
        op = f['op']
        if op in ('prism', 'revolve'):
            if op == 'prism':
                ax = 'XYZ'[f['ax']]
                s0, s1 = add(f'{ax}{i}_0', f['s0'], f.get('note', '')), add(f'{ax}{i}_1', f['s1'])
                expr = f"prism({f['ax']}, {s0}, {s1}, {_ents(f['outer'])}, {[eval(_ents(e)) for e in f.get('inner', [])]!r})"
            else:
                expr = f"revolve({_ents(f['profile'])})"
            mode = f.get('mode', 'add')
            if first:
                B.append(f'result = {expr}')
                first = False
            else:
                B.append(f"result = result.{'union' if mode == 'add' else 'cut'}({expr})")
        elif op == 'hole':
            d = add(f'D{i}', f['d'], f.get('note', 'hole'))
            B.append(f"result = result.cut(hole({d}, {[_f(v) for v in f['p']]}, {[_f(v) for v in f['ax']]}, {_f(f['h'])}))")
        elif op == 'holes':  # polar pattern
            d = add(f'D{i}', f['d'], f.get('note', 'pattern hole'))
            n = f'N{i}'
            P.append(f"{n} = {int(f['n'])}  # count")
            pcd = add(f'PCD{i}', f['pcd'], 'pitch circle dia')
            a0 = add(f'A{i}', f['a0'], 'start angle deg')
            c = f['c']
            B.append(f"for i in range({n}):\n    result = result.cut(hole({d}, polar_pt({f['k']}, {_f(c[0])}, {_f(c[1])}, "
                     f"{_f(f['z0'])}, {pcd} / 2, {a0} + 360 * i / {n}), {f['ax']}, {_f(f['h'])}))")
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
    ok = []
    for f in feats:
        try:
            if f['op'] == 'prism':
                sol = g['prism'](f['ax'], f['s0'], f['s1'], eval(_ents(f['outer'])), [eval(_ents(e)) for e in f.get('inner', [])])
            elif f['op'] == 'revolve':
                sol = g['revolve'](eval(_ents(f['profile'])))
            else:
                ok.append(f)
                continue
            v = sol.val()
            if v.isValid() and v.Volume() > 1e-6:
                ok.append(f)
                continue
        except Exception:
            pass
        log(f"  dropped unbuildable {f['op']} feature ({f.get('note', '')})")
    return ok


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
