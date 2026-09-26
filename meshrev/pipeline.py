"""End-to-end: scan mesh -> feature tree -> parametric CadQuery script + STEP + report data."""
import json
import time
from pathlib import Path
import numpy as np
from . import prep, body, residual, cad, verify, knowledge


def tolerances(sigma, diag):
    """Precision tiers: A mating/contact (machined), B general, C cast/appearance."""
    return dict(A=max(0.05, 3 * sigma), B=0.3, C=max(0.5, 0.004 * diag))


def a_rev(m, fn):
    p, w = m.triangles_center, m.area_faces
    rho = np.hypot(p[:, 0], p[:, 1]) + 1e-9
    nt = np.abs(fn[:, 0] * -p[:, 1] / rho + fn[:, 1] * p[:, 0] / rho)
    side = np.abs(fn[:, 2]) < 0.95
    return float(w[side & (nt < 0.087)].sum() / w[side].sum())


def holes_from_prisms(m, feats, h):
    """Circular inner loops / circular cut prisms -> precise hole features (refined on the mesh)."""
    keep, holes = [], []
    for f in feats:
        if f['op'] != 'prism' or f.get('tier') != 'C':
            keep.append(f)
            continue
        k, circ = f['ax'], []
        if f['mode'] == 'cut' and f['outer'][0][0] == 'C':
            circ.append(f['outer'][0])
        else:
            circ += [e[0] for e in f['inner'] if e[0][0] == 'C']
            f['inner'] = [e for e in f['inner'] if e[0][0] != 'C']
            keep.append(f)
        for _, cu, cv, r in circ:
            p = [0.0, 0.0, 0.0]
            p[(k + 1) % 3], p[(k + 2) % 3], p[k] = cu, cv, f['s0'] - h
            rh = body.refine_hole(m, p, k, r, f['s1'] - f['s0'] + 2 * h)
            if rh is None or rh[2] < 0.5 * rh[1]:
                continue
            p, d, hh, tilt, rough = rh
            ax = [0, 0, 0]
            ax[k] = 1
            holes.append(dict(op='hole', p=p, ax=ax, d=d, h=hh, tilt=tilt, k=k, rough=rough))
    return keep, holes


def reconstruct(path, log=print):
    t0 = time.time()
    m, fn, info = prep.prepare(path)
    tol = tolerances(info['sigma'], float(np.linalg.norm(m.extents)))
    info['tol'] = tol
    log(f"loaded {info['n_faces']} faces, sigma={info['sigma']:.3f} mm, extents={np.round(m.extents, 2).tolist()}")
    # rotational candidate: recentre on the revolve axis
    c = body.axis_center(m, fn)
    square = abs(m.extents[0] - m.extents[1]) / max(m.extents[:2]) < 0.05
    feats, holes, sinfo = body.stack(m, fn, tol['A'])
    ar = None
    if square and len(sinfo['levels']) > 10:
        m.apply_translation([-c[0], -c[1], 0])
        info['T'][:2, 3] -= c
        ar = a_rev(m, fn)
    if ar is not None and ar > 0.45:
        mode = 'revolve'
        occ, rs, zs, h = body.rz_profile(m)
        feats = body.revolve(m, occ, rs, zs, h, tol['A'], sinfo['zplanes'])
        holes = []
    else:
        mode = 'stack'
        if ar is not None:  # undo recentre
            m.apply_translation([c[0], c[1], 0])
            info['T'][:2, 3] += c
            feats, holes, sinfo = body.stack(m, fn, tol['A'])
    log(f"base: {mode} ({len(feats)} features, {len(sinfo['levels'])} levels, A_rev={ar}) {time.time() - t0:.1f}s")
    # residual features (analysis by synthesis)
    g = residual.Grid(m, float(np.clip(m.extents.max() / 160, 0.15, 1.0)))
    M = residual.mesh_voxels(m, g)
    for it in range(2):
        new = residual.residual_features(m, feats + holes, g, M, tol['C'])
        new, nh = holes_from_prisms(m, new, g.h)
        log(f"residual pass {it + 1}: +{len(new)} prisms, +{len(nh)} holes")
        if not new and not nh:
            break
        feats += [f for f in new if f['mode'] == 'add'] + [f for f in new if f['mode'] == 'cut']
        holes += nh
    feats = [f for f in feats if f.get('mode', 'add') == 'add'] + [f for f in feats if f.get('mode') == 'cut']
    feats += [h for h in holes if h['op'] == 'prism']
    holes = [h for h in holes if h['op'] != 'prism']
    feats = [f for f in feats if f.get('mode', 'add') == 'add'] + [f for f in feats if f.get('mode') == 'cut']
    feats, holes, notes = knowledge.regularize(feats, holes, tol)
    info.update(mode=mode, notes=notes, time=time.time() - t0)
    return m, feats + holes, info


def run(path, out, ai=False, log=print):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(path).stem
    m, feats, info = reconstruct(path, log)
    if ai:
        from . import ai as A
        feats, info = A.review(feats, info, log)
    feats = cad.validate(feats, log)
    src = cad.script(feats, info['T'], f'{name}.py')
    (out / f'{name}.py').write_text(src)
    shape = cad.build(src)
    cad.export(shape, out / f'{name}.step')
    import trimesh
    scan = trimesh.load(path, force='mesh')
    model = cad.to_mesh(shape)
    pts, d = verify.deviation(scan, model)
    st = verify.stats(d, info['tol']['B'])
    log(f"deviation: rms={st['rms']:.3f} p95={st['p95']:.3f} within±{info['tol']['B']}={st['within'] * 100:.1f}%")
    info['deviation'] = st
    summary = dict(file=str(path), mode=info['mode'], sigma=info['sigma'], tol=info['tol'],
                   deviation=st, notes=info['notes'], features=knowledge.summary(feats))
    (out / f'{name}.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float))
    try:
        from . import report
        report.write(out / f'{name}.html', scan, model, pts, d, summary)
    except ImportError:
        pass
    return summary
