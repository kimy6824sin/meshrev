"""End-to-end: scan mesh -> feature tree -> parametric CadQuery script + STEP + report data."""
import json
import time
from pathlib import Path
import numpy as np
from . import prep, body, residual, cad, verify, knowledge


def tolerances(sigma, diag):
    """Precision tiers: A mating/contact (machined), B general, C cast/appearance."""
    return dict(A=max(0.05, 3 * sigma), B=0.3, C=max(0.8, 0.01 * diag))


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
        gear = body.detect_gear(m)
        if gear:
            occ[np.ix_((zs >= gear['s0'] - h) & (zs <= gear['s1'] + h), rs > gear['r_root'] - h)] = 0
            gear['r_in'] = gear['r_root'] - max(1.0, 3 * h)
        feats = body.revolve(m, body.outer_shell(occ), rs, zs, h, tol['A'], sinfo['zplanes'])
        holes = []
        if gear:
            feats.append(gear)
    else:
        mode = 'stack'
        if ar is not None:  # undo recentre
            m.apply_translation([c[0], c[1], 0])
            info['T'][:2, 3] += c
            feats, holes, sinfo = body.stack(m, fn, tol['A'])
    for k in ((0, 1, 2) if mode == 'revolve' else (0, 1)):
        holes += body.axis_holes(m, k, tol['A'], exclude_axis=(mode == 'revolve' and k == 2))
    log(f"base: {mode} ({len(feats)} features, {len(holes)} holes, {len(sinfo['levels'])} levels, A_rev={ar}) {time.time() - t0:.1f}s")
    # residual features (analysis by synthesis)
    g = residual.Grid(m, float(np.clip(m.extents.max() / 160, 0.15, 1.0)))
    M = residual.mesh_voxels(m, g)
    for it in range(3):
        new = residual.residual_features(m, feats + holes, g, M, tol['C'])
        new, nh = holes_from_prisms(m, new, g.h)
        log(f"residual pass {it + 1}: +{len(new)} prisms, +{len(nh)} holes")
        if not new and not nh:
            break
        feats += [f for f in new if f['mode'] == 'add'] + [f for f in new if f['mode'] == 'cut']
        holes += nh
    feats += [h for h in holes if h['op'] == 'prism']
    holes = [h for h in holes if h['op'] != 'prism']
    feats = [knowledge.gear_intent(f, tol) if f['op'] == 'gear' and 'module' not in f else f for f in feats]
    feats, holes, notes = knowledge.regularize(feats, holes, tol)
    info.update(mode=mode, notes=notes, time=time.time() - t0, zplanes=sinfo['zplanes'])
    return m, feats + holes, info


def run(path, out, ai=False, log=print):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(path).stem
    m, feats, info = reconstruct(path, log)
    if ai:
        from . import ai as A
        feats, info = A.review(feats, info, log)
    feats = cad.safe_sequence(cad.validate(feats, log), log)
    src = cad.script(feats, info['T'], f'{name}.py')
    (out / f'{name}.py').write_text(src)
    shape = cad.build(src)
    cad.export(shape, out / f'{name}.step')
    model = cad.to_mesh(shape)
    model.apply_transform(info['T'])  # compare in the canonical frame
    pts, d = verify.deviation(m, model)
    st = verify.stats(d, info['tol']['B'])
    ts = verify.tier_stats(d, verify.tiers(m, feats, pts, info['sigma'], info['zplanes']), info['tol'])
    log(f"deviation: rms={st['rms']:.3f} p95={st['p95']:.3f} within±{info['tol']['B']}={st['within'] * 100:.1f}% | " +
        ' '.join(f"{k}({v['frac'] * 100:.0f}%): rms {v['rms']:.3f}, ±{info['tol'][k]:.2f} {v['within'] * 100:.0f}%" for k, v in ts.items()))
    summary = dict(file=str(path), mode=info['mode'], sigma=info['sigma'], tol=info['tol'],
                   deviation=st, tiers=ts, notes=info['notes'], features=knowledge.summary(feats))
    (out / f'{name}.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float))
    try:
        from . import report
        report.write(out / f'{name}.html', m, model, pts, d, summary)
    except ImportError:
        pass
    return summary
