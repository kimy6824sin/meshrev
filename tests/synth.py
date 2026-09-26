"""Synthetic benchmark: CAD parts with known parameters -> noisy scan mesh -> reconstruct -> compare.
Run: python tests/synth.py   (also used by tests/test_synth.py)"""
import sys
import tempfile
from pathlib import Path
import numpy as np
import cadquery as cq
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meshrev import cad, pipeline  # noqa: E402


def flange():
    """Plate 60x44x10 (R6 corners) + boss D32x8 + bore D20 + 4 x M6 clearance (D6.6) on PCD 46."""
    s = cq.Workplane().rect(60, 44).extrude(10).edges('|Z').fillet(6)
    s = s.union(cq.Workplane().circle(16).extrude(18)).faces('>Z').workplane().hole(20)
    s = s.faces('<Z').workplane().polarArray(23, 45, 360, 4).hole(6.6)
    return s, dict(bore=20.0, hole=6.6, n=4, pcd=46.0)


def shaft():
    """Turned stepped shaft with a D30 collar, a groove and a D5 cross hole."""
    prof = [(0, 0), (10, 0), (10, 20), (15, 20), (15, 26), (8, 26), (8, 50), (7, 50), (7, 53), (8, 53), (8, 60), (0, 60)]
    s = cq.Workplane('XZ').polyline(prof).close().revolve(360, (0, 0, 0), (0, 1, 0))
    s = s.cut(cq.Workplane('YZ').workplane(offset=-20).center(0, 40).circle(2.5).extrude(40))
    return s, dict(cross=5.0, d=[20.0, 30.0, 16.0])


def gear():
    """Spur gear z=30 m=2 (tip D64) with D20 bore, width 12."""
    z, m = 30, 2.0
    ra, rf = m * (z + 2) / 2, m * (z - 2.5) / 2
    pts = []
    for i in range(z):
        a = 2 * np.pi * i / z
        for da, r in ((-0.35, rf), (-0.18, ra), (0.18, ra), (0.35, rf)):
            pts.append((r * np.cos(a + da * 2 * np.pi / z), r * np.sin(a + da * 2 * np.pi / z)))
    s = cq.Workplane().polyline(pts).close().extrude(12).faces('>Z').workplane().hole(20)
    return s, dict(z=z, bore=20.0)


def scan(shape, sigma=0.03, spikes=0.002, seed=0):
    """Tessellate, subdivide, add Gaussian noise + spike outliers, random pose."""
    rng = np.random.default_rng(seed)
    m = cad.to_mesh(shape.val(), 0.01)
    m.merge_vertices()
    m = m.subdivide_to_size(1.0)
    v = m.vertices + rng.normal(0, sigma, m.vertices.shape)
    k = rng.random(len(v)) < spikes
    v[k] += rng.normal(0, 10 * sigma, (k.sum(), 3))
    m.vertices = v
    R = trimesh.transformations.random_rotation_matrix(rng.random(3))
    R[:3, 3] = rng.uniform(-50, 50, 3)
    m.apply_transform(R)
    return m


def run(name, sigma=0.03):
    shape, truth = globals()[name]()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f'{name}.stl'
        scan(shape, sigma).export(p)
        s = pipeline.run(p, d, log=lambda *a: None)
    return s, truth


if __name__ == '__main__':
    for n in ('flange', 'shaft', 'gear'):
        s, t = run(n)
        print(n, s['mode'], 'rms %.3f' % s['deviation']['rms'], 'truth', t)
        for f in s['features']:
            if f['op'] in ('hole', 'holes', 'gear'):
                print('  -', f)
