"""Parametric model reconstructed by meshrev. Edit the PARAMETERS and re-run: python 5.py"""
import math
import cadquery as cq

# ---------------- PARAMETERS (mm) ----------------
Z1 = 98  # 齿轮/齿圈 z=98，模数≈1.787
Z1_0 = 21.0752
Z1_1 = 28.4481
D2 = 10.0  # M8 通孔(粗) (实测Ø10.075)
N2 = 6  # count
PCD2 = 75.0  # pitch circle dia
A2 = 35.86  # start angle deg
D3 = 6.65  # 孔 (实测Ø6.656)
N3 = 6  # count
PCD3 = 75.0  # pitch circle dia
A3 = 5.85  # start angle deg

# ---------------- helpers ----------------
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


# ---------------- FEATURE TREE ----------------
result = revolve([('L', 85.0574, 28.3936, 85.0574, 21.3328), ('L', 85.0574, 21.3328, 77.2917, 20.8008), ('L', 77.2917, 20.8008, 57.7774, 1.3031), ('L', 57.7774, 1.3031, 55.1894, 1.0193), ('L', 55.1894, 1.0193, 47.845, 1.0193), ('A', 47.845, 1.0193, 45.9474, -0.1227, 44.5827, 0.0238), ('L', 44.5827, 0.0238, 19.9604, 0.0442), ('L', 19.9604, 0.0442, 20.0735, 3.5206), ('L', 20.0735, 3.5206, 56.4476, 3.5206), ('L', 56.4476, 3.5206, 77.3587, 24.386), ('L', 77.3587, 24.386, 81.0421, 24.5504), ('A', 81.0421, 24.5504, 82.119, 26.3191, 81.7327, 28.2401), ('L', 81.7327, 28.2401, 85.0574, 28.3936)])
TOOTH1 = [-0.008, -0.0048, -0.0016, 0.0016, 0.0048, 0.008, 0.0112, 0.0144, 0.0176, 0.0208, 0.024, 0.0272, 0.0305, 0.0337, 0.0369, 0.0401, 0.0433, 0.0465, 0.0497, 0.0529], [84.983, 85.0009, 85.1009, 85.3178, 85.8303, 86.7766, 87.6994, 88.5323, 89.251, 89.329, 89.3301, 89.33, 89.3124, 89.013, 88.1224, 87.1222, 86.1658, 85.4709, 85.1636, 85.0256]  # one tooth: angle rad, radius
result = result.union(gear(Z1, *TOOTH1, Z1_0, Z1_1, 83.983, [0.0, 0.0]))
for i in range(N2):
    result = result.cut(hole(D2, polar_pt(2, 0.0, 0.0, 0.05, PCD2 / 2, A2 + 360 * i / N2), [0, 0, 1], 3.45))
for i in range(N3):
    result = result.cut(hole(D3, polar_pt(2, 0.0, 0.0, 0.05, PCD3 / 2, A3 + 360 * i / N3), [0, 0, 1], 3.5))

# back to the scan's original placement
result = result.val().moved(cq.Location(cq.Vector(-136.25805276388564, 0.009777081724220997, 0.013966500476500296), cq.Vector(0.6429409422202117, 0.41648350763599945, 0.6427817924336383), 134.77748730270724))
if __name__ == "__main__":
    cq.exporters.export(cq.Workplane().add(result), __file__.replace(".py", ".step"))
