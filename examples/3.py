"""Parametric model reconstructed by meshrev. Edit the PARAMETERS and re-run: python 3.py"""
import math
import cadquery as cq

# ---------------- PARAMETERS (mm) ----------------
Z0_0 = 0.05
Z0_1 = 9.95
Z1_0 = 9.95
Z1_1 = 20.0
D2 = 27.0  # 孔 (实测Ø27.030)
D3 = 6.2  # 孔 (实测Ø6.223)
N3 = 2  # count
PCD3 = 46.0  # pitch circle dia
A3 = 63.85  # start angle deg
D4 = 5.0  # M6 螺纹孔(建模为底孔Ø5.0) (实测Ø5.657)
N4 = 2  # count
PCD4 = 46.0  # pitch circle dia
A4 = 133.95  # start angle deg

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
result = prism(2, Z0_0, Z0_1, [('A', -10.8989, -26.0718, -14.2027, -24.2804, -15.5785, -21.5956), ('L', -15.5785, -21.5956, -20.582, -4.271), ('L', -20.582, -4.271, -20.9624, -0.8125), ('A', -20.9624, -0.8125, -16.7849, 12.5174, -5.789, 20.2389), ('L', -5.789, 20.2389, 8.4761, 25.8115), ('A', 8.4761, 25.8115, 13.4336, 24.9437, 15.4661, 21.8744), ('L', 15.4661, 21.8744, 20.5572, 3.8884), ('A', 20.5572, 3.8884, 17.789, -11.0908, 5.5932, -20.3651), ('L', 5.5932, -20.3651, -9.0224, -26.0718), ('L', -9.0224, -26.0718, -10.8989, -26.0718)], [])
result = result.union(prism(2, Z1_0, Z1_1, [('L', -4.5184, -20.4927, -5.0435, -18.5811), ('A', -5.0435, -18.5811, -8.8193, -15.3548, -13.3672, -16.1948), ('A', -13.3672, -16.1948, -19.4695, -7.7647, -20.9289, 2.5416), ('L', -20.9289, 2.5416, -20.9289, 16.7887), ('L', -20.9289, 16.7887, -20.3759, 18.811), ('A', -20.3759, 18.811, -17.9421, 21.0465, -14.8136, 21.5112), ('L', -14.8136, 21.5112, 2.6152, 20.8507), ('L', 2.6152, 20.8507, 4.6316, 20.407), ('A', 4.6316, 20.407, 7.7488, 15.7044, 13.3631, 16.1467), ('A', 13.3631, 16.1467, 19.3258, 8.0881, 20.9417, -1.7937), ('L', 20.9417, -1.7937, 20.9417, -16.9542), ('A', 20.9417, -16.9542, 18.5856, -20.7772, 14.6408, -21.5446), ('L', 14.6408, -21.5446, -2.4087, -20.9013), ('L', -2.4087, -20.9013, -4.5184, -20.4927)], []))
result = result.cut(hole(D2, [0.0074, -0.0169, 0.05], [0.0, 0.0, 1.0], 19.95))
for i in range(N3):
    result = result.cut(hole(D3, polar_pt(2, 0.05, 0.0, 0.05, PCD3 / 2, A3 + 360 * i / N3), [0, 0, 1], 9.9))
for i in range(N4):
    result = result.cut(hole(D4, polar_pt(2, 0.0, -0.05, 9.95, PCD4 / 2, A4 + 360 * i / N4), [0, 0, 1], 10.05))

# back to the scan's original placement
result = result.val().moved(cq.Location(cq.Vector(93.51563895893864, 74.21475603986504, 66.7665031293809), cq.Vector(-8.368278456136892e-07, 0.00033625853134317663, -0.9999999434647484), 63.865263870546215))
if __name__ == "__main__":
    cq.exporters.export(cq.Workplane().add(result), __file__.replace(".py", ".step"))
