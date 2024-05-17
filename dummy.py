from pyiga import bspline, geometry, assemble
from pyiga import bspline, assemble, vform, geometry, vis, solvers
kv = bspline.make_knots(3, 0.0, 1.0, 50)    # knot vector over (0,1) with degree 3 and 50 knot spans
geo = geometry.quarter_annulus()            # a NURBS representation of a quarter annulus
K = assemble.stiffness((kv,kv), geo=geo)    # assemble a stiffness matrix for the 2D tensor product
                                            # B-spline basis over the quarter annulus
import scipy
from pyiga import bspline, assemble, vform, geometry, vis, solvers
import numpy as np
import matplotlib.pyplot as plt
n = 2
f = geometry.circular_arc(pi/n)
g = geometry.line_segment([0,0], [1,1])
G = geometry.outer_product(f,g) # create pie slice
# vis.plot_geo(G)
G1 = G.rotate_2d(pi*0.25) # rotate pie slice to upright position
# vis.plot_geo(G1)

G2 = G1.scale([0.726,1]) # stretch pie slice in x & y-dir
# vis.plot_geo(G2)

geos = [
    G2,
    G2.rotate_2d(pi*.4),
    G2.rotate_2d(2*pi*.4),
    G2.rotate_2d(3*pi*.4),
    G2.rotate_2d(4*pi*.4),
]

for g in geos:
    vis.plot_geo(g)
    
axis('equal')
axis('off')