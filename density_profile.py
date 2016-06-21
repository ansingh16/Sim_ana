from matplotlib import pylab as plt
from pymses.utils import constants as C
import numpy as np

gal_center = [ 0.461293 , 0.501006, 0.519529 ]

gal_radius = 0.318608

gal_thickn = 0.05

gal_normal = [ 0.000094, 0.011879 , 0.999929]


from pymses import RamsesOutput
output = RamsesOutput("/home/ankit/ramses/trunk/old_run/17May2016/lambda0.1",4)

source = output.amr_source(["rho"])


from pymses.utils.regions import Cylinder
cyl = Cylinder(gal_center, gal_normal, gal_radius, gal_thickn)

from pymses.analysis import sample_points
points = cyl.random_points(1.0e4) # 1M sampling points
point_dset = sample_points(source, points)

import numpy
rho_weight_func = lambda dset: dset["rho"]
r_bins = numpy.linspace(0.0, gal_radius, 200)


from pymses.analysis import bin_cylindrical
rho_profile = bin_cylindrical(point_dset, gal_center, gal_normal,rho_weight_func, r_bins, divide_by_counts=True)


 
m=np.linspace(0.0,gal_radius, num=199)*output.info["unit_length"].express(C.kpc)

"""print rho_profile.shape
print m.shape
print rho_profile
print m
"""
plt.plot(m,rho_profile)

