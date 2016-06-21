import matplotlib as mpl; mpl.rcParams['savefig.dpi'] = 144
import matplotlib.pyplot as plt
import numpy as np
import pymses
import math
from scipy.optimize import curve_fit



#curve fitting question

def line(r, a, r_d):
    return a*np.exp(-r/r_d)



    
#Radial data creation
    
def radial_data(data,annulus_width=1,working_mask=None,x=None,y=None,rmax=None):
    """
    r = radial_data(data,annulus_width,working_mask,x,y)
    
    A function to reduce an image to a radial cross-section.
    
    INPUT:
    ------
    data   - whatever data you are radially averaging.  Data is
            binned into a series of annuli of width 'annulus_width'
            pixels.
    annulus_width - width of each annulus.  Default is 1.
    working_mask - array of same size as 'data', with zeros at
                      whichever 'data' points you don't want included
                      in the radial data computations.
      x,y - coordinate system in which the data exists (used to set
             the center of the data).  By default, these are set to
             integer meshgrids
      rmax -- maximum radial value over which to compute statistics
    
     OUTPUT:
     -------
      r - a data structure containing the following
                   statistics, computed across each annulus:
          .r      - the radial coordinate used (outer edge of annulus)
          .mean   - mean of the data in the annulus
          .std    - standard deviation of the data in the annulus
          .median - median value in the annulus
          .max    - maximum value in the annulus
          .min    - minimum value in the annulus
          .numel  - number of elements in the annulus
    """
    
# 2010-03-10 19:22 IJC: Ported to python from Matlab
# 2005/12/19 Added 'working_region' option (IJC)
# 2005/12/15 Switched order of outputs (IJC)
# 2005/12/12 IJC: Removed decifact, changed name, wrote comments.
# 2005/11/04 by Ian Crossfield at the Jet Propulsion Laboratory
 
    import numpy as ny

    class radialDat:
        """Empty object container.
        """
        def __init__(self): 
            self.mean = None
            self.std = None
            self.median = None
            self.numel = None
            self.max = None
            self.min = None
            self.r = None

    #---------------------
    # Set up input parameters
    #---------------------
    data = ny.array(data)
    
    if working_mask==None:
        working_mask = ny.ones(data.shape,bool)
    
    npix, npiy = data.shape
    if x==None or y==None:
        x1 = ny.arange(-npix/2.,npix/2.)
        y1 = ny.arange(-npiy/2.,npiy/2.)
        x,y = ny.meshgrid(y1,x1)

    r = abs(x+1j*y)
    

    working_mask = ny.ones(data.shape,bool)
    



    if rmax==None:
        rmax = r[working_mask].max()

    #---------------------
    # Prepare the data container
    #---------------------
    dr = ny.abs([x[0,0] - x[0,1]]) * annulus_width
    radial = ny.arange(rmax/dr)*dr + dr/2.
    nrad = len(radial)
    radialdata = radialDat()
    radialdata.mean = ny.zeros(nrad)
    radialdata.std = ny.zeros(nrad)
    radialdata.median = ny.zeros(nrad)
    radialdata.numel = ny.zeros(nrad)
    radialdata.max = ny.zeros(nrad)
    radialdata.min = ny.zeros(nrad)
    radialdata.r = radial
    
    #---------------------
    # Loop through the bins
    #---------------------
    for irad in range(nrad): #= 1:numel(radial)
      minrad = irad*dr
      maxrad = minrad + dr
      thisindex = (r>=minrad) * (r<maxrad) * working_mask
      if not thisindex.ravel().any():
        radialdata.mean[irad] = ny.nan
        radialdata.std[irad]  = ny.nan
        radialdata.median[irad] = ny.nan
        radialdata.numel[irad] = ny.nan
        radialdata.max[irad] = ny.nan
        radialdata.min[irad] = ny.nan
      else:
        radialdata.mean[irad] = data[thisindex].mean()
        radialdata.std[irad]  = data[thisindex].std()
        radialdata.median[irad] = ny.median(data[thisindex])
        radialdata.numel[irad] = data[thisindex].size
        radialdata.max[irad] = data[thisindex].max()
        radialdata.min[irad] = data[thisindex].min()
    
    #---------------------
    # Return with data
    #---------------------
    
    return radialdata
    
    

N = 164

# Generate Sampling Points. Generates 64 numbers in the range (0,1) along each dimension.
x, y, z = np.mgrid[0:1:complex(0,N), 0:1:complex(0,N), 0:1:complex(0,N)]

# Reshape
npoints = np.prod(x.shape)
x1 = np.reshape(x, npoints)
y1 = np.reshape(y, npoints)
z1 = np.reshape(z, npoints)

# Arrange for Pymses
pxyz = np.array([x1, y1, z1])
pxyz = pxyz.transpose()


# Prepare Ramses Output
basedir = "/home/ankit/ramses/trunk/old_run/17May2016/lambda0.1"
output = pymses.RamsesOutput("%s" % basedir, 4 , verbose=False)
source = output.amr_source(["rho", "vel", "P"])

# Sample Hydro Fields
dset = pymses.analysis.sample_points(source, pxyz, use_C_code=True)


#Pymses returns all data in a flat list. We must reshape this again for further use.
rho = np.reshape(dset["rho"], (N,N,N))
vx = np.reshape(dset["vel"][:,0], (N,N,N))
vy = np.reshape(dset["vel"][:,1], (N,N,N))
vz = np.reshape(dset["vel"][:,2], (N,N,N))



dz = 1.0/N
Sigma = np.sum(rho*dz, axis=2)



# Plot Surface Density
fig= plt.subplot(1,2,1)
ax=plt.gca()
im = ax.imshow(np.log10(Sigma), interpolation='none', cmap="bone")
plt.colorbar(im)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Surface Density")


#calculating the annular average of the disk

Ddata = radial_data(Sigma,1,x,y)


#calling curvefit

popt, pcov = curve_fit(line,Ddata.r, Ddata.mean)

print popt



#print radialprofile.size

fig = plt.subplot(1,2,2)
plt.plot(Ddata.r,Ddata.mean)

fp= open("my2.txt","w")
np.savetxt(fp, np.c_[Ddata.r,Ddata.mean])


fp.close()
