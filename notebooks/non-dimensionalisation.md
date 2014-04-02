Non-dimensionalisation
======================

We have performed experiments over a range of physical parameters.
Combining these experiments in an ensemble is possible provided that
the data is non-dimensionalised.

Non-dimensionalising the data consists of two steps:

1. Dividing through by length / time / velocity scales.

2. Re-sampling to a regular grid that is the same for all runs.

```python
%matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

import gc_turbulence as g

r = g.ProcessedRun(cache_path=g.default_processed + 'r13_12_16a.hdf5')
```


### Scales

There are two length scales - horizontal and vertical. The
horizontal length scale is defined as the lock length $L$. The
vertical length scale is defined as the fluid depth $H$.

There velocity scale is $U = \sqrt{g' H}$, where the reduced gravity
$g'$ is

$$
g' = g \frac{(\rho_{lock} - \rho_{ambient})}{\rho_{ambient}}
$$

We can define a time scale $T$ as $H / U$.

```python
p = r.attributes

L = p['L']
H = p['H']
g_ = 9.81 * (p['rho_lock'] - p['rho_ambient']) / p['rho_ambient']
U = (g_ * H) ** .5
T = H / U

# underscore indicates non-dimensional
X_ = r.Xf / L
Z_ = r.Zf / H
T_ = r.Tf / T

U_ = r.Uf / U
V_ = r.Vf / U
W_ = r.Wf / U
```

Be careful. There are experiments with varying $L$. It may appear
that we can compare these runs non-dimensionally but this is not
neccesarily true as they capture different phases in the gravity
current lifetime.

### Re-sampling

Scaling the dimensional quantities also scales their spacing on the
measurement grid, i.e. as well as scaling $X, Z, T$ we also scale
$dx, dz, dt$. This means that runs with different parameters are
sampled differently in non-dimensional space.

To completely compare runs we have to re-sample to a regular grid in
non-dimensional space. 

Our dimensional sampling intervals are: 

- time: 0.01 seconds
- vertical: 0.001158 metres
- horizontal: 0.001439 metres

We need to choose non-dimensional sampling intervals as these need
to be the same across all runs.

It doesn't make sense to sample any run to a higher resolution than
it was measured at.

We also need to choose the limits of our non-dimensional grid, which
can't extend beyond the edges of the dimensional grid.

We'll work here with the idea that we are looking at full depth runs
with $H=0.25$ and $L=0.25$.

Set the sampling intervals by doubling the non dim interval of the
fastest run.

```python
dx_ = 0.01
dz_ = 0.012
dt_ = 0.015
```

The actual resampling is done with map coordinates:

```python

# this is where the top and bottom of the valid z region are
z0 = 0.007 / H
z1 = 0.12 / H
z_coords = np.arange(z0, z1, dz_)

# valid x region
x0 = 3.18 / L
x1 = 3.32 / L
x_coords = np.arange(x0, x1, dx_)

# time where stuff happens relative to the front
t0 = -4.99 / T
t1 = 20 / T
t_coords = np.arange(t0, t1, dt_)

# work out the grid coordinates to index the nondim arrays with
gz = (z_coords - Z_[:, 0, 0][0]) * (H / dz)
gx = (x_coords - X_[0, :, 0][0]) * (L / dx)
gt = (t_coords - T_[0, 0, :][0]) * (T / dt)
## or, we can just define the grid coordinates. but this relies on
## us knowing what the limits of the non dim space are??
## or will this actually capture everything?

coords = np.concatenate([c[None] for c in np.meshgrid(gz, gx, gt,
                                                      indexing='ij')], axis=0)

xs = ndi.map_coordinates(X_, coords, order=1, cval=np.nan)
zs = ndi.map_coordinates(Z_, coords, order=1, cval=np.nan)
ts = ndi.map_coordinates(T_, coords, order=1, cval=np.nan)

ufs = ndi.map_coordinates(U_, coords, order=3, cval=np.nan)
vfs = ndi.map_coordinates(V_, coords, order=3, cval=np.nan)
wfs = ndi.map_coordinates(W_, coords, order=3, cval=np.nan)

```
