Interpolation of missing values
===============================

The data computed by Dynamic Studio contains some missing values,
represented by the velocity being identically zero. We would like to
fill in these gaps in the data.

First of all, *do we actually need to do this?* We can probably get
away with not interpolating the regular data, but we have problems
when we make the front relative transform. The front relative
transform is implemented with `scipy.ndimage.map_coordinates` which
doesn't like `nan` values. If you apply `map_coordinates` to data
that hasn't had nans removed, you end up with nans everywhere.

As the front relative transform is a fundamental technique in this
work, allowing us to do analysis on a stationary gravity current, it
is essential that we find a way to deal with this issue.

Let's get an idea of the scale of the problem:

```python
import numpy as np
import gc_turbulence as g

r = g.SingleLayerRun(cache_path=g.default_cache + 'r13_12_17c.hdf5')
r.load()
pp = g.PreProcessor(r)
pp.extract_valid_region()

velocities = (pp.U, pp.V, pp.W)

zeros = [np.where(d[:] == 0) for d in velocities]

missing = [z[0].size for z in zeros]

total_size = [d[:].size for d in velocities]

proportion = [float(m) / t for m, t in zip(missing, total_size)]

print "Proportion of missing values for each velocity component:"
print proportion
```
Proportion of missing values for each velocity component:
[0.0019078614229048865, 0.0019078614229048865, 0.0019078614229048865]


0.2% - that's pretty small, but bear in mind here that this is
across the entireity of the data. The invalid values are largely
present when the front is passing through.

```python
pp.transform_to_lock_relative()
pp.transform_to_front_relative()
```


### Resources

http://stackoverflow.com/questions/5551286/filling-gaps-in-a-numpy-array

http://stackoverflow.com/questions/20753288/filling-gaps-on-an-image-using-numpy-and-scipy

http://stackoverflow.com/questions/12923593/interpolation-of-sparse-grid-using-python-preferably-scipy

http://stackoverflow.com/questions/14119892/python-4d-linear-interpolation-on-a-rectangular-grid

http://stackoverflow.com/questions/16217995/fast-interpolation-of-regularly-sampled-3d-data-with-different-intervals-in-x-y/16221098#16221098


### Approach

The simplest solution is to replace the invalid data with the
nearest valid neighbour.

http://stackoverflow.com/questions/5551286/filling-gaps-in-a-numpy-array


Another approach is to find the valid bound of each contiguous
invalid region and linearly interpolate from this. Or we can do
something fancy like a rbf.

We have a choice over whether we do the interpolation over just
space, for each time step; or, whether we use the full 3D data.

If we do it at each time step then we can just use 2d interpolation
- just need to find a way to apply this efficiently. 


We need a conceptual approach to solving this problem.

We have 3 dimensions - two space and one time. How many of them do
we actually need to use in the interpolation?

The speed of the current is

```python
fx, ft = pp.fit_front()
uf = (np.diff(fx) / np.diff(ft)).mean()
print "front speed = ", uf
```

The sampling intervals are

```python
dz = pp.Z[1, 0, 0] - pp.Z[0, 0, 0]
dx = pp.X[0, 1, 0] - pp.X[0, 0, 0]
dt = pp.T[0, 0, 1] - pp.T[0, 0, 0]

print "Sampling intervals:"
print "time: %s seconds" % dt
print "vertical: %s metres" % dz
print "horizontal: %s metres" % dx
```
Sampling intervals:
time: 0.0100002 seconds
vertical: 0.00115751 metres
horizontal: 0.00143858 metres

In the front relative coordinates we can consider the time interval
as an equivalent distance, which is

```python
print "equivalent time: %s metres" % (uf * dt)
```
equivalent time: 0.000636176 metres

That is, the time sampling is finer than the spatial sampling.
