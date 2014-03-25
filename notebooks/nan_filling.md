Interpolation of missing values in gridded data
===============================================

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
%matplotlib
import numpy as np
import matplotlib.pyplot as plt
import gc_turbulence as g

r = g.ProcessedRun(cache_path=g.default_processed + 'r13_12_17c.hdf5')

velocities = (r.U, r.V, r.W)

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
across all of  the data. The invalid values are largely present when
the front is passing through.


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


Linear interpolation
--------------------

Let's try find time slice with the most missing values and take a
subsection of it:

```python
sample = np.s_[20:30, :, :]

u = r.U[sample]
v = r.V[sample]
w = r.W[sample]
x = r.X[sample]
z = r.Z[sample]
t = r.T[sample]

# replace the zeros with nans
u[u==0] = np.nan

# find the time index with the most invalid values
import scipy.stats as stats
mode_index, mode_count = stats.mode(np.where(np.isnan(u))[-1])

sample_2d = np.s_[..., mode_index[0]]

us = u[sample_2d]
xs = x[sample_2d]
zs = z[sample_2d]

plt.contourf(xs, zs, us, 50)
```

The white areas are what we want to interpolate over. We can do this
using the `LinearNDInterpolator` from `scipy.interpolate`:

```python
import scipy.interpolate as interp

valid = np.where(~np.isnan(us))
invalid = np.where(np.isnan(us))

valid_points = np.vstack((zs[valid], xs[valid])).T
valid_values = us[valid]

imputer = interp.LinearNDInterpolator(valid_points, valid_values)

invalid_points = np.vstack((zs[invalid], xs[invalid])).T
invalid_values = imputer(invalid_points)

infilled = us.copy()
infilled[invalid] = invalid_values

plt.figure()
plt.contourf(xs, zs, infilled, 50)
```

This scales terribly:

```python
valid = np.where(~np.isnan(u))
invalid = np.where(np.isnan(u))

valid_points = np.vstack((z[valid], x[valid], t[valid])).T
valid_values = u[valid]

imputer = scipy.interpolate.LinearNDInterpolator(valid_points, valid_values)

invalid_points = np.vstack((z[invalid], x[invalid])).T
invalid_values = imputer(invalid_points)
```

Alternately we could use griddata, which basically wraps the above
method.

Regardless, constructing the linear interpolator on 10% of a single
runs data takes a very long time. It scales with $N^3$ and the full
N is about 50 million.

We are constructing a Qhull triangulation on a set of points that
come from a regular grid. However we aren't using the fact that we
have a regular grid at all.



Interpolation from valid shell
------------------------------

We can be more efficient by taking advantage of the fact that we
know where our data is invalid - we don't have to construct an
interpolator across the entire data field, just over the regions
that contain invalid data, which are localised.

Each invalid region of the data is surrounded by a shell of valid
data. We can use this shell of valid data as the source for a linear
interpolator and then compute the estimated values of the
interpolated data inside the shell on the regular grid.

We follow this approach:

1. Label the invalid regions of the data
2. Find the valid shell of each region.
3. Construct an interpolator for each valid shell
4. For each label, evaluate the corresponding interpolator over the
   internal coordinates of the label.

Labelling the regions:

```python
import scipy.ndimage as ndi

invalid = np.isnan(u)
valid = ~invalid

# diagonally connected neighbours
connection_structure = np.ones((3, 3, 3))
labels, n = ndi.label(invalid, structure=connection_structure)
```

We find the valid shell by exploiting the fact that our data is on a
rectangular grid and using binary dilation:

```python
def find_valid_shell(label):
    """For an n-dimensional boolean input, return an array of the
    same shape that is true on the exterior surface of the true
    volume in the input."""
    # we use two iterations so that we get the corner pieces as well
    dilation = ndi.binary_dilation(label, structure=np.ones((3, 3, 3)))
    shell = dilation & ~label
    return shell
```

We have now drastically reduced the number of points that our
interpolator uses in its construction.

Construct an interpolator from a valid shell:

```python
def construct_interpolator(valid_shell):
    valid_points = np.vstack((z[valid_shell], x[valid_shell], t[valid_shell])).T
    valid_values = u[valid_shell]
    # this is how to work with three components:
    # valid_values = np.vstack((u[valid_shell], v[valid_shell], w[valid_shell])).T
    interpolator = interp.LinearNDInterpolator(valid_points, valid_values)
    return interpolator
```

Evaluate the points inside the shell:

```python
label = labels == 1
valid_shell = find_valid_shell(label)

interpolator = construct_interpolator(valid_shell)
invalid_points = np.vstack((z[label], x[label], t[label])).T
invalid_values = interpolator(invalid_points)
```

As we are using linear interpolation, we actually need only compute
a single interpolator for the entire field. We can compute the valid
shell around all of the invalid regions and use that as input:

```python
def compare_techniques():
    complete_valid_shell = np.where(find_valid_shell(invalid))
    complete_interpolator = construct_interpolator(complete_valid_shell)

    single_valid_shell = np.where(find_valid_shell(labels==1))
    single_interpolator = construct_interpolator(single_valid_shell)

    # only evaluate inside the labels == 1 shell
    label1 = np.where(labels == 1)
    invalid_points = np.vstack((z[label1], x[label1], t[label1])).T

    complete_invalid_values = complete_interpolator(invalid_points)
    single_invalid_values = single_interpolator(invalid_points)
    return complete_invalid_values, single_invalid_values

print np.allclose(*compare_techniques())
```

Putting it all together:

```python
invalid = np.isnan(u)
complete_valid_shell = np.where(find_valid_shell(invalid))
interpolator = construct_interpolator(complete_valid_shell)

invalid_points = np.vstack((z[invalid], x[invalid], t[invalid])).T
invalid_values = interpolator(invalid_points)

uc = u.copy()
uc[invalid] = invalid_values

ucs = uc[sample_2d]
xs = x[sample_2d]
zs = z[sample_2d]

plt.figure()
plt.contourf(xs, zs, ucs, 50)
```

### Performance

This is slow. Our single interpolator method evaluates a lot of
space on the exterior of the valid shells. Our patch by patch method
calculates a binary dilation for each patch, which is time consuming
over ~5000 patches.

`ndimage.find_objects` returns a slice that isolates an individual
object in some space.

```python
import numpy as np
import gc_turbulence as g
import scipy.ndimage as ndi
import scipy.interpolate as interp

run = g.SingleLayerRun(cache_path=g.default_cache + 'r13_12_17c.hdf5')
run.load()
pp = g.PreProcessor(run)
pp.extract_valid_region()
pp.filter_zeroes()
# pp.interpolate_zeroes()

invalid = np.isnan(pp.U)
coords = pp.Z, pp.X, pp.T
data = pp.U, pp.V, pp.W

# find the complete valid shell
invalid_with_shell = ndi.binary_dilation(invalid, structure=np.ones((3, 3, 3)))
complete_valid_shell = invalid_with_shell & ~invalid
# isolate disconnected volumes. A volume consists of an invalid core
# and a valid shell
volumes, n = ndi.label(invalid_with_shell)

all_coords = np.concatenate([c[None] for c in coords])

def fillin(volume):
    """Construct the interpolator for a single shell and evaluate
    inside."""
    # construct the interpolator for a single shell
    shell = volume & ~invalid
    inside_shell = volume & invalid

    valid_points = all_coords[..., shell].T
    # valid_values = np.vstack(d[shell] for d in data).T
    valid_values = pp.U[shell]

    # coordinates of all of the invalid points
    invalid_points = all_coords[..., inside_shell].T

    interpolator = interp.LinearNDInterpolator(valid_points, valid_values)
    # this is nan outside of the shell
    invalid_values = interpolator(invalid_points)
    return invalid_values

    pp.U[inside_shell] = invalid_values
    return
    in_shell = ~np.isnan(invalid_values)
    # XXX: how do we make in_shell index the structured array?
    # this doesn't work because we're actually assigning to a copy
    # pp.U[invalid][in_shell] = invalid_values[in_shell]
    # need to use flat indexing
    # see https://stackoverflow.com/a/17050613/1157089
    pp.U.flat[np.flatnonzero(invalid)[in_shell]] = invalid_values[in_shell]


def fillall(n=n):
    for i in range(1, n):
        print "# {} \r".format(i),
        sys.stdout.flush()
        fillin(volumes==i)
```

Alternately we can use `find_objects`. We binary dilate the invalid
data so that there is some valid data around it. We then label this
and get the slices from find objects. For each slice we construct an
interpolator from the points that are in the valid shell and
interpolate the points that are invalid.

```python
import numpy as np
import gc_turbulence as g
import scipy.ndimage as ndi
import scipy.interpolate as interp

run = g.SingleLayerRun(cache_path=g.default_cache + 'r13_12_17c.hdf5')
run.load()
pp = g.PreProcessor(run)
pp.extract_valid_region()
pp.filter_zeroes()
# pp.interpolate_zeroes()

invalid = np.isnan(pp.U)
coords = pp.Z, pp.X, pp.T
data = pp.U, pp.V, pp.W

invalid_with_shell = ndi.binary_dilation(invalid, structure=np.ones((3, 3, 3)))
complete_valid_shell = invalid_with_shell & ~invalid
labels, n = ndi.label(invalid_with_shell)
slices = ndi.find_objects(labels)

def interpolate_region(slice):
    nans = invalid[slice]
    shell = complete_valid_shell[slice]

    valid_points = np.vstack(c[slice][shell] for c in coords).T
    valid_values = pp.U[slice][shell]

    interpolator = interp.LinearNDInterpolator(valid_points, valid_values)

    invalid_points = np.vstack(c[slice][nans] for c in coords).T
    invalid_values = interpolator(invalid_points).astype(valid_values.dtype)

    # TODO: multiproc here

    pp.U[slice][nans] = invalid_values

    ## PROBLEM: the slice might contain points that are outside of
    ## the shell. These points will be over written with nan
    ## regardless of their original state. We need to not overwrite
    ## these points if they have already been filled in by
    ## interpolation from another shell.
    ##

    ## Is this guaranteed not to happen from the dilation that we
    ## have used?

    ## No. But we can just run the process repeatedly until the data
    ## doesn't contain nans

    # np.putmask(pp.U[slice][nans], ~np.isnan(invalid_values), invalid_values)
    # copy interpolated values to U only if they aren't nan
    # np.copyto(pp.U[slice][nans], invalid_values, where=~np.isnan(invalid_values))

def intall():
    for i, slice in enumerate(slices):
        print "# {} \r".format(i),
        sys.stdout.flush()
        interpolate_region(slice)
```

This took 15 mins for our test run. We could parallelise it,
calculating the invalid values in parallel and storing them in
memory, then assigning to the array when all are calculated.

This method still hangs on non convex regions, but not for as long.
There might be a way to reduce the hang by ignoring points that
dont' contribute much.

Got a big performance gain by computing the shell properly. I was
doing two iterations instead of defining a diagonally connected
structure. This roughly halves the number of points to construct the
interpolator, which has a big effect on time because this scales as
N^2.

The `interpolate_region` method is by far the fastest. It doesn't
remove all of the nans in a single pass because a single slice can
have bits of another volume in it. You can add an extra step to the
method to only calculate the interpolation only for points that are
explicitly part of the volume that the slice references but this is
computationally expensive as it has to be done in the inner loop. It
is much quicker to perform multiple passes.

TODO on Tuesday:

- Is single core version of `interpolate_region` faster? No.

- Write into main code, use separate file if necessary.

- Rescale time axis for interpolation

- Deal with anomalous high / low values as well.

- Re-process a load of runs.
- Clean up this notebook into descriptive method.

- Write in histograms to analysis and rewrite 2d_pdf to call them.
- Histograms for u, w, magnitude, var(v)


### Implementation

[This commit](www.github.com/aaren/lab_turbulence/....)

Usage:

```python
import numpy as np
import matplotlib.pyplot as plt
import gc_turbulence as g

run = g.SingleLayerRun(cache_path=g.default_cache + 'r13_12_17c.hdf5')
run.load()
pp = g.PreProcessor(run)
pp.extract_valid_region()
pp.filter_zeroes()

# find the time index with the most invalid values
import scipy.stats as stats
mode_index, mode_count = stats.mode(np.where(np.isnan(pp.U))[-1])
idx = int(mode_index[0])

sample_2d = np.s_[..., idx]
us = pp.U[sample_2d]
xs = pp.X[sample_2d]
zs = pp.Z[sample_2d]

fig, axes = plt.subplots(ncols=2)
axes[0].contourf(xs, zs, us, 50)
plt.draw()

pp.interpolate_nan(sub_region=np.s[:, :, idx - 100: idx + 100])

axes[1].contourf(xs, zs, us, 50)
```


### Validation

Is our method actually working?

We could take some complete data (no nans) and set some of it equal to
nan, then apply the interpolation above and see how close we get to
the actual values.


### Filtering anomalies

```python
import scipy.ndimage as ndi

o = ndi.uniform_filter(pp.U, size=3) / 9
thresh = 0.1
anomaly = np.where(np.abs(o - pp.U) > thresh)
pp.U[anomaly] = np.nan
# run nan interpolation again

```


### Extension

The obvious way to improve this method is to upgrade the
interpolator to something more fancy than a linear method.
For example, we could use a radial basis function:
`scipy.interpolate.rbf`.
