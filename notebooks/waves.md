```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import gc_turbulence as g
```

Removing waves from lab data
----------------------------

Lab data contain waves, which are complicating the frequency
analysis of the turbulence.

Assume that these waves are linear and superpose onto the velocity
data. Then if we can isolate the wave signal we can subtract it from
the data and recover wave free data.

What do the waves look like?

```python
index = 'r14_01_14a'
cache_path = g.default_processed + index + '.hdf5'
r = g.ProcessedRun(cache_path=cache_path)

u_levels = np.linspace(*np.percentile(r.Uf[...], (1, 99)), num=100)

tf = r.Tf[:, 0, :]
zf = r.Zf[:, 0, :]
```

As we might expect, there are no waves in the mean front relative data:

```python
mean = np.mean(r.Uf[...], axis=1)
plt.contourf(tf, zf, mean, levels=u_levels)
```

If we subtract the mean from the data and look at a single vertical
slice we can see waves:

```python
mean_subtracted = r.Uf[...] - mean[..., None, ...]
c = plt.contourf(tf, zf, mean_subtracted[:, 30, :], 100)
```

We can see that the waves are present throughout the data. In the
non turbulent region outside of the current we can see them quite
well:

```python
c = plt.contourf(tf[-5:], zf[-5:], mean_subtracted[-5:, 30, :], 100)
wave_levels = c.levels
```

We can see that these are standing waves when we plot in the non
current relative coordinate system:

XXX: they could just be long waves?? i.e. long compared to the horizontal
measurement window.

```python
x = r.X[0, :, :]
t = r.T[0, :, :]
plt.contourf(t, x, np.mean(r.U[-10:, :, :], axis=0), levels=wave_levels)
plt.xlabel('time')
plt.ylabel('horizontal in measurement window')
```

This suggests a means of extracting the wave signal. Transform the
data to the front relative system and take the mean of the current
velocity; subtract this from the data and transform back to the lab
relative system; then average in the vertical and horizontal to
obtain the wave signal.

We can then subtract this from the velocity data and obtain non wavy
velocities.

We need to modify the front transform routines a bit:

```python
r.sample_time_shape = self.T.shape[:2] + [1]
# get the real start time of the data and the
# sampling distance in time (dt)
rt = r.T[0, 0, :]
t0 = rt[0]
dt = rt[1] - rt[0]

def relative_sample_times(dt, shape):
    """Create the 3d array of front relative sample times, i.e.
    the time (in seconds) relative to the passage of the gravity
    current front in the FRONT frame.

    This is just a 1d array in the time axis, repeated over the
    z and x axes.
    """
    # start and end times (s) relative to front passage
    # TODO: move somewhere higher level
    pre_front = -5
    post_front = 20
    relative_sample_times = np.arange(pre_front, post_front, dt)

    # extend over x and z
    sz, sx, _ = shape
    relative_sample_times = np.tile(relative_sample_times, (sz, sx, 1))

    return relative_sample_times


def compute_front_relative_transform_coords(relative_sample_times,
                                            trajectory, dt):
    """Calculate the index coordinates needed to transform the
    data from the LAB frame to the FRONT frame.

    In general these coordinates can be non-integer and
    negative. The output from this function is suitable for
    using in map_coordinates.

    fit - defaults to '1d', which is to fit a straight line to the
            current and resample the time from that.

            Using None will turn off any fitting and will
            just use the raw time / space detection.

    """
    front_space, front_time = trajectory

    # compute the times at which we need to sample the original
    # data to get front relative data by adding the time of
    # front passage* onto the relative sampling times
    # *(as a function of x)
    rtf = front_time[None, ..., None] + relative_sample_times

    # now we transform real coordinates to index coordinates.
    # You might want to skip this step and just compute the
    # index coordinates straight out. The reason not to do this
    # is that the coordinates can be negative and non-integer.
    # map_coordinates is used as a fancy indexer for these
    # coordinates.

    # grid coordinates of the sampling times
    # (has to be relative to what the time is at
    # the start of the data).
    t_coords = (rtf - t0) / dt

    # z and x coords are the same as before
    # Actually, the x_coords should be created by extending
    # front_space - it works here because front_space is the
    # same as self.X[0, :, 0]
    z_coords, x_coords = np.indices(t_coords.shape)[:2]

    # required shape of the coordinates array is
    # (3, rz.size, rx.size, rt.size)
    coords = np.concatenate((z_coords[None],
                             x_coords[None],
                             t_coords[None]), axis=0)
    return coords

def transform(vector, coords, order=3):
    return ndi.map_coordinates(vector, coords, cval=np.nan, order=order)

def transform_to_front_relative(self, fit='1d'):
    """Transform the data into coordinates relative to the
    position of the gravity current front, i.e. from the LAB
    frame to the FRONT frame.

    The LAB frame is the frame of reference in which the data
    were originally acquired, with velocities relative to the
    lab rest frame, times relative to the experiment start and
    space relative to the calibration target.

    The FRONT frame is the frame of reference in which the
    gravity current front is at rest, with velocities relative
    to the front, times relative to the time of front passage
    and space as in the LAB frame.

    Implementation takes advantage of regular rectangular data
    and uses map_coordinates.
    """
    trajectory = self.fit_front()
    dt = self.dt
    rsample_times = relative_sample_times(dt, self.sample_time_shape)

    coords = self.compute_front_relative_transform_coords(rsample_times,
                                                          trajectory,
                                                          dt)

    # use order 0 because 6x as fast here (3s vs 20s) and for x
    # and z it makes no difference
    self.Xf = transform(self.X, coords, order=0)
    self.Zf = transform(self.Z, coords, order=0)

    # these are the skewed original times (i.e. LAB frame)
    self.Tfs = transform(self.T, coords, order=3)
    # these are the times relative to front passage (i.e. FRONT frame)
    self.Tf = rsample_times

    # the streamwise component is in the FRONT frame
    fs = self.front_speed
    self.Uf = transform(self.U) - fs
    # cross-stream, vertical components
    self.Vf = transform(self.V, coords)
    self.Wf = transform(self.W, coords)

    # N.B. there is an assumption here that r.t, r.z and r.x are
    # 3d arrays. They are redundant in that they repeat over 2 of
    # their axes (r.z, r.x, r.t = np.meshgrid(z, x, t, indexing='ij'))


# transform from lab relative to front relative
front_time = r.ft[...]
front_data = relative_sample_times(dt=0.01, shape=r.T.shape) + front_time[None, ..., None]

t0 = r.T[...].min()

t_coords = (front_data - t0) / dt

z_coords, x_coords = np.indices(t_coords.shape)[:2]

coords = np.concatenate((z_coords[None],
                         x_coords[None],
                         t_coords[None]), axis=0)

# transform from front relative to lab relative
front_time = r.ft[...]
lab_data = r.T[...] - front_time[None, ..., None]

tf0 = r.Tf[...].min()

tf_coords = (lab_data - tf0) / dt

zf_coords, xf_coords = np.indices(t_coords.shape)[:2]

fcoords = np.concatenate((zf_coords[None],
                          xf_coords[None],
                          tf_coords[None]), axis=0)

U = transform(r.Uf, fcoords, order=0) - r.front_speed

# compute a mean through the current and transform it to the lab
# frame
mean_u = np.mean(r.Uf, axis=1, keepdims=True)
full_mean_u = np.repeat(mean_u, r.Uf.shape[1], axis=1)
trans_full_mean_u = transform(full_mean_u, fcoords, order=0) + fs

# now we can have mean subtracted data in the lab frame
mean_sub_u = r.U[...] - trans_full_mean_u[...]

# from which we might be able to get the wave signal,
# using the fact that it is homogeneous in x and z in the lab frame
waves = np.mean(np.mean(mean_sub_u, axis=0, keepdims=True), axis=0, keepdims=True)

u_no_waves = r.U - waves
#BOOM!
```

Looking at this signal, it seems that the waves might be fast rather
than stationary. Is there a wave frame? What speed should these
waves have?

```python
sub = np.s_[20, :, 250: 2000]
plt.contourf(r.T[sub], r.X[sub], mean_sub_u[sub], 100)
```

When we look closer we can see that these are surface waves in deep
water. Deep water waves have phase velocity varying with wavelength
and here we have a range of wavelengths and speeds. Our method has
largely worked because surface waves are *fast*, with almost
vertical trajectories in (t, x) allowing us to average over x.

Strictly we cannot average over x, only over z. If we do this, we
gain information about the wave trajectories at the expense of noise
coming from the current.

Comparison:

```python
rwaves = np.mean(mean_sub_u, axis=0, keepdims=True)
fig, axes = plt.subplots(nrows=2)
axes[0].contourf(rwaves[0], 100)
axes[0].set_title('only z mean')
axes[1].contourf(np.repeat(waves, rwaves.shape[1], axis=1)[0], 100)
axes[1].set_title('x and z mean')
```

The 'noise' coming from the current is any process that is happening
on a scale that is longer than the time that we observe the current
for, e.g. a slow overturning eddy. We are making the assumption that
the mean subtracted structure averages to zero over time. This is
likely true for small eddies but may fail at larger scales.

For example, the head of the current could shed vortices that
propagate backwards relative to the head. These will show up in the
mean subtracted field. Large structures may not be averaged out over
z and will show up as a propagating feature in (t, x). Averaging
over x will reduce the signature of this sort of structure.

Another limit in this approach is the assumption of linear
superposition. When linear waves collide, the result is easily
obtained as the addition of the waves. This is generally not the
case with non-linear waves. For example, KdV solitons combine in a
manner more like particles colliding - they can not be modelled by
simply adding them together. More exotic solitons might have phase
changes on colliding, e.g. in the BDO model.

The surface waves here are very linear looking, however the gravity
current is decidedly non-linear in character. It is not clear at all
that we can combine the two by addition. That said, assuming linear
superposition is a good first approximation.

When we subtract the waves from the velocity pattern we get a
cleaner looking structure:

```python
ix = 10
u_levels = np.linspace(-0.03, 0.1, 100)
fig, axes = plt.subplots(nrows=4)
axes[0].set_title('before')
axes[0].contourf(r.T[ix], r.X[ix], r.U[ix], levels=u_levels)
axes[1].set_title('after, mean(x, z)')
axes[1].contourf(r.T[ix], r.X[ix], (r.U - waves)[ix], levels=u_levels)
axes[2].set_title('after, mean(z)')
axes[2].contourf(r.T[ix], r.X[ix], (r.U - rwaves)[ix], levels=u_levels)
axes[3].set_title('after, difference')
axes[3].contourf(r.T[ix], r.X[ix], (waves - rwaves)[0], levels=np.linspace(-0.015, 0.015, 100))
fig.tight_layout()
```

The improvement in data clarity is significant regardless of the
averaging used.

Something to check is whether the wave pattern varies run to run.
The wave speeds will vary with the reduced gravity on the surface.

How do we decide which mean to take? It isn't clear - they both have
advantages.

We could use both averaging methods, with multiple transforms
between frames. Method:

1. Compute mean from Uf and subtract

2. Transform mean subtracted Uf to LAB frame

3. Compute wave field as mean(x, z)

4. Subtract wave field from mean subtracted Uf -> varying-current

5. Transform varying-current back to FRONT frame

6. Subtract from mean subtracted Uf

7. Transform back to LAB frame

8. Compute final wave field as mean(z)

XXX: this doesn't really make sense does it?
It sort of does - the basic idea is to use the fact that the waves
are homogeneous in z and nearly homogeneous in x, combined with the
current being nearly statistically homogeneous in x when we
transform to the front frame. The implementation below gets it:

```python
# compute mean from uf
mean_uf = np.mean(r.Uf[...], axis=1, keepdims=True)
# expand mean over all x
full_mean_uf = np.repeat(mean_uf, r.Uf.shape[1], axis=1)
# transform to lab frame
trans_mean_uf = transform(full_mean_uf, fcoords, order=0) + fs
# replace nan with zero
trans_mean_uf[np.isnan(trans_mean_uf)] = 0
# subtract mean current from lab frame
mean_sub_u = r.U[...] - trans_mean_uf

# compute wave field as mean(x, z)
waves_xz = np.mean(np.mean(mean_sub_u, axis=0, keepdims=True), axis=1, keepdims=True)
waves_z = np.mean(mean_sub_u, axis=0, keepdims=True)

# subtract wave field from mean subtracted u
varying_current = mean_sub_u - waves_xz

# transform varying current back to front frame
trans_varying_current = transform(varying_current, coords, order=0)

# subtract the varying current from the velocity
# and compute a new mean
new_mean = np.mean(r.Uf - trans_varying_current, axis=1, keepdims=True)
full_new_mean = np.repeat(new_mean, r.Uf.shape[1], axis=1)
trans_new_mean = transform(full_new_mean, fcoords, order=0) + fs
trans_new_mean[np.isnan(trans_new_mean)] = 0

# subtract transformed mean from u
new_mean_sub_u = r.U[...] - trans_new_mean - varying_current

# compute new waves
new_waves = np.mean(new_mean_sub_u, axis=0, keepdims=True)

# hmmmm.... some sort of mean signal left in the waves - we don't
# want to be subtracting this.
```

example plots:

```python
plt.contourf(new_waves[0], 100)
plt.contourf(new_waves[0] - waves_xz[0], 100)
plt.contourf(new_waves[0] - waves_z[0], 100)
plt.contourf((r.U - new_waves)[0], levels=u_levels)
ex = np.s_[20, 50, :]
plt.plot(r.U[ex])
plt.plot(new_waves[ex])
plt.plot(waves_x[ex])
plt.plot(waves_xz[ex])
```


There are two further ways of extracting the wave signal:

1. Find a vertical level that is unaffected by the gravity current
   and extract a full time series from this level.

2. Use the pre-current ambient to isolate a partial signal and
   extrapolate to the remainder of the time series.

In some runs the current is deep and method 1 is not practical. We
therefore need to extrapolate the time series.

```python
# extract velocities up to point of current onset and take the mean
# over x and z
laminar_slice = np.s_[:, :, 250:2500]
u = r.U[laminar_slice]
v = r.V[laminar_slice]
w = r.W[laminar_slice]

t = r.T[laminar_slice][0, 0]

mean_u = np.mean(np.mean(u, axis=0), axis=0)
mean_v = np.mean(np.mean(v, axis=0), axis=0)
mean_w = np.mean(np.mean(w, axis=0), axis=0)

# basic 20-point (5hz) smoothing
window = np.hanning(20)
smooth_u = np.convolve(mean_u, window / window.sum(), mode='same')
smooth_v = np.convolve(mean_v, window / window.sum(), mode='same')
smooth_w = np.convolve(mean_w, window / window.sum(), mode='same')
```

```python
plt.plot(t, smooth_u, label='u')
plt.xlabel('time')
title = plt.title('pre-current streamwise velocity')
```

We want to extrapolate the time series `smooth_u`. A matlab method
is [here][dsp_stackexchange]. We use an auto-regressive method to
extrapolate the time series by using a linear model, assuming each
point is a linear combination of previous points.

[dsp_stackexchange]: http://dsp.stackexchange.com/questions/101/how-do-i-extrapolate-a-1d-signal

```python
from spectrum import arburg

n = 500  # order of AR model
p = 500  # number of samples in extrapolation
m = 150  # point at which to start prediction

AR_coeffs, AR_noisevariance, AR_reflection_coeffs = arburg(smooth_u, order=500)

extrapolated_u = np.hstack([smooth_u, np.zeros(p)])

# filter data 
sig.lfilter()
```


Rotation correction
-------------------

This points us towards an additional correction to be applied to the
velocities. Linear surface waves should be homogeneous in the
vertical and cross stream axes due to the symmetry of the system.
That is, outside of turbulent flow, we should not have any variation
in $v$, the cross stream component of velocity.

```python
plt.plot(t, mean_u, label=r'$u$ (streamwise)')
plt.plot(t, mean_v, label=r'$v$ (cross stream)')
plt.plot(t, mean_w, label=r'$w$ (vertical)')
plt.xlabel('time')
plt.ylabel('pre-current velocities')
plt.legend()
```

We can see that the $u$ and $v$ components are zero at the same
time and that they move in anti-phase, suggesting that the laser
light sheet is tilted in these axes. The $w$ component does not show
covariance beyond what we might expect for deep water waves, so we
can assume that the light sheet is vertical.

It is possible that the correction is not linear and that it varies
with the velocity, however we can implement a simple first order
correction with this difference.

We can compute the optimal (by least squares) rotation matrix that
maps between two sets of points using SVD via the [Kabsch algorithm][kabsch].

[kabsch]: http://en.wikipedia.org/wiki/Kabsch_algorithm

[](igl.ethz.ch/projects/ARAP/svd_rot.pdf)
Challis, J.H. (1995). A procedure for determining rigid body transformation parameters. J. Biomechanics 28, 733-737.
[](http://nghiaho.com/?page_id=671)

```python
# vector of recorded points (2xn)
vec = np.vstack((smooth_u.flat, smooth_v.flat))

# real cross stream should be zero
# conserve speed to find real streamwise
mod_u = np.hypot(smooth_u, smooth_v)
ur = np.sign(smooth_u) * np.sign(smooth_v) * mod_u
vr = np.zeros(ur.shape)

# vector of real points (2xn)
vecr = np.vstack((ur.flat, vr.flat))

# mean subtract
vec_ = vec - np.mean(vec, axis=1, keepdims=True)
vecr_ = vecr - np.mean(vecr, axis=1, keepdims=True)

# now do least squares minimisation through SVD to find the rotation
# matrix R, vecr = R vec

# svd of the covariance matrix
U, W, Vh = linalg.svd(np.dot(vec_, vecr_.T))
d = linalg.det(np.dot(Vh.T, U.T))

S = np.ones(W.size)
S[-1] = d

R = np.dot(Vh.T, S * U.T)
theta = np.arccos(R[0, 0])
```
