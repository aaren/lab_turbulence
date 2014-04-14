Dynamic mode decomposition of gravity currents
==============================================

This is going to be an exploration of applying the Dynamic Mode
Decomposition (DMD) to PIV measurements of gravity currents in the
lab.

We are going to do the following:

1. apply the DMD to data from a single run and look at the
   structure.

2. apply the DMD to an ensemble of multiple runs and compare with
   single run decomposition.

3. Reconstruct a single run from a limited number of the low order
   modes and compare, statistically, with the original data.

4. Reconstruct from the ensemble DMD.


Single run setup
----------------

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

import modred as mr

import gc_turbulence as g

r = g.ProcessedRun(cache_path=g.default_processed + 'r13_12_16a.hdf5')
```

Sanity check (plotting non-dimensionalised front relative data):

```python
example = np.s_[:,30,:]
u_levels = np.linspace(-0.5, 0.2, 100)
plt.contourf(r.Tf_[example], r.Zf_[example], r.Uf_[example], levels=u_levels)
```

verify no nans anywhere:

```python
nans = {v: np.isnan(getattr(r, v)[:]).sum() for v in r.vectors.names if v != 'front_speed'}
print nans
```

There are nans in the front relative coords because they attempt to
get data from outside of the measurement region. This is by design,
but we need to be able to reduce the data to a volume that contains
no nans.

```python
def find_nan_slice(data):
    """Find the slice that contains nans in the data, assuming that
    they are contiguous.
    """
    return ndi.find_objects(np.isnan(data[...]))

def complement(nan_slices):
    """Compute the slice that complements a slice that contains
    nans, i.e. return the slice that will not include any nans.
    """
    sz, sx, st = nan_slices[0]
    return (slice(None), slice(None), slice(st.stop, None))

u = r.Uf_[complement(find_nan_slice(r.Uf_[:]))]
z = r.Zf_[complement(find_nan_slice(r.Uf_[:]))][example]
x = r.Xf_[complement(find_nan_slice(r.Uf_[:]))][example]
t = r.Tf_[complement(find_nan_slice(r.Uf_[:]))][example]
```

DMD basics
----------

Let's recap the basis of the Dynamic Mode Decomposition.

Given a sequence of flow fields, i.e. our data, the DMD gives us a
series of *dynamic modes* with corresponding complex eigenvalues
and mode norms representing energy content.

The complex eigenvalues (Ritz values) $\phi_i$ can be transformed
into frequency space with

$$
\lambda_i = \log \phi_i / \delta
$$

where $\delta$ is the step size in the decomposition axis.
$\lambda_i$ is complex and represents the growth / decay and
frequency of the corresponding mode.



DMD method
----------

```python
def make_snapshots(data):
    """Create the matrix of snapshots by flattening the non decomp
    axes so we have a 2d array where we index the decomp axis like
    snapshots[:,i]

    The decomposition axis is the x dimension of the front relative
    data.
    """
    iz, ix, it = data.shape
    snapshots = data.transpose((0, 2, 1)).reshape((-1, ix))
    return snapshots

def calculate_dmd(data, n_modes=5):
    """Dynamic mode decomposition, using Uf as the series of vectors."""
    iz, ix, it = data.shape
    # matrix of snapshots, performing spatial decomposition along
    # the x axis
    snapshots = data.transpose((0, 2, 1)).reshape((-1, ix))

    modes, ritz_values, norms \
        = mr.compute_DMD_matrices_snaps_method(snapshots, range(n_modes))

    # as array, reshape to data dims with mode number as first index
    reshaped_modes = modes.A.reshape((iz, it, -1)).transpose((2, 0, 1))
    return reshaped_modes, ritz_values, norms

def calculate_pod(data, n_modes=5):
    """Dynamic mode decomposition, using Uf as the series of vectors."""
    iz, ix, it = data.shape
    # matrix of snapshots, performing spatial decomposition along
    # the x axis
    snapshots = data.transpose((0, 2, 1)).reshape((-1, ix))

    modes, eigen_vals \
        = mr.compute_POD_matrices_snaps_method(snapshots, range(n_modes))

    # as array, reshape to data dims with mode number as first index
    reshaped_modes = modes.A.reshape((iz, it, -1)).transpose((2, 0, 1))
    return reshaped_modes, eigen_vals
```

POD of a single run
-------------------

```python
pmodes, pv = calculate_pod(u, n_modes=10)

fig, axes = plt.subplots(nrows=len(modes))

for i, mode in enumerate(pmodes):
    axes[i].contourf(t, z, mode / 0.015, 100)

fig.tight_layout()
```

DMD of a single run
-------------------

```python
modes, ritz_values, norms = calculate_dmd(u, n_modes=10)

fig, axes = plt.subplots(nrows=len(modes[0::2]))

for i, mode in enumerate(modes[0::2]):
    axes[i].contourf(t, z, np.abs(mode), 100)
```
