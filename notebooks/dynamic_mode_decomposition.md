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

t = r.Tf_[example]
z = r.Zf_[example]
u = r.Uf_[example]
v = r.Vf_[example]
w = r.Wf_[example]
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

uf_ = r.Uf_[complement(find_nan_slice(r.Uf_[:]))]
uf = r.Uf[complement(find_nan_slice(r.Uf[:]))]
z = r.Zf_[complement(find_nan_slice(r.Uf_[:]))][example]
x = r.Xf_[complement(find_nan_slice(r.Uf_[:]))][example]
t = r.Tf_[complement(find_nan_slice(r.Uf_[:]))][example]
```

DMD basics
----------

Let's recap the basis of the Dynamic Mode Decomposition (DMD).

Our data takes the form of a series of $N$ 2D snapshots of a flow
field. Each snapshot $\vec{u}_i$ contains a value at each of $M$
measurement points.

Our data is expressed on a regular grid, but the DMD does not
require this.

We reshape the data to form an $N\byM$ matrix - the data vector.
This vector describes the trajectory of our data through an
$M$-dimensional vector space.

The DMD is a means of fitting the trajectory $\vec{u}(t)$

Just as we can approximate any function $f(x)$ with a series of
fourier modes that cycle in space, we can fit $\vec{u}(t)$ with a
superposition of modes that cycle in the $M$-dimensional data space.

The DMD fits the trajectory $D_T$ with a superposition of modes
expressed through the basis $\vec{v}$:

$$
\vec{u}(t) = \Sum{j}{} \vec{v}_j \mu_i^t
$$

$$
\vec{u}(t) = \Sum{j}{} \vec{v}_j \exp{\lambda_i t}
$$

Performing the DMD gives us a set of complex modes $\phi_i$ and
corresponding complex frequencies $\lambda_i$. Hence each mode has a
frequency and decay rate.

The fit above is calculated by assuming that there is a linear
operator $\mat{A}$ that describes our system, mapping one snapshot
onto the next

$$
u_{i + 1} = \mat{A} u_{i}
$$

The *dynamic modes* $phi_i$ are approximations to the *eigenvectors*
of the matrix $\mat{A}$, with corresponding *eigenvalues*
$\lambda_i$. That is, the dynamic modes are invariant under the
system operator.

The dynamic modes represent spatially coherent structures of
distinct frequencies in the data.

As $\lambda_i$ are complex, we know the decay rates of each dynamic
mode, and the *DMD decomposes our data into saturated and transient
oscillatory modes* [@bagheri2013].


### Relation to the fourier transform

The DMD has a strong parallel with the fourier transform (and is
identical under mean subtraction [@chen-etal2012] sec4.3).

We decompose our data into a series of oscillating modes. In the
fourier transform we decompose directly into frequency space and
have complex coefficients that represent the frequency and phase of
each mode.

In the DMD, we decompose into a complex mode space where each mode
is associated with a complex frequency.

The fourier transform cannot capture growth rates and therefore
cannot isolate transient modes.


DMD method
----------

```python
def to_snaps(data):
    """Create the matrix of snapshots by flattening the non decomp
    axes so we have a 2d array where we index the decomp axis like
    snapshots[:,i]
    
    i.e. for M data points per snapshot and N snapshots, the
    snapshot matrix is shape (M, N)

    The decomposition axis is the x dimension of the front relative
    data.
    """
    iz, ix, it = data.shape
    snapshots = data.transpose((0, 2, 1)).reshape((-1, ix))
    return snapshots

def to_data(modes, data=u):
    """Reshape mode shaped data into what we originally put in.

    modes are shape (M, n_modes) where M is the number of
    measurement points (M = ix * it)
    """
    iz, ix, it = data.shape
    reshaped_modes = np.asarray(modes).reshape((iz, it, -1)).transpose((2, 0, 1))
    return reshaped_modes

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

fig, axes = plt.subplots(nrows=len(pmodes))

for i, mode in enumerate(pmodes):
    axes[i].contourf(t, z, mode / 0.015, 100)
    axes[i].set_xticks([])

fig.tight_layout()
```

DMD of a single run
-------------------

```python
modes, ritz_values, norms = calculate_dmd(u, n_modes=10)
amodes, aritz_values, anorms = calculate_dmd(u, n_modes=u.shape[1]-1)

fig, axes = plt.subplots(nrows=len(modes[0::2]))

for i, mode in enumerate(modes[0::2]):
    axes[i].contourf(t, z, mode.real, 100)
    axes[i].set_xticks([])
```

Reconstruction of a single run
------------------------------

We can reconstruct data from a POD basis by summing over a subset of
the basis multiplied by build coefficients $b_i$:

$$
R = \Sum{i}{} b_i \phi_i
$$

The build coefficients are calculated by projecting the POD back
onto the original data and represent the temporal evolution of each
basis vector. This is necessary as the POD basis does not contain
temporal information.

Reconstruction with a subset of the POD basis allows us to form a
low order representation of a dataset. As POD modes are ranked by
statistical energy we can account for the majority of the variance
in the data by only using the first few modes.

Where the POD isolates statistically uncorrelated structures, the
DMD isolates spatially coherent fourier modes.

We can view the DMD as the optimal solution to fitting the data
$\vec{u}(t)$ with the modes $\vec{\phi_i}$

$$
\vec{u}(t) = \Sum{j = 0}{r} \vec{\phi_j} \exp{\lambda_j t}
$$

Once we have computed the modes we can reconstruct the data with a
set $m$ of modes,

$$
\vec{u}_k = \vec{u}(\Delta t k) = \Sum{m}{} \vec{\phi_j} \exp{\lambda_j k \Delta t}
$$

### Mode selection

The problem with DMD reconstruction is that there is no clear way to
select the subset of modes, $m$. DMD does not rank modes
statistically because the modes are not statistically orthogonal. In
general we cannot say which set of modes will let us best
approximate the data, especially with complex dynamics.

@chen-etal2012 proposed selecting modes with "Optimised DMD", where
we find the best fit of $m$ modes to $p$ points in data space,
allowing for a residual at each point. However they only
demonstrated their method for 3 modes as it requires a numerical
search over a vast parameter space.

@jovanovic-etal2014 propose "Sparse DMD", where we minimise a cost
function involving the mode amplitudes and a parameterised proxy for
the number of modes. A numerical search finds the parameter which
selects a required number of modes. We then search again to find the
optimal mode amplitudes.

@semeraro-etal select *consistent* modes iteratively by projecting
the results of one iteration on the previous one, varying the
snapshots with an origin time. Modes are retained if their
projection is greater than a threshold.

-- does this last just amount to a cross validation?
-- they don't actually technically detail the method.

This procedure necessarily selects modes that have a growth rate
closer to zero.

#### Sparse DMD

I created a python implementation of sparse DMD based on Jovanovic's
Matlab source code.

We can apply this to our data:

```python
import all_dmdsp as sparse_dmd

snapshots = to_snaps(u)
dmd = sparse_dmd.SparseDMD(snapshots=snapshots)

# control parameter to vary (lets us select sparsity)
gammaval = np.logspace(0, 5, 100)
answer = dmd.dmdsp(gammaval)

xdmd = dmd.xdmd
Edmd = dmd.Edmd

plt.plot(np.log(Edmd.imag), np.abs(xdmd),'ko')
plt.xlabel('frequency')
ply.ylabel('amplitude')
```


Ensemble DMD
------------

```python
run_indices = ['r13_12_16a',
               'r13_12_16b',
               'r13_12_16c',
               'r13_12_16d',
               'r13_12_16e',]

caches = [g.default_processed + idx + '.hdf5' for idx in run_indices]
runs = [g.ProcessedRun(cache_path=c) for c in caches]
```

We need to find the non nan slice for the whole ensemble. In fact we
need everything on the same grid, which isn't guaranteed because of
the interpolation in post processing.

TODO: can we at least ensure a regular grid in post processing?
we can assume that is the case for now (and it does appear to be if
you pull the same index from each run), but we still need to find
the slice that doesn't get nan from any run.

create a function `ensemble_complement`. This can work either by
calling `complement` on each run or by ORing the isnan arrays.

```python
def ensemble_complement(runs):
    allnans = (np.isnan(r.Uf_[:]) for r in runs)
    allnan = reduce(np.logical_or, nans)
    bad_slice = ndi.find_objects(allnan)
    return complement(bad_slice)

good_slice = ensemble_complement(runs)
```

### Decomposition ensemble

Stack the data along the decomposition (x) axis:

```python
eu = np.hstack(r.Uf_[good_slice] for r in runs)
```

and perform the decomposition on this. The number of modes is
limited by the number of fields in the decomposition axis, i.e. we
have five times as many possible modes as in the single run case.

As half of the modes are conjugate to the others, this is a *bit*
like the nyquist limit in the fourier transform. The difference here
is that we are completely unrestricted in the frequencies that we
can pull out. The restriction comes in *mode* space rather than
frequency space.

TODO: do the decomposition and compare to the single layer case

- How does the distribution of $\lambda_i$ change?

It is possible to reconstruct a single run from its DMD modes
because of the phase information contained in the eigen-frequencies
$\lambda_i$.

This phase information is destroyed when we perform DMD on an ensemble.

* Can't we reconstruct the ensemble though?

```python
es = to_snaps(eu)
emodes, erv, en = mr.compute_DMD_matrices_snaps_method(es, slice(None))
```

The problem with this method is that extending the decomposition
axis increases the number of computed modes. We don't have a way of
selecting which modes are dynamically important.

Therefore, attempt sparse DMD or use some coherency measure from
comparison with the POD modes (if we can figure out what
@schmid-etal2011 means by this).


### Data ensemble

Stack the snapshot data along the data axis:

```python
des = np.vstack(to_snaps(r.Uf_[good_slice]) for r in runs)
demodes, derv, den = mr.compute_DMD_matrices_snaps_method(des, range(10))

# reshape to [ensembles, modes, iz, it]
iz, ix, it = eu.shape
rdemodes = [a.reshape(iz, it, -1).transpose((2, 0, 1)) 
                for a in np.vsplit(demodes.A, len(runs))]
```

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

import modred as mr

import gc_turbulence as g
import sparse_dmd

r = g.ProcessedRun(cache_path=g.default_processed + 'r13_12_16a.hdf5')


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

uf_ = r.Uf_[complement(find_nan_slice(r.Uf_[:]))]
uf = r.Uf[complement(find_nan_slice(r.Uf[:]))]

suf = sparse_dmd.SparseDMD.to_snaps(uf, decomp_axis=1)
dmd = sparse_dmd.SparseDMD(suf)

gammaval = np.logspace(0, 5, 200)
dmd.compute_dmdsp(gammaval)

dmd.compute_sparse_reconstruction(Ni=75, data=uf, decomp_axis=1)
d = dmd.reconstruction.data
rd = dmd.reconstruction.rdata

u_levels = np.linspace(-0.1, 0.03)

fig, ax = plt.subplots()
for i in range(109):
    ax.contourf(rd[:, i, :], levels=u_levels)
    plt.draw()
    fig.savefig('plots/reconstruct_%s.png' % i)
    fig.clf()
    print i
    a2.contourf(d[:, i, :], levels=u_levels)
    plt.draw()
    fig.savefig('plots/original_%s.png' % i)
    print i
    fig.clf()
```
