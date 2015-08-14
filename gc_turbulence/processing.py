from functools import partial
import logging

import numpy as np
import h5py
import scipy.ndimage as ndi

from . import transform
from .attributes import ProcessorAttributes, ProcessedAttributes
from .inpainting import Inpainter
from .filters import butterpass
from .runbase import H5Cache
from .waves import WaveExtractor


class PreProcessor(ProcessorAttributes, H5Cache):
    """Apply basic pre processing to raw Dynamic Studio output to make
    it usable in analysis.

    Usage:

        r = RawRun
        pp = PreProcessor(r)
        pp.execute()
        pp.write_data(hdf5_cache_path)

    Stages:

        - transform X, Z to lock / base relative.
        - extract valid region
        - interpolation of zero values (replace with nan?)
        - write to hdf5

    N.B. You won't be able to write anything until you
    have run `pp.execute()`.

    There are more methods on this class (for each of the
    steps in execute) but if they are run out of order you
    will probably get weird results.
    """
    def __init__(self, run):
        self.run = run
        self.run.load()
        self.has_executed = False

    def execute(self):
        """Execute the pre-processing steps in the
        right order. Can't write data until this has
        been done.
        """
        steps = ['extract_valid_region',
                 'filter_zeroes',
                 'interpolate_nan',
                 'transform_to_lock_relative',  # TODO: put this first
                 ]

        for step in steps:
            logging.info('starting {}'.format(step))
            processor = getattr(self, step)
            processor()
            logging.info('finished {}'.format(step))

        self.has_executed = True

    def transform_to_lock_relative(self):
        """This method changes the X and Z coordinates such that
        they have their origin at the lock gate and the base of
        the tank, respectively.
        """
        self.X = self.X + self.horizontal_offset
        self.Z = self.Z + self.vertical_offset

    def extract_valid_region(self):
        """Extract the valid data region from the run and
        convert to SI units.
        """
        valid = self.compute_valid_slice()

        r = self.run

        self.X = r.x[valid] / 1000
        self.Z = r.z[valid] / 1000
        self.T = r.t[valid]
        self.U = r.u[valid]
        self.V = r.v[valid]
        self.W = r.w[valid]

    def compute_valid_slice(self):
        """Determine the slice to be used to pull out the valid
        region."""
        # find the indices that correspond to the rectangular view
        # that we are going to take into the data
        x_min, x_max = self.valid_region_xlim
        z_min, z_max = self.valid_region_ylim

        X = self.run.x[:, :, 0] / 1000
        Z = self.run.z[:, :, 0] / 1000

        valid = (X > x_min) & (X < x_max) & (Z > z_min) & (Z < z_max)
        iz, ix = np.where(valid)

        ix_min, ix_max = ix.min(), ix.max()
        iz_min, iz_max = iz.min(), iz.max()

        # valid region in x, z
        valid_slice = np.s_[iz_min: iz_max, ix_min: ix_max, :]

        return valid_slice

    def filter_zeroes(self):
        """Set all velocities that are identically zero to be nan."""
        self.U[self.U == 0] = np.nan
        self.V[self.V == 0] = np.nan
        self.W[self.W == 0] = np.nan

    def filter_anomalies(self):
        """Find anomalous data and set to nan.

        You should run this either before filter_zeroes or after
        interpolate_nan, or you get lots more nans.

        After running this you should run interpolate_nan.
        """
        # TODO: write me!
        smoothed = ndi.uniform_filter(self.U, size=3)
        thresh = 0.05  # TODO: set more generally
        bad = np.abs(self.U - smoothed) > thresh
        self.U[bad] = np.nan

    def interpolate_nan(self, sub_region=None, scale='auto'):
        """The raw data contains regions with velocity identical
        to zero. These are non physical and can be removed by
        interpolation.
        """
        if scale == 'auto':
            scale = transform.front_speed(self)

        inpainter = Inpainter(self, sub_region=sub_region, scale=scale)
        inpainter.paint(processors=12)

    def non_dimensionalise(self):
        """Take the original, dimensional run data, divide by length
        / time scales and resample to get non-dimensionalised data
        on a regular grid.

        NB. This method is gregarious with the data. It will take
        all of the run data and non-dim. It will not restrict to a
        particular regular grid, so you won't be able to stack
        multiple runs directly if they occupy different volumes in
        non-dimensional space (which they will if they have
        different parameters).
        """
        # FIXME: work with two layer runs
        p = self.run.attributes

        # determine the scaling factors
        L = p['L']  # length
        H = p['H']  # height

        # acceleration (reduced gravity)
        g_ = 9.81 * (p['rho_lock'] - p['rho_ambient']) / p['rho_ambient']

        U = (g_ * H) ** .5  # speed
        T = H / U  # time

        # Sampling intervals in dim space. These shouldn't vary run
        # to run but they might. Maybe add this as a sanity check
        # for each run?
        dz = np.diff(self.Z[:2, 0, 0])[0]
        dx = np.diff(self.X[0, :2, 0])[0]
        dt = np.diff(self.T[0, 0, :2])[0]

        # Sampling intervals in non dim space. These are set here to
        # be constant across all runs. They were roughly determined
        # by doubling the non-dimensionalised intervals of the
        # fastest / tallest run (so that we don't try and oversample
        # anything).
        # TODO: define this elsewhere (class attribute?)
        dx_ = 0.01
        dz_ = 0.012
        dt_ = 0.015

        # as well as scaling the quantities, we have to scale the
        # sampling interval. The dimensional non dim interval is
        # dt_ * T; the dimensional interval is dt.
        zoom_factor = (dz / (H * dz_),
                       dx / (L * dx_),
                       dt / (T * dt_))

        zoom_kwargs = {'zoom':  zoom_factor,
                       'order': 1,          # spline interpolation.
                       'mode': 'constant',  # points outside the boundaries
                       'cval': np.nan,      # are set to np.nan
                       }

        self.Z_ = ndi.zoom(self.Z[:] / H, **zoom_kwargs)
        self.X_ = ndi.zoom(self.X[:] / L, **zoom_kwargs)
        self.T_ = ndi.zoom(self.T[:] / T, **zoom_kwargs)

        self.U_ = ndi.zoom(self.U[:] / U, **zoom_kwargs)
        self.V_ = ndi.zoom(self.V[:] / U, **zoom_kwargs)
        self.W_ = ndi.zoom(self.W[:] / U, **zoom_kwargs)

        self.Zf_ = ndi.zoom(self.Zf[:] / H, **zoom_kwargs)
        self.Xf_ = ndi.zoom(self.Xf[:] / L, **zoom_kwargs)
        self.Tf_ = ndi.zoom(self.Tf[:] / T, **zoom_kwargs)

        self.Uf_ = ndi.zoom(self.Uf[:] / U, **zoom_kwargs)
        self.Vf_ = ndi.zoom(self.Vf[:] / U, **zoom_kwargs)
        self.Wf_ = ndi.zoom(self.Wf[:] / U, **zoom_kwargs)

    def write_data(self, path):
        """Save everything to a new hdf5."""
        if not self.has_executed:
            print "Data has not been processed! Not writing."
            return

        self.hdf5_write_prep(path)
        h5file = h5py.File(path, 'w')

        for vector in self.save_vectors.names:
            data = getattr(self, vector)
            h5file.create_dataset(vector, data.shape, dtype=data.dtype)
            h5file[vector][...] = data

        for k, v in self.run.attributes.items():
            h5file.attrs[k] = v

        h5file.close()


class ProcessedRun(ProcessedAttributes, H5Cache):
    """Wrapper around a run that has had its data quality controlled."""
    def __init__(self, cache_path=None, forced_load=False):
        """Initialise a processed run.

        cache_path - hdf5 to load from
        forced_load - load hdf5 even its keys aren't the same as vectors
        """
        self.cache_path = cache_path
        if self.cache_path:
            self.init_cache(self.cache_path)
            self.load(force=forced_load)

            self.index = self.attributes['run_index']

            self.init_vectors()

        self.has_executed = False

    def init_vectors(self):
        self.z = self.Z[:, 0, 0]
        self.x = self.X[0, :, 0]
        self.t = self.T[0, 0, :]

    def execute(self):
        """Execute the pre-processing steps in the
        right order. Can't write data until this has
        been done.
        """
        steps = ['init_vectors',
                 'extract_waves',
                 'subtract_waves',
                 'transform',
                 ]

        for step in steps:
            logging.info('starting {}'.format(step))
            processor = getattr(self, step)
            processor()
            logging.info('finished {}'.format(step))

        self.has_executed = True

    def critical_frequency(self):
        """Calculate the frequency below which we expect periodic features
        to not be averaged out by mean subtraction.
        """
        c = self.front_speed
        fcrit = c / (self.x.max() - self.x.min())
        return fcrit

    def get_waves(self, component='u', tmin=0, tmax=3000, vertical=True):
        velocity = getattr(self, component.upper())
        preindex = np.s_[..., tmin:tmax]
        pre_front = velocity[preindex].mean(axis=0)

        full_length = self.t.size - tmin

        extractor = WaveExtractor(component=component, z=self.z, x=self.x)

        pre_waves = extractor.extract_waves(data=pre_front,
                                            nf=4,
                                            component=component,
                                            length=full_length,
                                            vertical=vertical)

        prewaveless = velocity[..., tmin:] \
            - pre_waves.sum(axis=-1)[..., :full_length]

        bandpass = partial(butterpass, order=6)
        zxwaves = extractor.get_zxwaves(prewaveless,
                                        component=component,
                                        cutoff=0.45,
                                        bandpass=bandpass)

        prewaveless = prewaveless - zxwaves
        background = prewaveless[..., :tmax].mean(axis=-1, keepdims=True)

        return pre_waves, zxwaves, background

    def extract_waves(self):
        """Extract the standing waves and set as attributes."""
        self.wU, self.wUr, self.Ubg = self.get_waves(component='u')
        self.wW, self.wWr, self.Wbg = self.get_waves(component='w')

    def subtract_waves(self):
        """Subtract the waves from data and overwrite."""
        self.U = self.U - self.wU.sum(axis=-1) - self.wUr - self.Ubg
        self.W = self.W - self.wW.sum(axis=-1) - self.wWr - self.Wbg

    @property
    def transformer(self):
        return transform.FrontTransformer(self.front_speed,
                                          dx=self.dx,
                                          dt=self.dt)

    @property
    def front_speed(self):
        return transform.front_speed(self)

    def transform(self):
        self.Uf = self.transformer.to_front(self.U) - self.front_speed
        self.Wf = self.transformer.to_front(self.W)

        # index and time of front onset
        lower_streamwise = self.Uf[:20].mean(axis=1).mean(axis=0)
        it0 = np.argmax(lower_streamwise > self.front_speed / 2)

        t0 = self.t[it0]

        # transformed coordinates
        # transformed x is actually in units of time
        self.xf = (self.x - self.x[0]) / self.front_speed
        self.dxf = self.dx / self.front_speed

        self.zf = self.z
        self.tf = self.t - t0
        self.t0 = t0  # (to replace ftf)

        self.Zf, self.Xf, self.Tf = np.meshgrid(self.zf, self.xf, self.tf,
                                                indexing='ij')

    def write_data(self, path):
        """Save everything to a new hdf5."""
        if not self.has_executed:
            print "Data has not been processed! Not writing."
            return

        self.hdf5_write_prep(path)
        h5file = h5py.File(path, 'w')

        for vector in self.save_vectors.names:
            logging.info('writing {}'.format(vector))
            data = getattr(self, vector)
            h5file.create_dataset(vector, data.shape, dtype=data.dtype)
            h5file[vector][...] = data

        for k, v in self.attributes.items():
            h5file.attrs[k] = v

        h5file.close()
