"""Functions that directly relate to standing waves."""

import numpy as np

import scipy.interpolate as interp


def kn(n, L):
    """Return the wavenumber for a given mode of standing wave."""
    return np.pi * n / L


def standing_frequency(n=None, k=None, g=9.81, H=1):
    """Standing wave frequency for a given mode number or
    wave number.
    """
    if n is None and k is None:
        return
    elif k is None:
        k = kn(n)
    w2 = g * k * np.tanh(k * H)
    return np.sqrt(w2) / (2 * np.pi)


def vertical_u(z, k, H=0.25):
    """Vertical profile of horizontal velocity component."""
    return np.cosh(k * z) / np.sinh(k * H)


def vertical_w(z, k, H=0.25):
    """Vertical profile of vertical velocity component."""
    return np.sinh(k * z) / np.sinh(k * H)


class StandingWaves(object):
    def __init__(self, L=5.50, H=0.25, g=9.81, z=None,
                 freqs=None, fmax=20, fres=500):
        self.L = L
        self.H = H
        self.g = g

        self.z = z or np.linspace(0, 1)
        self.freqs = freqs or np.linspace(0, fmax, fres)

    def standing_frequency(self, n=None, k=None, scaling=1.006):
        return scaling * standing_frequency(n=n,
                                            k=k,
                                            scaling=scaling,
                                            g=self.g,
                                            H=self.H)

    def vertical_u(self, z, k):
        return vertical_u(z, k, H=self.H)

    def vertical_w(self, z, k):
        return vertical_w(z, k, H=self.H)

    def theoretical_vertical_profile(self, z, k, component='u', norm='max'):
        """Compute the vertical profile for given z, k for
        given component of velocity (either 'u' or 'w'), and
        with optional normalisation to the max (default) or
        mean of the profile or a scalar.

        k, z can be scalar or vector

        Returns the vertical profile over [z, k]
        """
        k = np.atleast_2d(k)
        z = np.atleast_2d(z).T

        if component == 'u':
            v = self.vertical_u(z, k)
        elif component == 'w':
            v = self.vertical_w(z, k)

        if norm == 'max':
            return v / v.max(axis=0, keepdims=True)
        elif norm == 'mean':
            return v / v.mean(axis=0, keepdims=True)
        elif type(norm) in (float, int):
            return v / norm
        else:
            return v

    def create_k_interpolator(self, with_zero=True):
        """Create a lookup function that finds k as a function of
        frequency."""
        k = np.hstack(([0], np.logspace(-3, 4.5, 100)))
        return interp.interp1d(self.standing_frequency(k=k), k)

    @property
    def k_f(self):
        """Wavenumber (radial) as a function of frequency (non-radial)."""
        if not hasattr(self, '_k_f'):
            self._k_f = self.create_k_interpolator()
        return self._k_f

    def freq_vertical_profile(self, component='u', freqs=None, norm='max'):
        """For a given velocity component ('u', 'w'), create a function
        that returns the vertical profile of that component for given
        z, f.
        """
        lfreqs = freqs or self.freqs[1:]
        freqs = freqs or self.freqs
        # Can't actually evaluate vertical profile at f=0,
        # so remove this and tack on the limit (approaching 1 for u)
        profile = self.theoretical_vertical_profile(z=self.z,
                                                    k=self.k_f(lfreqs),
                                                    component=component,
                                                    norm=norm)

        if component == 'u':
            limit = np.ones(self.z.size)
        elif component == 'w':
            limit = self.z / self.H

        if norm == 'max':
            limit /= limit.max()
        elif norm == 'mean':
            limit /= limit.mean()
        else:
            limit /= norm

        profile = np.hstack((limit[:, None], profile))

        return interp.RectBivariateSpline(self.z, freqs, profile)

    def uprofile(self):
        if not hasattr(self, '_uprofile'):
            self._uprofile = self.create_vertical_profile('u', norm='max')
        return self._uprofile

    def wprofile(self):
        if not hasattr(self, '_wprofile'):
            self._wprofile = self.create_vertical_profile('w', norm='max')
        return self._wprofile
