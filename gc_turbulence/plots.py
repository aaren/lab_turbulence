import numpy as np

import matplotlib.pyplot as plt

import scipy.signal as sig


def plot_smooth_fft(signal, **kwargs):
    window = kwargs.pop('window', 'hanning')
    size = kwargs.pop('size', 2**15)
    xlim = kwargs.pop('xlim', (0.05, 2.5))
    ylim = kwargs.pop('ylim', (0, 5))
    ax = kwargs.pop('ax', None) or plt.gca()

    window = sig.get_window(window, signal.size, fftbins=False)

    fft = np.fft.rfft(signal * window, size)
    freqs = np.fft.rfftfreq(size, d=0.01)
    power = np.abs(fft)

    ax.semilogx(freqs, power, **kwargs)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def contour_fft(signal, x=None, axis=-1, size=2**15, window='hanning',
                xlim=(0.05, 2.5), levels=(0, 1, 50), ax=None):
    ax = ax or plt.gca()
    window = sig.get_window(window, signal.shape[axis], fftbins=False)

    fft = np.fft.rfft(signal * window, size, axis=axis)
    freqs = np.fft.rfftfreq(size, d=0.01)

    lvl = np.linspace(*levels)
    if x is None:
        x = np.arange(signal.shape[0])

    Ff, Xf = np.meshgrid(freqs, x)

    which = (freqs > xlim[0]) & (freqs < xlim[1])
    F = Ff[:, which]
    X = Xf[:, which]

    return ax.contourf(F, X, np.abs(fft)[:, which], levels=lvl)


def plot_total_fft(signal, xlim=(0.01, 3), ylim=(0, 10), window='hanning',
                   size=2**15, ax=None):
    ax = ax or plt.gca()
    window = sig.get_window(window, signal.shape[-1], fftbins=False)

    rfft = np.fft.rfft(signal * window, n=size, axis=-1)
    freqs = np.fft.rfftfreq(size, d=0.01)

    ax.plot(freqs, np.abs(rfft).mean(axis=0))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def z_fft(signal, z=None, mean=False, axis=-1, size=2**15, window='hanning',
          xlim=(0.05, 5), levels=(0, 2, 50)):
    window = sig.get_window(window, signal.shape[axis], fftbins=False)

    fft = np.fft.rfft(signal * window, size, axis=axis)
    freqs = np.fft.rfftfreq(size, d=0.01)

    lvl = np.linspace(*levels)
    if z is None:
        z = np.arange(signal.shape[0])

    Ff, Zf = np.meshgrid(freqs, z)

    which = (freqs > xlim[0]) & (freqs < xlim[1])
    F = Ff[:, which]
    Z = Zf[:, which]

    abs_fft = np.abs(fft).mean(axis=1)

    if mean:
        abs_fft /= abs_fft.mean(axis=0)

    plt.contourf(F, Z, abs_fft[:, which], levels=lvl)
