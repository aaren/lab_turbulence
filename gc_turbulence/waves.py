import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fftpack as fftpack
import scipy.optimize as opt
import scipy.interpolate as interp

from standing_waves import StandingWaves

from .filters import dct_waves, brickpass


def find_nearest_idx(array, value):
    return (np.abs(array - value)).argmin()


def find_closest_peaks(power, freqs, guess_freqs):
    """Given a power spectrum over frequencies `freqs`
    find the array indices of the power spectrum peaks
    closest to the guess_freqs.
    """
    # find the maxima in the power spectrum
    maxima = sig.argrelextrema(power, np.greater)

    maxima_freqs = np.zeros(freqs.shape)
    maxima_freqs[maxima] = freqs[maxima]

    # determine the peaks as the closest maxima to
    # each of the standing wave frequencies
    peak_indices = [find_nearest_idx(maxima_freqs, f) for f in guess_freqs]
    return peak_indices


def construct_wave(amplitude, f, time, decay=0):
    return amplitude * np.exp(2j * np.pi * f * time) * np.exp(decay * time)


def wave_fft(amplitude, f, size, window, dt, decay=0):
    time = np.arange(window.size) * dt
    wave = (2 / window.sum()) \
        * construct_wave(amplitude, f, time, decay=decay) \
        * window
    return np.fft.rfft(wave, size)


def subtract_amplitude_fft(fft, amplitude, f, size, window, decay=0):
    return fft - wave_fft(amplitude, f, size, window, decay=decay)


def minimise_this_decay((amplitude_re, f, amplitude_im, decay),
                        fft, size, window):
    amplitude = np.complex(amplitude_re, amplitude_im)
    return np.linalg.norm(subtract_amplitude_fft(fft, amplitude, f,
                                                 size, window, decay=decay))


def minimise_power_decay(f, fft, freqs, size, window, bounds=None):
    power = np.abs(fft)
    peak = find_closest_peaks(power, np.array(freqs),
                              guess_freqs=np.array([f]))
    idx = np.abs(freqs - f).argmin()

    idx = peak

    amplitude = power[idx]
    # get phase from the fft (assume it doesn't change)
    #  phase = np.angle(fft[idx])

    amplitude = fft[idx]

    if bounds is not None:
        damp = np.abs(amplitude) * 0.3
        bounds = [(amplitude.real - damp, amplitude.real + damp),
                  bounds,
                  (amplitude.imag - damp, amplitude.imag + damp),
                  (-0.02, -0.002)]
    minim = opt.minimize(minimise_this_decay,
                         # method='TNC',
                         x0=(amplitude.real,
                             freqs[peak],
                             amplitude.imag,
                             -0.007),
                         args=(fft, size, window),
                         bounds=bounds)

    opt_amplitude_re = minim.x[0]
    opt_freq = minim.x[1]
    opt_amplitude_im = minim.x[2]
    opt_decay = minim.x[3]

    camplitude = np.complex(opt_amplitude_re, opt_amplitude_im)

    return camplitude, opt_freq, opt_decay, wave_fft(camplitude, opt_freq,
                                                     size, window, opt_decay)


class WaveExtractor(object):
    def __init__(self, component, z, x):
        self.component = component
        self.waves = StandingWaves(z=z)
        self.x = x
        self.z = z

        self.dt = 0.01

    def extract_waves(self, data, nf, component='w', length=8000,
                      size=2**15, window='hanning',
                      plots=False, decay=True, vertical=True,
                      dct_index=3):
        freqs = np.fft.rfftfreq(size, d=self.dt)
        window = sig.get_window(window, data.shape[-1], fftbins=False)

        fft = np.fft.rfft(data * window, n=size, axis=-1)

        result = self.comb_frequencies_decay(fft.mean(axis=0), nf=nf,
                                             window=window,
                                             size=size,
                                             plots=plots)
        amplitudes, frequencies, tdecay, peak_ffts = result

        iamp = interp.RectBivariateSpline(self.x, freqs, np.abs(fft))
        amplitude_variation = iamp(self.x, frequencies[:nf])

        if plots:
            plt.figure()
            plt.plot(self.x, amplitude_variation, '--')

        dct = fftpack.dct(amplitude_variation, axis=0, norm='ortho')
        dct[dct_index:, :] = 0
        amplitude_variation = fftpack.idct(dct, axis=0, norm='ortho')

        if plots:
            plt.plot(self.x, amplitude_variation)

        # scale the amplitude variation to its mean
        amplitude_variation /= amplitude_variation.mean(axis=0)

        full_amplitudes = amplitude_variation * amplitudes[None, :]

        if not decay:
            tdecay = np.zeros(tdecay.size)

        time = np.arange(length) * self.dt

        # create waves with axes [z, x, t, k]
        # only varies over [x, t, k] at this point
        waves = construct_wave(full_amplitudes[None, :, None, :],
                               frequencies[None, None, None, :nf],
                               time=time[None, None, :, None],
                               decay=tdecay[None, None, None, :nf])
        waves = (2 / window.sum()) * waves.real

        if vertical:
            # compute vertical profile of computed frequencies
            # and extend the waves over that profile
            k = self.waves.k_f(frequencies)
            profile = self.waves.theoretical_vertical_profile(self.z, k,
                                                              norm='mean',
                                                              component=component)
            return profile[:, None, None, :] * waves

        else:
            # remove the empty z axis, returning [x, t, k]
            return waves.squeeze()

    def get_zxwaves(self, signal, component='u', cutoff=0.6,
                    bandpass=brickpass, flo=0.6, fhi=3):
        # create waves that capture the x variance from a mean over z
        xwaves = dct_waves(n=2, signal=signal.mean(axis=0),
                           cutoff=cutoff, bandpass=bandpass)

        # transform to frequency domain
        xfft = np.fft.rfft(xwaves, axis=-1)
        freqs = np.fft.rfftfreq(xwaves.shape[-1], d=self.dt)

        # indices of frequency limits
        ihi = np.abs(freqs - fhi).argmin()
        ilo = np.abs(freqs - flo).argmin()

        # normalise to *mean* of vertical so that we can directly
        # multiply the fft (which comes from a vertical mean)
        vertical_profile = self.waves.freq_vertical_profile(component=component,
                                                            norm='mean')

        # extend the fft uniformly over vertical
        zxfft = xfft[None, :, :] * np.ones((self.z.size, 1, 1))

        # scale the fft by the theoretical vertical profile, but only
        # in frequency range that matters (don't want to scale the
        # non-wave components that form the boundaries of the signal).
        # In principle we should care about the sharp edges here, but
        # it doesn't seem to have a significant effect.
        zxfft[:, :, ilo:ihi] = zxfft[:, :, ilo:ihi] \
            * vertical_profile(self.z, freqs)[:, None, ilo:ihi]

        # transform back to time domain
        return np.fft.irfft(zxfft, axis=-1)

    def comb_frequencies_decay(self, fft, nf, window, size,
                               df=0.01, plots=False):
        freqs = np.fft.rfftfreq(size, d=self.dt)

        amplitudes = np.zeros(nf, np.complex)
        frequencies = np.zeros(nf, np.float)
        decay = np.zeros(nf, np.float)
        peak_ffts = np.zeros((nf, freqs.size), np.complex)

        if plots:
            plt.figure()
            plt.plot(freqs, np.abs(fft), 'k', linewidth=3, alpha=0.5)
        # go through each guess frequency and find the nearest wave
        # amplitude (complex) and frequency that would minimise the
        # power spectrum, subtracting the optimal wave each time.
        N = np.arange(1, nf + 1)
        standing_frequencies = self.waves.standing_frequency(n=N)
        for i, f in enumerate(standing_frequencies):
            result = minimise_power_decay(f, fft, freqs,
                                          size, window,
                                          bounds=(f - df, f + df))
            amplitudes[i], frequencies[i], decay[i], peak_ffts[i] = result
            fft = fft - peak_ffts[i]

            if plots:
                plt.plot(freqs, np.abs(peak_ffts[i]))
                plt.plot(frequencies[i], np.abs(amplitudes[i]), 'o')
                plt.annotate(round(decay[i], 4),
                             xy=(frequencies[i], np.abs(amplitudes[i])),
                             xytext=(frequencies[i] / (1.2 - 0.05),
                                     1 - 0.1 * (i + 1)),
                             textcoords='axes fraction',
                             arrowprops=dict(facecolor='black', shrink=0.05))
                plt.xlim(0.05, 1.2)

        if plots:
            plt.plot(freqs, np.abs(fft))

        return amplitudes, frequencies.real, decay, peak_ffts
