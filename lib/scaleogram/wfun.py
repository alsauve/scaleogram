# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 2019

@author: Alexandre Sauve

Content :
    Helper for wavelet related functions
"""

import pywt
import numpy as np
import matplotlib.pyplot as plt
import six

DEFAULT_WAVELET = 'cmor1-1.5'


def set_default_wavelet(wavelet):
    """Sets the default wavelet to be used for scaleograms"""
    global DEFAULT_WAVELET
    DEFAULT_WAVELET =  wavelet

def get_default_wavelet():
    """Sets the default wavelet to be used for scaleograms"""
    global DEFAULT_WAVELET
    return DEFAULT_WAVELET


def periods2scales(periods, wavelet=None, dt=1.0):
    """Helper function to convert periods values (in the pseudo period
    wavelet sense) to scales values

    Arguments
    ---------
    - periods : np.ndarray() of positive strictly increasing values
        The ``periods`` array should be consistent with the ``time`` array passed
        to ``cws()``. If no ``time`` values are provided the period on the
        scaleogram will be in sample units.

        Note: you should check that periods minimum value is larger than the
        duration of two data sample because the sectrum has no physical
        meaning bellow these values.

    - wavelet : pywt.ContinuousWavelet instance or string name

    dt=[1.0] : specify the time interval between two samples of data
        When no ``time`` array is passed to ``cws()``, there is no need to
        set this parameter and the default value of 1 is used.

    Note: for a scale value of ``s`` and a wavelet Central frequency ``C``,
    the period ``p`` is::

        p = s / C

    Example : Build a spectrum  with constant period bins in log space
    -------
    import numpy as np
    import scaleogram as scg

    periods = np.logspace(np.log10(2), np.log10(100), 100)
    wavelet = 'cgau5'
    scales  = periods2scales(periods, wavelet)
    data    = np.random.randn(512)  # gaussian noise
    scg.cws( data, scales=scales, wavelet=wavelet, yscale='log',
            title="CWT of gaussian noise with constant binning in Y logscale")
    """
    if wavelet is None:
        wavelet = get_default_wavelet()
    if isinstance(wavelet, six.string_types):
        wavelet = pywt.ContinuousWavelet(wavelet)
    else:
        assert(isinstance(wavelet, pywt.ContinuousWavelet))

    return (periods/dt) * pywt.central_frequency(wavelet)



def get_wavlist():
    """Returns the list of continuous wavelet functions available in the
    PyWavelets library.
    """
    l = []
    for name in pywt.wavelist(kind='continuous'):
        # supress warnings when the wavelet name is missing parameters
        completion = {
            'cmor': 'cmor1.5-1.0',
            'fbsp': 'fbsp1-1.5-1.0',
            'shan': 'shan1.5-1.0' }
        if name in completion:
            name =  completion[name]# supress warning
        l.append( name+" :\t"+pywt.ContinuousWavelet(name).family_name )
    return l

WAVLIST = get_wavlist()



def child_wav(wavelet, scale):
    """Returns an array of complex values with the child wavelet used at the
    given ``scale``.

    The ``wavelet`` argument can be either a string like 'cmor1-1.5' or
    a ``pywt.ContinuousWavelet`` instance
    """

    wavelet = _wavelet_instance(wavelet)

    # the following code has been extracted from pywt.cwt() 1.0.2
    precision = 10
    int_psi, x = pywt.integrate_wavelet(wavelet, precision=precision)
    step = x[1] - x[0]
    j = np.floor(
            np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step))
    if np.max(j) >= np.size(int_psi):
                j = np.delete(j, np.where((j >= np.size(int_psi)))[0])

    return int_psi[j.astype(np.int)]



def _wavelet_instance(wavelet):
    """Function responsible for returning the correct pywt.ContinuousWavelet
    """
    if isinstance(wavelet, pywt.ContinuousWavelet):
        return wavelet
    if isinstance(wavelet, six.string_types):
        return pywt.ContinuousWavelet(wavelet)
    else:
        raise ValueError("Expecting a string name for the wavelet,"+
                         " or pywt.ContinuousWavelet. Got: "+str(wavelet))


def fastcwt(data, scales, wavelet, sampling_period=1.0, method='auto'):
    """
    Compute the continuous wavelet transform (CWT) and has the same signature
    as ``pywt.cwt()`` but is faster for large signals length and scales.

    Parameters
    ----------
    signal : array to compute the CWT on
    
    scales: dilatation factors for the CWT
    
    wavelet: wavelet name or pywt.ContinuousWavelet

    method=['auto'] | 'conv' | 'fft' for selecting the convolution method
        the `'auto'` keyword switch automatically to the best complexity at each
        scales. While the `'fft'` and `'conv'` uses `numpy.fft` and `numpy.conv`
        respectively.

    In practice the `'fft'` method is implemented by using the convolution 
    theorem which states::
        convolve(wav,sig) == ifft(fft(wav)*fft(sig))

    Zero padding is adjusted to keep at bay circular convolution side effects.


    Example::

        %time (coef1, freq1) = fastcwt(np.arange(140000), np.arange(2,200), 'cmorl1-1')
        => CPU times: user 12.6 s, sys: 2.2 s, total: 14.8 s
        => Wall time: 14.9 s

        %time (coef1, freq1) = pywt.cwt(np.arange(140000), np.arange(2,200), 'cmorl1-1')
        => CPU times: user 1min 50s, sys: 401 ms, total: 1min 51s
        => Wall time: 1min 51s
    """

    # accept array_like input; make a copy to ensure a contiguous array
    data = np.array(data)
    if not isinstance(wavelet, (pywt.ContinuousWavelet, pywt.Wavelet)):
        wavelet = pywt.DiscreteContinuousWavelet(wavelet)
    if np.isscalar(scales):
        scales = np.array([scales])
    dt_out = None  # currently keep the 1.0.2 behaviour: TODO fix in/out dtype consistency
    if data.ndim == 1:
        if wavelet.complex_cwt:
            dt_out = complex
        out = np.zeros((np.size(scales), data.size), dtype=dt_out)
        precision = 10
        int_psi, x = pywt.integrate_wavelet(wavelet, precision=precision)
        
        if method in ('auto', 'fft'):
            # - to be as large as the sum of data length and and maximum wavelet
            #   support to avoid circular convolution effects
            # - additional padding to reach a power of 2 for CPU-optimal FFT
            size_pad = lambda s: 2**np.int(np.ceil(np.log2(s[0] + s[1])))
            size_scale0 = size_pad( (len(data), 
                                     np.take(scales, 0) * ((x[-1] - x[0]) + 1)) )
            fft_data = None
        elif not method == 'conv':
            raise ValueError("method must be in: 'conv', 'fft' or 'auto'")

        for i in np.arange(np.size(scales)):
            step = x[1] - x[0]
            j = np.floor(
                np.arange(scales[i] * (x[-1] - x[0]) + 1) / (scales[i] * step))
            if np.max(j) >= np.size(int_psi):
                j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
            int_psi_scale = int_psi[j.astype(np.int)][::-1]
               
            if method == 'conv':
                conv = np.convolve(data, int_psi_scale)
            else:
                size_scale = size_pad( (len(data), len(int_psi_scale)) )
                if size_scale != size_scale0:
                    # the fft of data changes when padding size changes thus
                    # it has to be recomputed
                    fft_data = None
                size_scale0 = size_scale
                nops_conv = len(data) * len(int_psi_scale)
                nops_fft  = (2+(fft_data is None)) * size_scale * np.log2(size_scale)
                if (method == 'fft') or ((method == 'auto') and (nops_fft < nops_conv)):
                    if fft_data is None:
                        fft_data = np.fft.fft(data, size_scale)
                    fft_wav = np.fft.fft(int_psi_scale, size_scale)
                    conv = np.fft.ifft(fft_wav*fft_data)
                    conv = conv[0:len(data)+len(int_psi_scale)-1]
                else:
                    conv = np.convolve(data, int_psi_scale)
                
            coef = - np.sqrt(scales[i]) * np.diff(conv)
            if not np.iscomplexobj(out):
                coef = np.real(coef)
            d = (coef.size - data.size) / 2.
            if d > 0:
                out[i, :] = coef[int(np.floor(d)):int(-np.ceil(d))]
            elif d == 0.:
                out[i, :] = coef
            else:
                raise ValueError(
                    "Selected scale of {} too small.".format(scales[i]))
        frequencies = pywt.scale2frequency(wavelet, scales, precision)
        if np.isscalar(frequencies):
            frequencies = np.array([frequencies])
        for i in np.arange(len(frequencies)):
            frequencies[i] /= sampling_period
        return out, frequencies
    else:
        raise ValueError("Only dim == 1 supported")




def plot_wav_time(wav=None, real=True, imag=True,
                  figsize=None, ax=None, legend=True, clearx=False):
    """Plot wavelet representation in **time domain**
    see ``plot_wav()`` for parameters.
    """
    if wav is None:
        wav = get_default_wavelet()
    wav  = _wavelet_instance(wav)
    fun_wav, time = wav.wavefun()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # tme domain plot
    if real:
        ax.plot(time, fun_wav.real, label="real")
    if imag:
        ax.plot(time, fun_wav.imag, "r-", label="imag")
    if legend:
        ax.legend()
    ax.set_title(wav.name)
    if clearx:
        ax.set_xticks([])
    else:
        ax.set_xlabel('Time (s)')
    #ax.set_ylabel("Amplitude")
    ax.set_ylim(-1, 1)


    return ax


def plot_wav_freq(wav=None, figsize=None, ax=None, yscale='linear',
                  annotate=True, clearx=False):
    """Plot wavelet representation in **frequency domain**
    see ``plot_wav()`` for parameters.
    """

    if wav is None:
        wav = get_default_wavelet()
    wav  = _wavelet_instance(wav)
    fun_wav, time = wav.wavefun()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # frequency domain plot
    df   = 1 / (time[-1]-time[0])
    freq = np.arange(len(time)) * df
    fft  = np.abs(np.fft.fft(fun_wav)/np.sqrt(len(fun_wav)))
    ax.plot(freq, fft)
    #ax2.set_yscale('log')
    ax.set_xlim(0, df*len(freq)/2)
    ax.set_title("Frequency support")
    if clearx:
        ax.set_xticks([])
    else:
        ax.set_xlabel("Frequency [Hz]")

    ax.set_yscale(yscale)
    ax.set_ylim(-0.1, 1.1)

    central_frequency = wav.center_frequency
    if not central_frequency:
        central_frequency = pywt.central_frequency(wav)
    bandwidth_frequency = wav.bandwidth_frequency if wav.bandwidth_frequency else 0
    ax.plot(central_frequency*np.ones(2), ax.get_ylim())

    if annotate:
        ax.annotate("central_freq=%0.1fHz\nbandwidth_param=%0.1f" % (
                    central_frequency, bandwidth_frequency),
                    xy=(central_frequency, 0.5),
                    xytext=(central_frequency+2, 0.6),
                    arrowprops=dict(facecolor='black', shrink=0.01))

    return ax


def plot_wav(wav=None, figsize=None, axes=None,
             real=True, imag=True, yscale='linear',
             legend=True, annotate=True, clearx=False):

    if wav is None:
        wav = get_default_wavelet()
    wav  = _wavelet_instance(wav)
    fun_wav, time = wav.wavefun()

    if axes is None:
        fig, (ax1, ax2)= plt.subplots(1, 2, figsize=figsize)
    else:
        ax1, ax2 = axes

    plot_wav_time(wav, real=real, imag=imag, ax=ax1, legend=legend, clearx=clearx)
    plot_wav_freq(wav, yscale=yscale, ax=ax2, annotate=annotate, clearx=clearx)

    return ax1, ax2



plot_wav.__doc__ =     """
Quick helper function to check visually the properties of a wavelet
in time domain and the filter view in frequency domain.


Parameters
----------

- wav : continuous wavelate name or pywt.ContinuousWavelet
    If not provided, then the default wavelet for ``cws()`` is used.

- axes= (ax1, ax2) : allow to customize the plot destinations

- figsize=(width,eight) : forward the size (inches) to matplotlib
    If this parameter is provided, a new figure is created under the hood
    for display. It is only used if axes is absent

- real= [True]/False : plot real part in time domain

- imag= [True]/False : plot imaginary part in time domain

- yscale=['linear']|'log' allow to select Y axis scale in frequency domain

Returns
-------
- ax1, ax2 : matplotlib graphics elements


Continuous Wavelet list
-----------------------
- """+("\n- ".join(WAVLIST))





def plot_wavelets(wavlist=None, figsize=None):
    """Plot the matrix of all available continuous wavelets
    """

    wavlist = WAVLIST if wavlist is None else wavlist
    names = [ desc.split()[0] for desc in wavlist ]
    ncol = 4
    nrow = int((len(names)+1)/2)
    figsize = (12, 1.5*nrow) if figsize is None else figsize
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    plt.subplots_adjust( hspace=0.5, wspace=0.25 )
    axes = [ item for sublist in axes for item in sublist ] # flatten list

    for i in range(int(len(names))):
        plot_wav(names[i], axes=(axes[i*2], axes[i*2+1]),
                 legend=False, annotate=False, clearx=True)





#if __name__ == '__main__':
#    plot_wav()
#    plot_wavelets()
#    plt.draw()
#    plt.show()


