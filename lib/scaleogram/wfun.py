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

DEFAULT_WAVELET = 'cmor1-1.5'

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
    
    if isinstance(wavelet, str):
        wavelet = pywt.ContinuousWavelet(wavelet)
    assert isinstance(wavelet, pywt.ContinuousWavelet)
    
    # the following code has been extracted from pywt.cwt() 1.0.2
    precision = 10
    int_psi, x = pywt.integrate_wavelet(wavelet, precision=precision)
    step = x[1] - x[0]
    j = np.floor(
            np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step))
    if np.max(j) >= np.size(int_psi):
                j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
    
    return int_psi[j.astype(np.int)]
    



def plot_wav_time(wav=DEFAULT_WAVELET, real=True, imag=True,
                  figsize=None, ax=None):
    """Plot wavelet representation in **time domain**
    see ``plot_wav()`` for parameters.
    """
    # build wavelet time domain signal
    if isinstance(wav, str):
        try:
            wav = pywt.ContinuousWavelet(wav)
        except Exception as e:
            raise ValueError("the wav parameter mus be a continuous wavelet"+ \
                             " (see docstring). pywt returns: "+str(e))
    fun_wav, time = wav.wavefun()
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # tme domain plot
    if real:
        ax.plot(time, fun_wav.real, label="real")
    if imag:
        ax.plot(time, fun_wav.imag, "r-", label="imag")
    ax.legend()
    ax.set_title(wav.family_name)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel("Amplitude")
    
    return ax
    

def plot_wav_freq(wav=DEFAULT_WAVELET, figsize=None, ax=None, yscale='linear'):
    """Plot wavelet representation in **frequency domain**
    see ``plot_wav()`` for parameters.
    """
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
    ax.set_xlabel("Frequency [Hz]")

    # useful annotations
    central_frequency = wav.center_frequency
    if not central_frequency:
        central_frequency = pywt.central_frequency(wav)
    bandwidth_frequency = wav.bandwidth_frequency
    #if not bandwidth_frequency:
    #    # The wavlet does not provide the value, let's infer it!
    #    # bandwith defined here from the half of the peak value
    #    w = np.where(fft[0:int(len(fft)/2)]/fft.max() >= 0.5)
    #    bandwidth_frequency = 2* (freq[w].max()-freq[w].min())
    ax.set_yscale(yscale)
    ax.set_ylim(ax.get_ylim())
    ax.plot(central_frequency*np.ones(2), ax.get_ylim())
    ax.annotate("central_freq=%0.1fHz\nbandwidth=%0.1fHz" % (
                central_frequency, bandwidth_frequency), 
                xy=(central_frequency, 0.5), 
                xytext=(central_frequency+2, 0.6),
                arrowprops=dict(facecolor='black', shrink=0.01))

    return ax


def plot_wav(wav=DEFAULT_WAVELET, figsize=None, axes=None, 
             real=True, imag=True, yscale='linear'):

    # build wavelet time domain signal
    if isinstance(wav, str):
        try:
            wav = pywt.ContinuousWavelet(wav)
        except Exception as e:
            raise ValueError("the wav parameter mus be a continuous wavelet"+ \
                             " (see docstring). pywt returns: "+str(e))
    fun_wav, time = wav.wavefun(level=8)

    if axes is None:
        fig, (ax1, ax2)= plt.subplots(1, 2, figsize=figsize)
    else:
        ax1, ax2 = axes
    
    plot_wav_time(wav, real=real, imag=imag, ax=ax1)
    plot_wav_freq(wav, yscale=yscale, ax=ax2)

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



if __name__ == '__main__':
    plot_wav()
    plt.draw()
    plt.show()
    