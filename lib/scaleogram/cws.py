# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 2019

@author: Alexandre Sauve


This file provide an easy to use and portable scaleogram plot tool
for wavelet based data analysis using Continuous Wavelet Transform (CWT).

This tool has been designed with in mind:
- Ease of use
- Intuitive options for quality plots using matplotlib conventions
- Portability (python2 / python3)
- Speed : the plot is drawn with pmeshgrid which is resonably fast


Requirements:
    pip install PyWavelet
    pip install matplotlib>=2.0.0


Basic usage:
    from scaleogram import scaleogram
    scaleogram(numpy_array)

"""


from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pywt


DEFAULT_WAVELET = 'cmor1-1.5'

def get_wavlist():
    l = []
    for name in pywt.wavelist(kind='continuous'):
        completion = {
            'cmor': 'cmor1.5-1.0',
            'fbsp': 'fbsp1-1.5-1.0',
            'shan': 'shan1.5-1.0' }
        if name in completion:
            name =  completion[name]# supress warning
        l.append( name+" :\t"+pywt.ContinuousWavelet(name).family_name )
    return l

WAVLIST = get_wavlist()

CBAR_DEFAULTS = {
    'vertical'   : { 'aspect':30, 'pad':0.03, 'fraction':0.05 },
    'horizontal' : { 'aspect':40, 'pad':0.12, 'fraction':0.05 }
}


class CWT:
    """Container for Continuous Wavelet Transform
    Allow to plot several scaleograms with the same transform
    """
    def __init__(self, time, signal=None, scales=None,
                 wavelet=DEFAULT_WAVELET):
        # allow to build the spectrum for signal only
        if signal is None:
            signal = time
            time   = np.arange(len(time))

        # build a default scales array
        if scales is None:
            scales = np.arange(1, min(len(time)/10, 100))
        if scales[0] <= 0:
            raise ValueError("scales[0] must be > 0, found:"+str(scales[0]) )

        # Compute CWT
        dt = time[1]-time[0]
        coefs, scales_freq = pywt.cwt(signal, scales, wavelet, dt)

        self.signal = signal
        self.time   = time
        self.scales= scales

        self.wavelet= wavelet
        self.coefs  = coefs
        self.scales_freq = scales_freq
        self.dt     = dt



def cws(time, signal=None, scales=None, wavelet=DEFAULT_WAVELET,
         spectrum='amp', coi=True, yaxis='period',
         cscale='linear', cmap='jet', clim=None,
         cbar='vertical', cbarlabel=None,
         cbarkw=None,
         xlim=None, ylim=None, yscale=None,
         xlabel=None, ylabel=None, title=None,
         figsize=None, ax=None):

    if isinstance(time, CWT):
        c = time
        time, signal, scales, dt  = c.time, c.signal, c.scales, c.dt
        coefs, scales_freq        = c.coefs, c.scales_freq
    else:
        # allow to build the spectrum for signal only
        if signal is None:
            signal = time
            time   = np.arange(len(time))

        # build a default scales array
        if scales is None:
            scales = np.arange(1, min(len(time)/10, 100))
        if scales[0] <= 0:
            raise ValueError("scales[0] must be > 0, found:"+str(scales[0]) )

        # wavelet transform
        dt = time[1]-time[0]
        coefs, scales_freq = pywt.cwt(signal, scales, wavelet, dt)

    # create plot area or use the one provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # adjust y axis ticks
    scales_period = 1./scales_freq  # needed also for COI mask
    xmesh = np.concatenate([time, [time[-1]+dt]])
    if yaxis == 'period':
        ymesh = np.concatenate([scales_period, [scales_period[-1]+dt]])
        ylim  = ymesh[[-1,0]] if ylim is None else ylim
        ax.set_ylabel("Period" if ylabel is None else ylabel)
    elif yaxis == 'frequency':
        df    = scales_freq[-1]/scales_freq[-2]
        ymesh = np.concatenate([scales_freq, [scales_freq[-1]*df]])
        # set a useful yscale default: the scale freqs appears evenly in logscale
        yscale = 'log' if yscale is None else yscale
        ylim   = ymesh[[-1, 0]] if ylim is None else ylim
        ax.set_ylabel("Frequency" if ylabel is None else ylabel)
        #ax.invert_yaxis()
    elif yaxis == 'scale':
        ds = scales[-1]-scales[-2]
        ymesh = np.concatenate([scales, [scales[-1] + ds]])
        ylim  = ymesh[[-1,0]] if ylim is None else ylim
        ax.set_ylabel("Scale" if ylabel is None else ylabel)
    else:
        raise ValueError("yaxis must be one of 'scale', 'frequency' or 'period', found "
                          + str(yaxis)+" instead")

    # limit of visual range
    xr = [time.min(), time.max()]
    if xlim is None:
        xlim = xr
    else:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # adjust logarithmic scales (autodetected by default)
    if yscale is not None:
        ax.set_yscale(yscale)

    # choose the correct spectrum display function and name
    if spectrum == 'amp':
        values = np.abs(coefs)
        sp_title = "Amplitude"
        cbarlabel= "abs(CWT)" if cbarlabel is None else cbarlabel
    elif spectrum == 'real':
        values = np.real(coefs)
        sp_title = "Real"
        cbarlabel= "real(CWT)" if cbarlabel is None else cbarlabel
    elif spectrum == 'imag':
        values = np.imag(coefs)
        sp_title = "Imaginary"
        cbarlabel= "imaginary(CWT)" if cbarlabel is None else cbarlabel
    elif spectrum == 'power':
        sp_title = "Power"
        cbarlabel= "abs(CWT)$^2$" if cbarlabel is None else cbarlabel
        values = np.power(np.abs(coefs),2)
    elif hasattr(spectrum, '__call__'):
        sp_title = "Custom"
        values = spectrum(coefs)
    else:
        raise ValueError("The spectrum parameter must be one of 'amp', 'real', 'imag',"+
                         "'power' or a lambda() expression")

    # labels and titles
    ax.set_title("Continuous Wavelet Transform "+sp_title+" Spectrum"
                 if title is None else title)
    ax.set_xlabel("Time/spatial domain" if xlabel is None else xlabel )


    if cscale == 'log':
        isvalid = (values > 0)
        cnorm = LogNorm(values[isvalid].min(), values[isvalid].max())
    elif cscale == 'linear':
        cnorm = None
    else:
        raise ValueError("Color bar cscale should be 'linear' or 'log', got:"+
                         str(cscale))

    # plot the 2D spectrum using a pcolormesh to specify the correct Y axis
    # location of scales
    qmesh = ax.pcolormesh(xmesh, ymesh, values, cmap=cmap, norm=cnorm)

    if clim:
        qmesh.set_clim(*clim)

    # fill visually the Cone Of Influence
    # (locations subject to invalid coefficients near the borders of data)
    if coi:
        # convert the wavelet scales frequency into time domain periodicity
        scales_coi = scales_period
        max_coi  = scales_coi[-1]

        # produce the line and the curve delimiting the COI masked area
        mid = int(len(time)/2)
        time0 = np.abs(time[0:mid+1]-time[0])
        ymask = np.zeros(len(time), dtype=np.float16)
        ymhalf= ymask[0:mid+1]  # compute the left part of the mask
        ws    = np.argsort(scales_period) # ensure np.interp() works
        minscale, maxscale = sorted(ax.get_ylim())
        if yaxis == 'period':
            ymhalf[:] = np.interp(time0,
                  scales_period[ws], scales_coi[ws])
            yborder = np.zeros(len(time)) + maxscale
            ymhalf[time0 > max_coi]   = maxscale
        elif yaxis == 'frequency':
            ymhalf[:] = np.interp(time0,
                  scales_period[ws], 1./scales_coi[ws])
            yborder = np.zeros(len(time)) + minscale
            ymhalf[time0 > max_coi]   = minscale
        elif yaxis == 'scale':
            ymhalf[:] = np.interp(time0, scales_coi, scales)
            yborder = np.zeros(len(time)) + maxscale
            ymhalf[time0 > max_coi]   = maxscale
        else:
            raise ValueError("yaxis="+str(yaxis))

        # complete the right part of the mask by symmetry
        ymask[-mid:] = ymhalf[0:mid][::-1]

        plt.plot(time, ymask)
        ax.fill_between(time, yborder, ymask, alpha=0.5, hatch='/')

    # color bar stuff
    if cbar:
        cbarkw   = CBAR_DEFAULTS[cbar] if cbarkw is None else cbarkw
        colorbar = plt.colorbar(qmesh, orientation=cbar, ax=ax, **cbarkw)
        if cbarlabel:
            colorbar.set_label(cbarlabel)

    return ax


cws.__doc__ = """
Build and displays the 2D spectrum for Continuous Wavelet Transform

Call signatures::

    # build the CWT and displays the scaleogram
    ax = cws(signal)
    ax = cws(time, signal)

    # use a previously computed Continuous Wavelet Transform
    cwt = CWT(time, signal)
    ax  = cws(CWT)


Arguments
----------

- time : array of time/spatial domain locations
    Can be filled with signal values if this function is called with only
    one argument.
    This array should have a constant sampling rate for the spectrum to
    have sense. Missing time samples will degrade the interpretation.

- signal : (OPT) data to perform CWT on.
    Optional if the signal is provided as 1st arg.



Parameters
----------

- scales=[1..n] : an array of increasing values > 0.
    Important note: These scales are related to PyWavelet internal use
    and are consistent with period of times. This is the contrary of the
    theoretical wavelet scale parameter which is frequency related.
    In practice the construction of the PyWavelet signal uses arrays of
        ``n = s * 16`` values
    to build the mother wavelet signal at scale ``s``.

    Example::

        scales=np.arange(1,200, 2)

- wavelet= str | pywt.ContinuousWavelet : mother wavelet for CWT
    Note: for the continuous transform, there is no scaling function
    by contrast with the discrete transform.

    Example::

        wavelet=pywt.ContinuousWavelet('cmor1-1.5')

- spectrum=<type> : selects the type of processing for the CWT data
    <type> can be any of:
    - ['amp'] displays abs(CWT) (L1 norm)
    - 'real' / 'imag' for real or imaginary part only
    - 'power' for abs(CWT)**2
    - "lambda(np.array): np.array" to apply a custom processing on spectrum

- coi= [True]/False : nable/disable masking of the Cone Of Influence (COI).
    The COI relates to the regions near the borders of the spectrum where
    side effects occurs. At these locations the wavelet support used for the
    convolution is not fully contained by the signal data.
    The COI allow to show visually where the data cannot be fully trusted.
    The only way to get better values at larges time scales is...
    ...of course more data! :-)

        https://fr.mathworks.com/help/wavelet/ref/conofinf.html

- ax=None|matplotlib.AxesSubplot : allow to build complex plot layouts
    If no Axes are provided subplots() is called to build the figure.

- figsize=(width, height) : set figure size
    This is ignored if Axes are provided through ax=

- xlim=(min, max) : sets the display limits of X axis

- ylim=(min, max) : sets the display limits of Y axis

- clim=(min, max) : sets the color range limits for spectrum values
    (in other words: z axis range)

- cmap=<name> : matplotlib color map name (or instance) for the spectrum
    To view available maps type::

        matplotlib.pyplot.colormaps()

- cscale=['linear'] | 'log' :  selects scaling for CWT spectrum
    This parameter impacts the color bar ticks values and spectrum colors.

- yscale=['linear'] | 'log' :  selects scaling for the Y axis

- cbar= ['vertical'] | 'horizontal' : selects the color bar location

- cbarkw={} --  pass a hash of parameters to the matplotlib colorbar() call

- yaxis=<units type> : selects the Y axis units.
    - ['period'] : Convert scales to human readable period values which depend
        on the time input parameter.
        If time is not provided, periods are in number of samples units.
    - 'frequency' : Converts scales to frequency units depending on the
        time argument value.
        If time is not provided, the frequency represent the number of
        oscillations over the whole signal length.
        In this mode ``yscale`` is set to 'log' by default (if not provided).
    - 'scale' : display the wavelet scales parameter (in the PyWavelet sense
        as described in the Argument section) this should be used mainly for
        debug purposes as this parameter has no physical sense.
        The scale array support non constant step size.


Returns
-------
    ax : the matplplot ``AxesSubplot`` drawable area


Continuous Wavelet list
-----------------------
- """+("\n- ".join(WAVLIST))




def plot_wav(wav, figsize=None, axes=None):

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

    ax1.plot(fun_wav.real, label="real")
    ax1.plot(fun_wav.imag, "r-", label="imag")
    ax1.legend()
    ax1.set_title(wav.family_name)
    ax1.set_xlabel('Sample index')
    ax1.set_ylabel("Amplitude")
    ax2.plot(np.abs(np.fft.fft(fun_wav)))
    ax2.set_yscale('log')
    ax2.set_xlim(0, len(fun_wav)/2)
    ax2.set_title("Frequency support")
    ax2.set_xlabel("Frequency [Nb of oscillations/window]")

    return ax1, ax2



plot_wav.__doc__ =     """
Quick helper function to check visually the properties of a wavelet
in time domain and the filter view in frequency domain.


Parameters
----------

- wav : continuous wavelate name or wavelet instance as retured by pywt.

- axes= (ax1, ax2) : allow to customize the plot destinations

- figsize=(width,eight) : if provided then forward the size (inches)
    to matplotlib and build a new figure for display
    It is only used if axes is absent


Returns
-------
- ax1, ax2 : matplotlib graphics elements


Continuous Wavelet list
-----------------------
- """+("\n- ".join(WAVLIST))



def test_cws():
    """Graphical output test function
    """
    print("Multi-purpose test demo (needs graphic output for matplotlib)")

    print("  Plot Mexican Hat wavelet in time and frequency domains")
    plot_wav(DEFAULT_WAVELET)

    print("  Plot scaleogram of a gaussian")
    time = np.arange(1024)-512
    signal = np.cos(2*np.pi/52*time)
    fig = plt.figure()
    plt.plot(time, signal)
    plt.title("Input signal: cos($2*\pi/52*t$)")

#    cws(signal) # KISS

    scales = np.arange(1, 200, 2)
    cwt = CWT(time, signal, scales, 'cmor1-1.5')
    ax = cws(cwt,
              wavelet=DEFAULT_WAVELET,
              #yaxis='frequency',
              yaxis='period',
              #yaxis='scale',
              spectrum='power', coi=1,
              title="scaleogram of cos($2*\pi/52*t$): expect an horizontal bar",
              xlabel="sample index",
              #ylim=(40, 20),
              )
    plt.tight_layout()
    ax.annotate('cos($2*\pi/52*t$)', xy=(0, 52), xytext=(-100, 40),
                color="white", fontsize=15,
                arrowprops=dict(facecolor='white', shrink=0.05))
    ax.annotate('Cone Of Influence', xy=(-512+100, 100), xytext=(-300, 90),
                color="white", fontsize=15,
                arrowprops=dict(facecolor='white', shrink=0.05))
    plt.draw()

    print("  Plot test grid")
    fig = plt.figure()
    nrow, ncol = 3, 3

    ax = plt.subplot(nrow, ncol, 1)
    ax = cws(cwt, ax=ax, cbar='vertical', cmap='Reds',
             title="cbar='vertical'")

    ax = plt.subplot(nrow, ncol, 2)
    ax = cws(cwt, ax=ax, cbar='horizontal', cmap="Reds",
             title="cbar='horizontal'", xlabel="")
    ax.set_xticks([])

    ax = plt.subplot(nrow, ncol, 3)
    ax = cws(cwt, ax=ax, cbar='horizontal', cmap='Reds', cscale='log',
             title="cscale='log'", xlabel="")
    ax.set_xticks([])

    ax = plt.subplot(nrow, ncol, 4)
    ax = cws(cwt, ax=ax, cbar=0, cmap='Greens' ,yaxis='frequency',
             title="yaxis='freq'",  )

    ax = plt.subplot(nrow, ncol, 5)
    ax = cws(cwt, ax=ax, cbar=0,
             title="yaxis='scale'", cmap='Greys', yaxis='scale' )

    ax = plt.subplot(nrow, ncol, 6)
    ax = cws(cwt, ax=ax, cbar=0, cmap='Reds', yscale='log',
             title="yscale='log'", ylabel="YLABEL", xlabel="XLABEL" )

    ax = plt.subplot(nrow, ncol, 7)
    ax = cws(cwt, ax=ax, cbar=0, cmap='Reds', spectrum='power', coi=0,
             title="spectrum='power', coi=0" )

    ax = plt.subplot(nrow, ncol, 8)
    ax = cws(cwt, ax=ax, cbar=0, cmap='Reds', spectrum='real',
             title="spectrum='real'", xlim=(-420, -380) )

    ax = plt.subplot(nrow, ncol, 9)
    ax = cws(cwt, ax=ax, cbar=0, cmap='Reds', spectrum='imag',
             title="spectrum='imag'", xlim=(-420, -380) )


    plt.subplots_adjust(wspace=0.1)
    plt.tight_layout()
    plt.draw()
    plt.show()

if __name__ == '__main__':
    test_cws()

