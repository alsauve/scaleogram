#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:50:42 2019

@author: Alexandre Sauvé
"""

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

try:
    from wfun import plot_wav, WAVLIST, periods2scales
    from cws  import CWT, cws, DEFAULT_WAVELET
except ImportError:
    # egg support
    from .wfun import plot_wav, WAVLIST, periods2scales
    from .cws  import CWT, cws, DEFAULT_WAVELET
    
    
def test_cws():
    """Graphical output test function
    """
    print("Multi-purpose test demo (needs graphic output for matplotlib)")

    print("  Plot "+str(DEFAULT_WAVELET)+" wavelet in time and frequency domains")
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
              spectrum='power',
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


def test_helpers():
    print("Testing helper functions")
    
    print("  testing periods2scales()")
    periods = np.arange(1,4)
    for wavelet in [ desc.split()[0] for desc in WAVLIST]:
        scales = periods2scales(periods, wavelet)
        assert((scales > 0).all()) # check central frequency availability



if __name__ == '__main__':
    test_helpers()
    test_cws()

