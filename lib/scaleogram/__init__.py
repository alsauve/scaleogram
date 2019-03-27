# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 2019

@author: Alexandre Sauve


This module provide a user friendly *scaleogram* plot tool for 
wavelet based data analysis using Continuous Wavelet Transform (CWT).
A scaleogram is the equivalent of a spectrogram for wavelets, but it has
a better compromise between time and frequency domain.

The module has been designed with in mind:
    
- Ease of use
- Intuitive options for quality plots using matplotlib conventions
- Portability (python2 / python3)
- Speed : the plot is drawn with pmeshgrid which is resonably fast

Requirements::
    
    pip install PyWavelet
    pip install matplotlib>=2.0.0


Basic usage::
    
    import scaleogram as scg
    scg.cws(numpy_array)


"""

from __future__ import absolute_import
from .cws  import  cws, CWT
from .test import  test_cws, test_helpers
from .wfun import  child_wav, plot_wav_time, plot_wav_freq, plot_wav
from .wfun import  periods2scales, plot_wavelets
from pkg_resources import get_distribution, DistributionNotFound
import os.path


try:
    _dist = get_distribution('scaleogram')
    # Normalize case for Windows systems
    dist_loc = os.path.normcase(_dist.location)
    here = os.path.normcase(__file__)
    if not here.startswith(os.path.join(dist_loc, 'scaleogram')):
        # not installed, but there is another version that *is*
        raise DistributionNotFound
except DistributionNotFound:
    __version__ = 'Please install this project with setup.py'
else:
    __version__ = _dist.version

__all__ = [ cws, child_wav, test_cws, CWT, 
           plot_wav, plot_wav_time, plot_wav_freq ]


