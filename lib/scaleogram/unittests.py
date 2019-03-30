#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:44:34 2019

@author: Alexandre Sauvé
"""


from __future__ import absolute_import
import unittest
import numpy as np
import pywt as pywt
try:
    from wfun import fastcwt, WAVLIST, get_default, set_default, \
        periods2scales
except:
    from .wfun import fastcwt, WAVLIST, get_default, set_default, \
        periods2scales


class Test_wfun(unittest.TestCase):

    def test_fastcwt(self):
        signal = np.zeros(100, dtype=np.float64)
        scales = np.arange(1,150)
        for wavelet in [ l.split()[0] for l in WAVLIST]:
            coef1, freq1 = pywt.cwt(signal, scales, wavelet)
            coef2, freq2 = fastcwt( signal, scales, wavelet)

    def test_periods2scales(self):
        periods = np.arange(1,4)
        for wavelet in [ desc.split()[0] for desc in WAVLIST]:
            scales = periods2scales(periods, wavelet)
            assert((scales > 0).all()) # check central frequency availability
    
    def test_accessors(self):
        default = get_default()
        new_default = "mexh"
        set_default(new_default)
        assert(new_default == get_default())
        set_default(default) # restore original value
        assert(default == get_default())



if __name__ == '__main__':
    unittest.main()
    
    