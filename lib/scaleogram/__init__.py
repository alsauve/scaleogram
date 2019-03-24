# -*- coding: utf-8 -*-
from __future__ import absolute_import
from .cws import  cws, plot_wav, test_cws
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

__all__ = [ cws, plot_wav, test_cws ]


