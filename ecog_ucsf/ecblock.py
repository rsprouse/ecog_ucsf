#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick-n-dirty loading of UCSF ecog data.
"""

# Authors: Ronald L. Sprouse (ronald@berkeley.edu)
# 
# Copyright (c) 2015, The Regents of the University of California
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of the University of California nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import division
import numpy as np
import htkmfc
import os

class ECBlock(object):
    '''An object representing a block of UCSF ECOG data.'''
    def __init__(self, datadir='', subdir='ecogDS', data=np.array([]), htkrate=np.nan, badchan=[]):
        super(ECBlock, self).__init__()
        self.datadir = datadir
        self.subdir = subdir
        self.data = data   # Channel data for block
        self.htkrate = htkrate       # Sample rate of channel data from .htk file
        self.badchan = badchan        # List of bad channels in block

    def __repr__(self):
        r = "ECBlock(datadir='{:}', subdir='{:}', data={:}, htkrate={:}, badchan={:})".format(
            self.datadir, self.subdir, self.data, self.htkrate, self.badchan
        )
        return r

def int2wavname(n):
    '''Convert an integer in the range 1 to 256 to the ECOG file naming
convention where channel 1 is '11' and Channel 256 is '464'.'''
    return "Wav{:d}{:d}.htk".format(
        int(np.ceil(n/64)),
        int(np.mod(n-1, 64) + 1)
    )

def get_bad_channels(ddir, subdir='Artifacts', fname='badChannels.txt'):
    '''Return an array of bad channel numbers in ddir.'''
    with open(os.path.join(ddir, subdir, fname)) as f:
        return [int(n) for n in f.readline().strip().split()]
    
def read_block(ddir, subdir='ecogDS', channel_cb=None, badchan_subdir='Artifacts',
badchan='badChannels.txt'):
    '''Load all the Wav*.htk channel data in a block subdir into an ECBlock.

The channel_cb parameter may contain a callback function to apply to each
channel's data before storing in the data ndarray. The callback is applied
separately to each channel and may change the shape of the channel data,
as long as the channel shapes remain compatible with one another. If the
callback function changes the effective sample rate of the data, note that
the change will not be reflected in the rate attribute of the ECBlock
returned by read_block().

Returns an ECBlock object.
'''
    b = ECBlock(datadir=ddir)

    # Electrodes (channels) are numbered starting with 1.
    b.badchan = get_bad_channels(ddir, badchan_subdir, badchan)
    htk = htkmfc.openhtk(os.path.join(ddir, subdir, int2wavname(1)))
    b.htkrate = htk.sampPeriod * 1E-3
    c1 = np.squeeze(htk.getall())
    if channel_cb is not None:
        c1 = channel_cb(c1)
    b.data = np.empty([256] + list(c1.shape)) * np.nan
    if 1 not in b.badchan:
        b.data[0,] = c1
    for idx in range(2, 257):
        if idx not in b.badchan:
            htk = htkmfc.openhtk(os.path.join(ddir, subdir, int2wavname(idx)))
            if channel_cb is None:
                b.data[idx-1,] = np.squeeze(htk.getall())
            else:
                b.data[idx-1,] = channel_cb(np.squeeze(htk.getall()))
    return b
