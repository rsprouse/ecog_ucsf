#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick-n-dirty loading of UCSF ecog data.
"""

# Authors: Ronald L. Sprouse (ronald@berkeley.edu)
# 
# Copyright (c) 2018, The Regents of the University of California
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
import pandas as pd
from scipy.io import loadmat
from audiolabel import read_label
import ecog_ucsf.cmu_sphinx.htkmfc as htkmfc
import os

class ECBlock(object):
    '''An object representing a block of UCSF ECOG data.'''
    def __init__(self, basedir='', subdir='ecogDS', data=np.array([]),
datarate=np.nan, htkrate=np.nan, badchan=[]):
        super(ECBlock, self).__init__()
        self.basedir = basedir
        self.subdir = subdir
        self.data = data         # Channel data for block
        self.datarate = datarate # Data sample rate after converter applied
        self.htkrate = htkrate   # Sample rate of channel data from .htk file
        self.badchan = badchan   # List of bad channels in block

    @property
    def badmask(self):
        '''Boolean mask of indices for 'bad' time segments.'''
        return ~self.goodmask

    @property
    def ts(self):
        '''Return a time series that maps sample index to time.'''
        return np.arange(self.data.shape[1]) / self.datarate

    def __repr__(self):
        r = "ECBlock(basedir='{:}', subdir='{:}', data={:}, htkrate={:}, badchan={:})".format(
            self.basedir, self.subdir, self.data, self.htkrate, self.badchan
        )
        return r

def int2wavname(n):
    '''Convert an integer in the range 1 to 256 to the ECOG file naming
convention where channel 1 is '11' and Channel 256 is '464'.'''
    return "Wav{:d}{:d}.htk".format(
        int(np.ceil(n/64)),
        int(np.mod(n-1, 64) + 1)
    )

# TODO: check correctness! These are probably matlab-style indexes.
def get_bad_channels(basedir, subdir, fname):
    '''Return an array of bad channel numbers in basedir. Subtract 1 from
each value since numpy arrays are 0-based.'''
    with open(os.path.join(basedir, subdir, fname)) as f:
        return [int(n)-1 for n in f.readline().strip().split()]

def get_bad_segments(basedir, subdir, fname):
    '''Return a dataframe of bad segments.'''
    segfile = os.path.join(basedir, subdir, fname)
    if segfile.endswith('.mat'):
        m = loadmat(segfile)
        segs = pd.DataFrame(
            m['badTimeSegments'][:, [0,1]],
            columns=['t1', 't2']
        ).sort_values('t1').reset_index(drop=True)
    else:
        segs = pd.read_csv(
            segfile,
            sep=' ',
            header=None,
            usecols=[1,2],
            names=['secs', 'edge'],
            converters={
                # Convert to seconds.
                'secs': lambda x: int(x) * 1E-8,
                # Relabel 'b'->'t1' and 'e'->'t2'.
                'edge': lambda x: 't1' if x == 'b' else 't2'
            }
        )
        segs = segs.assign(segidx=np.repeat(np.arange(len(segs) // 2), 2)) \
                   .pivot_table(values='secs', index='segidx', columns='edge') \
                   .reset_index() \
                   .rename_axis(None, axis='columns') \
                   .loc[:, ['t1', 't2']]
    return segs

def replace_bad_segs(data, datarate, badsegs, val=np.nan):
    '''Given an input ndarray, return a modified ndarray in which values that occur
during the bad segment times are replaced with specified value (default np.nan).

Parameters
----------

data : ndarray
    input ndarray (n-dimensional data supported)

datarate : numeric
    sample rate of the data ndarray

badsegs : DataFrame
    bad segment dataframe as created by read_block(), i.e. columns 't1' and 't2'

val : numeric
    replacement value for bad segment samples (default np.nan)

Returns
-------

replaced : ndarray
    copy of input ndarray with replacement values
'''
    # Convert segment times to a mask of sample indexes.
    if len(badsegs) > 0:
        maxindex = len(data)-1
        segidx = np.minimum(
            (badsegs * datarate).apply(np.around).astype(np.int),
            maxindex
        )
        segmask = np.concatenate(
            [np.arange(r.t1, r.t2) for r in segidx.itertuples()]
        )
        # Assume time is the first axis.
        data[segmask] = val
    return data

def read_block(basedir, subdir='ecogDS', converter=None, dtype=np.float32,
replace=True, artifact_dir='Artifacts', badchan='badChannels.txt',
badsegs='badTimeSegments.mat', labeltiers=None):
    '''Load the Wav*.htk channel data and metadata in a block into an ECBlock.

Parameters
----------
basedir : str, required
    The base directory containing an ECOG block acquisition

subdir : str, optional (default 'ecogDS')
    The subdirectory in basedir from which to load the Wav*.htk files.

converter : function, optional (default None)
    The converter function to apply to each channel's data before storing
    in the output ndarray. The callback is applied separately to each channel
    (i.e. the data from each .htk file) and may change the shape of the
    channel data, as long as the channel shapes remain compatible with one
    another. If the callback function changes the effective sample rate of
    the data, note that the change will not be reflected in the htkrate
    attribute of the ECBlock returned by read_block().

dtype : data-type, optional (default 32-bit float)
    The desired data type for the output ndarray.

replace : boolean, optional (default True)
    If True, replace bad channel and bad segment values defined in the
    artifact subdirectory with np.nan.

artifact_dir : str, optional (default 'Artifacts')
    The subdirectory in basedir containing information on bad channels
    and bad segments.

badchan : str, optional (default 'badChannels.txt')
    The name of the file containing bad channel information.

badsegs : str, {*.mat, *.lab}, optional (default 'badTimeSegments.mat')
    The name of the file containing bad segment information. If the
    file extension is .mat the file will be loaded as a binary Matlab
    file. If the extension is .lab the file will be treated as a text file.

labeltiers : list
    List of tiers that will be passed to audiolabel's `read_label()`.
    If None, do not load labels.

Returns
-------
out : ECBlock
    ECBlock object containing ndarray block data and associated metadata.
'''
    b = ECBlock(basedir=basedir)
    
    b.name = os.path.basename(basedir)
    b.badchan = get_bad_channels(basedir, artifact_dir, badchan)
    b.badsegs = get_bad_segments(basedir, artifact_dir, badsegs)

    if labeltiers is not None:
        tg = os.path.join(basedir, b.name + '.TextGrid')
        tlist = read_label(tg, 'praat', tiers=labeltiers)
        for tidx, t in enumerate(tlist):
            masks = []
            for seg in b.badsegs.itertuples():
                masks.append( (seg.t2 < t.t1) | (seg.t1 > t.t2) )
            tlist[tidx] = t.assign(is_good=np.array(masks).all(axis=0))
            b.labels = dict(zip(labeltiers, tlist))

    htk = htkmfc.openhtk(os.path.join(basedir, subdir, int2wavname(1)))
    b.htkrate = htk.sampPeriod * 1E-4
    c = np.squeeze(htk.getall().astype(dtype))
    b.datarate = b.htkrate
    nsamp = c.shape[0] # Time axis expected to be first in all ecog .htk files
    if converter is not None:
        c = converter(c)
        if c.shape[0] != nsamp:  # if converter does resampling
            b.datarate = b.datarate * c.shape[0] / nsamp
    if replace is True:
        c = replace_bad_segs(c, b.datarate, b.badsegs)
    segedges = np.minimum(
        (b.badsegs * b.datarate).apply(np.around).astype(np.int),
        c.shape[0]
    )
    b.goodmask = np.array([True] * c.shape[0])
    if len(b.badsegs) == 0:
        b.badidx = np.array([])
    else:
        b.badidx = np.concatenate(
            [np.arange(r.t1, r.t2) for r in segedges.itertuples()]
        )
        b.goodmask[b.badidx] = False
    b.data = np.full([256] + list(c.shape), np.nan, dtype=dtype)
    if (replace is False) or (1 not in b.badchan):
        b.data[0,] = c
    for idx in range(1, 256):
        if (replace is False) or (idx not in b.badchan):
            htk = htkmfc.openhtk(
                os.path.join(basedir, subdir, int2wavname(idx+1))
            )
            c = np.squeeze(htk.getall().astype(dtype))
            if converter is None:
                b.data[idx,] = c
            else:
                b.data[idx,] = converter(c)
            if replace is True:
                c = replace_bad_segs(c, b.datarate, b.badsegs)
    return b
