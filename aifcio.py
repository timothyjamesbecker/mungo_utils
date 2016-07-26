"""
The aifcio module defines the functions:
read(file)
    Read a AIFC file and return a `aifcio.Aifc` object, with attributes
    `data`, `rate` and `sampwidth`.

based on the wavio module listed below
-----
Author: Warren Weckesser
License: BSD 2-Clause:
Copyright (c) 2015, Warren Weckesser
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import division as _division

import aifc as _aifc
import numpy as _np


__version__ = "0.0.4.dev0"


def _aifc2array(nchannels, sampwidth, rate, data):
    """data must be the string containing the bytes from the aif file."""
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    print('%s audio channels detected'%nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        print("24 bit sample depth detected")
        print("%sHz sample rate detected"%rate)
        a = _np.empty((num_samples, nchannels, 4), dtype=_np.uint8)
        raw_bytes = _np.fromstring(data, dtype=_np.uint8).byteswap() #
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1]).byteswap()
    else:
        print("%s bit sample depth detected"%(sampwidth*8))
        print("%sHz sample rate detected"%rate)
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = _np.fromstring(data, dtype='<%s%d' % (dt_char, sampwidth))
        result = a.reshape(-1, nchannels).byteswap()
    return result

class Aifc(object):
    """
    Object returned by `aifcio.read`.  Attributes are:
    data : numpy array
        The array of data read from the AIFC file.
    rate : float
        The sample rate of the AIFC file.
    sampwidth : int
        The sample width (i.e. number of bytes per sample) of the AIFC file.
        For example, `sampwidth == 3` is a 24 bit AIFC file.
    """

    def __init__(self, data, rate, sampwidth):
        self.data = data
        self.rate = rate
        self.sampwidth = sampwidth

    def __repr__(self):
        s = ("Aifc(data.shape=%s, data.dtype=%s, rate=%r, sampwidth=%r)" %
             (self.data.shape, self.data.dtype, self.rate, self.sampwidth))
        return s


def read(file):
    """
    Read a AIFC file.
    Parameters
    ----------
    file : string or file object
        Either the name of a file or an open file pointer.
    Returns
    -------
    wav : aifcio.Aifc() instance
        The return value is an instance of the class `wavio.Wav`,
        with the following attributes:
            data : numpy array
                The array containing the data.  The shape of the array
                is (num_samples, num_channels).  num_channels is the
                number of audio channels (1 for mono, 2 for stereo).
            rate : float
                The sampling frequency (i.e. frame rate)
            sampwidth : float
                The sample width, in bytes.  E.g. for a 24 bit AIFC file,
                sampwidth is 3.
    Notes
    -----
    This function uses the `wave` module of the Python standard libary
    to read the AIFC file, so it has the same limitations as that library.
    In particular, the function does not read files with floating point data.
    The array returned by `aifcio.read` is alway two-dimensional.  If the
    AIFC data is mono, the array will have shape (num_samples, 1).
    """
    aif = _aifc.open(file)
    rate = aif.getframerate()
    nchannels = aif.getnchannels()
    sampwidth = aif.getsampwidth()
    nframes = aif.getnframes()
    data = aif.readframes(nframes)
    aif.close()
    array = _aifc2array(nchannels, sampwidth, rate, data)
    a = Aifc(data=array, rate=rate, sampwidth=sampwidth)
#    print(a)
    return a


_sampwidth_dtypes = {1: _np.uint8,
                     2: _np.int16,
                     3: _np.int32,
                     4: _np.int32}
_sampwidth_ranges = {1: (0, 256),
                     2: (-2**15, 2**15),
                     3: (-2**23, 2**23),
                     4: (-2**31, 2**31)}


def _scale_to_sampwidth(data, sampwidth, vmin, vmax):
    # Scale and translate the values to fit the range of the data type
    # associated with the given sampwidth.

    data = data.clip(vmin, vmax)

    dt = _sampwidth_dtypes[sampwidth]
    if vmax == vmin:
        data = _np.zeros(data.shape, dtyp=dt)
    else:
        outmin, outmax = _sampwidth_ranges[sampwidth]
        if outmin != vmin or outmax != vmax:
            data = ((float(outmax - outmin)) * (data - vmin) /
                    (vmax - vmin)).astype(_np.int64) + outmin
            data[data == outmax] = outmax - 1
        data = data.astype(dt)

    return data
