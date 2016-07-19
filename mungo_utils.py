#Automated Mungo Enterprises Eurorack Focused WAV Bit Depth and Sample Rate Conversion Utilities
#would be nice to also have AIFF file support for all the Apple/Akai/samples out there

import os
import re
import sys
import time
import glob
import numpy as np
import wavio
import scipy.signal as sps

#either average all channels together or take the first channel only
def multi_to_mono(data,average=True):
    if len(data)>0 and len(data[0])>0:
        if average:
            s = len(data)    #channel count
            c = len(data[0]) #number of samples
            t = data[0,0]    #at least one channel and one sample to get its data type
            A = np.zeros((s,),dtype=t) #should be 32bit into or i4 int32
            for i in range(c):
                A += data[:,i] 
        else:
            A = data[:,0]
    else:
        A
    return A

#up or down samples based on the target samples
def resample(mono,target,module):
    resampled_data = sps.resample(mono,target[module]).astype('i4')
    return resampled_data

#reads a directory of audio files and does some conversion
def read_waves(in_dir,target,module,average=False):
    wavs = glob.glob(in_dir)
    data = []
    for wav in wavs:
        data += [resample(multi_to_mono(wavio.read(wav).data,average=False),target,module)]
    return data

def read_aiffs(in_dir,target,module,average=False):
    aiffs = glob.glob(in_dir)

#write a directory of audio files to mono WAV 16-bit integer
#Mungo uses naming conventions of W0 to W9 etc...
def write(out_dir,data,prefix,module):
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    j = 0
    for i in range(len(data)):
        if j%10==0: #make a new directory if needed
            j,last_dir = 0,out_dir+'/'+str(i/10)+'/'
        if not os.path.exists(last_dir):
            os.makedirs(last_dir)
        wavio.write(last_dir+prefix[module]+str(j)+'.wav',data[i],len(data[i]),sampwidth=2)
        j += 1
    return True

module = 'G0'
target = {'G0':500000,'S0':200000,'W0':4000,'C0':12000}
prefix = {'G0':'W','S0':'S','C0':'W','W0':'W'}
wav_in_dir = '/Users/tbecker/Music/Samples_Seeds/_Impulses/Samplicity M7 Main Wave, 24 bit, 44.1 Khz, v1.1/*/*M-to-S.wav'
data = read(wav_in_dir,target,module,False)

wav_out_dir = '/Users/tbecker/Music/_G0/M7/'
write(wav_out_dir,data,prefix,module)