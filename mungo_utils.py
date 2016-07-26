#!/usr/bin/env python
#Timothy Becker, 25/07/2016 version 0.0

import argparse
import os
import glob
import numpy as np
#here is the resampling algorithm library
import scipy.signal as sps
#here are libraries that can handle 24-bit depths
import wavio
import aifcio

#parse commandline arguments for usage
overview = """
Automated Mungo Enterprises Eurorack Focused WAV Bit Depth and Sample Rate Conversion Utilities\n
Tested with WAV and AIFF audio files with 1|2 chnannels and 16/24 bit depths\n
provide a folder glob pattern to search and this utility will match those full paths\n
and one by one convert to the target Mungo Module based on the published buffer sizes there\n
for exampl the C0 has a buffer of 12000, so any audio file will be auto resampled to that size\n
converted to a 16 bit WAV file format\n
"""
parser = argparse.ArgumentParser(description=overview)
parser.add_argument('-i', '--audio_input_dir',type=str, help='audio directory to search\t[required]')
parser.add_argument('-e', '--audio_ext',type=str, help='either or aif and wav audio extensions to convert\t[aif,wav]')
parser.add_argument('-m', '--mix',action='store_true', help='mix multiple channels\t[False]')
parser.add_argument('-t', '--target_mungo_module',type=str, help='the mungo target module to write to\t[G0=500K]')
parser.add_argument('-o', '--mungo_output_dir',type=str, help='mungo output directory\t[required]')
args = parser.parse_args()
#check all the options and set defaults that have not been specified
if args.audio_input_dir is not None:
    in_dir = args.audio_input_dir
    print('using audio input directory:\n%s'%in_dir)
else:
    print('input directory not specified')
    raise IOError
if args.mungo_output_dir is not None:
    mungo_out_dir = args.mungo_output_dir
else:
    print('mungo output directory not specified')
    raise IOError
if args.audio_ext is not None:
    exts = args.audio_ext.split(',') #comman seperated list
else:
    exts = ['aif','wav']
if args.mix is not None:
    mix = args.mix
else:
    mix = False
if args.target_mungo_module is not None:
    module = args.target_mungo_module
else:
    module = 'G0'

#either mix all channels together or take the first channel only
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
#using scipy.signal.resample which is an FFT based resampling method (which is slow but decent sounding)
def resample(mono,target,module):
    resampled_data = sps.resample(mono,target[module]).astype('i4')
    return resampled_data

#auto detect the file type based on supplied extensions given the input directory
#target the appropriate number of resampling buffer from mungo manuals
#avergae/mix the input channels if average is False, otherwise pick channel 1
def read_aifs_or_wavs(in_dir,
                      exts=['aif','wav'],
                      module='G0',
                      average=False,
                      target={'G0':500000,'S0':200000,'W0':4000,'C0':12000}):
    audio_files = []
    for ext in exts:
        audio_files += glob.glob(in_dir+'/*.'+ext) #load the extensions that we want
        
    data,err,ns = [],[],[]
    for audio_file in audio_files:
        try:
            print('processing %s'%audio_file)
            if audio_file.rsplit('.')[-1].upper().find('AIF')>-1: #search for aif style file extension
                data += [resample(multi_to_mono(aifcio.read(audio_file).data,average=False),target,module)]
            elif audio_file.rsplit('.')[-1].upper().find('WAV')>-1: #search for wav style file extension
                data += [resample(multi_to_mono(wavio.read(audio_file).data,average=False),target,module)]
            else:
                ns += [audio_file] #extension and type is not supported
            print('---------------------------------------------------')
        except Exception:
            err += [audio_file]
            pass
    if len(err)>0:
        print('Conversion errors with the following supported files:')
        for i in err: print i
    if len(ns)>0:
        print('The following files have unsupported file types:')
        for i in ns: print i
    return data
    
#write a directory of audio files to mono WAV 16-bit integer
#Mungo uses naming conventions of W0 to W9 etc...
#automatically builds output directories every 10 mungo files
def write_mungo(out_dir,
                data,
                module='G0',
                prefix={'G0':'W','S0':'S','C0':'W','W0':'W'}):
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
#now batch process all the inputs and autogenerate the mungo WAV files
data = read_aifs_or_wavs(in_dir,exts,module,mix)
write_mungo(mungo_out_dir,data,module)
