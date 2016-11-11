#!/usr/bin/env python
#Timothy Becker, 25/07/2016 version 0.0

import argparse
import os
import glob
import numpy as np
#here are libraries that can handle 24-bit depths
import wavio
import aifcio
#dsp libs need numpy and scipy
import dsp

#auto detect the file type based on supplied extensions given the input directory
#target the appropriate number of resampling buffer from mungo manuals
#avergae/mix the input channels if average is False, otherwise pick channel 1
def read_aifs_or_wavs(in_dir,
                      exts=['aif','wav'],
                      module='G0',
                      mix=False,
                      trim=False,
                      norm=False,
                      phase=False,
                      rev=False,
                      fade=256,
                      target={'G0':500000,'S0':200000,'W0':4000,'C0':12000}):
    audio_files = []
    for ext in exts:
        audio_files += glob.glob(in_dir+'/*.'+ext) #load the extensions that we want
        
    data,err,ns = [],[],[]
    for audio_file in audio_files:
        try:
            print('processing %s'%audio_file) #search for aif style file extension
            is_aif = audio_file.rsplit('.')[-1].upper().find('AIF')>-1
            is_wav = audio_file.rsplit('.')[-1].upper().find('WAV')>-1
            if not is_aif and not is_wav: #extension not supported
                ns += [audio_file]
            else:
                if   is_aif: mono,rate = dsp.multi_to_mono(aifcio.read(audio_file),mix)    #convert to mono
                elif is_wav: mono,rate = dsp.multi_to_mono(wavio.read(audio_file),mix)     #convert to mono
            if trim:  mono = dsp.trim(mono)
            if phase: mono = dsp.phase_vocoder(mono,rate,1024,1.0*target[module]/rate)     #timestretching via PV
            resampled = dsp.resample(mono,target,module)                                   #up/down sample
            if norm: resampled = dsp.normalize(resampled)                                  #normalize and clean final result
            if fade > 0: resampled = dsp.fade_out(resampled,fade)                          #exp fade out
            if rev: resampled = dsp.reverse(resampled)                                     #option reverse
            data += [resampled]
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

# this is a random IR generator with three main modes:
#[1] EXP exponential decay type IRs
#[2] HARM harmonic type (integer ratios) IRS
#[3] RAND random aka implulse noise type IRs
#m is the number of random IRs to make
#b is the buffer size
#s is the number of passes for each file
#r is the number of harmonic passes
def gen_C0_IRs(out_dir,IRs=100,buffersize=int(12E3),rev_prob=0.1,types=['EXP','HARM','RAND'],
               H=[16,32,64,81],passes=[1,2,3],harmonics=[1,2,4]):
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    data = []
    for i in range(IRs):
        h = np.random.choice(passes)
        new = [0 for j in range(buffersize)]
        for y in range(h):
            s = np.random.choice(types)
            if s=='EXP':
                print('EXP')
                new = list(dsp.fade_out(dsp.impulse_exp(new),buffersize))
            if s=='HARM':
                print('HARM')
                r = np.random.choice(harmonics)
                for x in range(r):
                    new = list(dsp.fade_out(dsp.fft_high_pass(dsp.impulse_harm(new,H)),buffersize))
            if s=='RAND':
                print('RAND')
                new = list(dsp.fade_out(dsp.impulse_rand(new),buffersize))
            if np.random.choice([True,False],p=[2*rev_prob,1.0-2*rev_prob]):
                new = new[::-1]
        data += [dsp.fft_high_pass(new)]
    j = 0
    for i in range(len(data)):
        if j%10==0: #make a new directory if needed
            j,last_dir = 0,out_dir+'/'+str(i/10)+'/'
        if not os.path.exists(last_dir):
            os.makedirs(last_dir)
        wavio.write(last_dir+'W'+str(j)+'.wav',data[i],len(data[i]),sampwidth=2)
        j += 1
    return True

def gen_W0_WTs(out_dir,WTs=100,buffersize=int(4E3),c_range=[0,10],h_range=[0,16],plot=False):
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    data = []
    for i in range(WTs):
        h = np.random.choice(range(h_range[0],h_range[1]))
        C = np.random.choice(range(c_range[0],c_range[1]),6)
        data += [dsp.nonlin_shape(buffersize,C,h,plot)]
    j = 0
    for i in range(len(data)):
        if j%10==0: #make a new directory if needed
            j,last_dir = 0,out_dir+'/'+str(i/10)+'/'
        if not os.path.exists(last_dir):
            os.makedirs(last_dir)
        wavio.write(last_dir+'W'+str(j)+'.wav',data[i],len(data[i]),sampwidth=2)
        j += 1
    return True    
    
#now can be used as a library too: import mungo_utils
if __name__ == '__main__':
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
    parser.add_argument('-I', '--audio_input_dir',type=str, help='audio directory to search\t[required]')
    parser.add_argument('-E', '--audio_ext',type=str, help='either or aif and wav audio extensions to convert\t[aif,wav]')
    parser.add_argument('-T', '--target_mungo_module',type=str, help='the mungo target module to write to\t[G0=500K]')
    parser.add_argument('-O', '--mungo_output_dir',type=str, help='mungo output directory\t[required]')
    #DSP arguments for optional processing
    parser.add_argument('-m', '--mix',action='store_true', help='mix multiple channels\t[False]')
    parser.add_argument('-n', '--norm',action='store_true', help='normalize audio and remove DC offset\t[False]')
    parser.add_argument('-f', '--fade',type=int, help='target buffer fade out in samples default is exponential fade\t[256]')
    parser.add_argument('-r', '--reverse',action='store_true', help='reverse audio buffer\t[False]')
    parser.add_argument('-p', '--phase',action='store_true', help='apply phase vocoder timestretch\t[False]')
    parser.add_argument('-t', '--trim',action='store_true', help='trim begining and end of file based on amplitude\t[False]')
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
    
    mix      = args.mix
    norm     = args.norm
    reverse  = args.reverse
    trim  = args.trim
    phase = args.phase
    if args.fade is not None:
        fade = args.fade
    else:
        fade = 96
    
    if args.target_mungo_module is not None:
        module = args.target_mungo_module
    else:
        module = 'G0'    
    #now batch process all the inputs and autogenerate the mungo WAV files
    data = read_aifs_or_wavs(in_dir,exts,module,mix,trim,norm,phase,reverse,fade)
    write_mungo(mungo_out_dir,data,module)
