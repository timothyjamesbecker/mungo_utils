#DSP tools for audio arrays: 
#mixing, resampling, delay, pitch shifting, filtering, phase vocoding

import sys
import numpy as np
import scipy.signal as sps

#either mix all channels together or take the first channel only
def multi_to_mono(multi,mix=True):
    data,rate = multi.data,multi.rate
    if len(data)>0 and len(data[0])>0:
        if mix:
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
    return A,rate

#remove DC offset and normalize the signal
def normalize(mono,limit=4,prop=0.25):
    #skip pass begining transients
    skip = np.int32(len(mono)*prop)
    print('normalization skipping past %s samples'%skip)
    #center the average of the signal
    print('audio array of %s'%type(mono[0]))    
    c = np.int32(np.int32(round(np.mean(mono[skip:]),0)))
    print('amplitude center at %s correcting for DC offset'%c)
    for i in range(len(mono)): #hard clipping on  the transients
        mono[i] = min(np.iinfo(np.int32).max-limit,mono[i]-c,max(np.iinfo(np.int32).min+limit,mono[i]-c))
    max_pos = np.argmax(np.abs(mono))
    print('file peak at position %s'%max_pos)
    #high pass the audio
    mono = fft_high_pass(mono)
    c = np.int32(np.int32(round(np.mean(mono),0)))
    print('correction of amplitude center now at %s correcting for DC offset'%c)
    return mono

#trim the start of the audio array if it is below the threashold
def trim(mono,start_threshold=512,stop_threshold=48,min_samples=96):
    i,j = 0,len(mono)-1
    while i < len(mono)-1 and np.abs(mono[i]) <= start_threshold: i += 1
    while j > 0 and np.abs(mono[j])           <= stop_threshold:  j -= 1
    #now check the indecies
    if i < j and j-i > min_samples:
        print('trimed to sample start %s'%i)
        print('trimed to sample end %s'%j)
        return mono[i:j]
    else:
        return mono

#reverse the buffer
def reverse(mono):
    return mono[::-1]

#apply a dcreasing vloume envelope to the array
def fade_out(mono,nsamples,log=False):
    linear = np.zeros((nsamples,),dtype='f4')
    x,step = 0.0,1.0/nsamples
    if log: #log fade
        for i in range(nsamples):
            linear[i] = np.log1p(x*(np.e-1))
            x += step
    else: #exponential fade
        for i in range(nsamples):
            linear[i] = x/np.exp(1.0-x)
            x += step
    if len(mono) >= nsamples:
        print('processing fade out')
        for i in range(nsamples):
            mono[-1*i] = int(round(mono[-1*i]*linear[i],0))
    #print(mono[-1*nsamples:])
    return mono

def fft_high_pass(data):
    A = np.fft.rfft(data)
    #print('fft shape is %s'%A.shape)
    A[0:10] = 0.0
    return np.array(np.fft.irfft(A),dtype='i4')

#steep high pass for DC offset filtering
def butter_highpass_filter(data):
    b, a = sps.butter(6, [200.0/(0.5*len(data))], btype='high')
    y = sps.lfilter(b, a, data)
    return np.array(y,dtype='i4')

#up or down samples based on the target samples
#using scipy.signal.resample which is an FFT based resampling method (which is slow but decent sounding)
def resample(mono,target,module):
    print('resampled to %sHz'%target[module])
    resampled_data = sps.resample(mono,target[module]).astype('i4')
    return resampled_data    

#decrease bit depth with a triangle dither
def bit_convert(mono,output_type='i2',dither=True):
    return []

# (c) V Lazzarini, 2010, GPL
def phase_vocoder(mono, sr, N=2048, tscale= 1.0):
    L,H = len(mono),N/4
    # signal blocks for processing and output
    phi  = np.zeros(N)
    out = np.zeros(N, dtype=complex)
    sigout = np.zeros(L/tscale+N)
    
    # max input amp, window
    amp = max(mono)
    win = sps.hanning(N)
    p = 0
    pp = 0    
    while p < L-(N+H):
        if p%1024==0: print '.',
        # take the spectra of two consecutive windows
        p1 = int(p)
        spec1 =  np.fft.fft(win*mono[p1:p1+N])
        spec2 =  np.fft.fft(win*mono[p1+H:p1+N+H])

        # take their phase difference and integrate
        phi += (np.angle(spec2) - np.angle(spec1))
        
        # bring the phase back to between pi and -pi
        for i in phi:
            while i   > np.pi: i -= 2*np.pi
            while i <= -np.pi: i += 2*np.pi
        out.real, out.imag = np.cos(phi), np.sin(phi)

        # inverse FFT and overlap-add
        sigout[pp:pp+N] += (win*np.fft.ifft(abs(spec2)*out)).real
        pp += H
        p += H*tscale
    print('')
    return np.array(amp*sigout/max(sigout), dtype='int16')

def pitch_shifter(mono, sr):
    return []