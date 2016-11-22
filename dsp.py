#DSP tools for audio arrays: 
#mixing, resampling, delay, pitch shifting, filtering, phase vocoding

import sys
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt

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

def sin_shape(mono,r):
    l = len(mono)
    f = np.abs(np.sin([i for i in range(0,l,r)]))
    i,s = np.float(0.0),1.0/np.float32(l)
    for i in range(0,l,r):
        mono[i] = i*s*f[i]
    return np.array(mono,dtype='i4')

#creates an AC-centered non-liniar wavetable shape taking coefficents C => sum(C)=1
def nonlin_shape(l,C,h,plot=False,limit=0):
    lin = np.arange(0.0,2.0*np.pi,2.0*np.pi/(l/2.0))
    if len(lin) == l/2:
        C = np.asarray(C,dtype='f4')
        C /= np.sum(C) #normalize to 1.0
        #the waveshaping function here
        raw = C[0]*lin[:len(lin/2)+1]/h+\
              C[1]*np.power(lin[:len(lin/2)+1]*h,0.5) +\
              C[2]*np.power(lin[:len(lin/2)+1]*h,0.25) +\
              C[3]*np.log1p(lin[:len(lin/2)+1]**h) +\
              C[4]*np.tanh(lin[:len(lin/2)+1]**h) +\
              C[5]*np.sin(h*np.pi*np.array(lin[:len(lin/2)+1],dtype='f4')/l)**2
        #can extend into some type of non linear grammar (with a non linear parser)
        nonlin = np.zeros((l,),dtype='f4')
        nonlin[l/2-1:-1] = raw
        nonlin[:l/2]     = raw[::-1]*-1.0
        nonlin = nonlin[:-2]
        MIN,MAX = np.min(nonlin),np.max(nonlin)     #now scale it to [np.iinfo(np.int32).min+4,np.iinfo(np.int32).max-4]
        nonlin *= ((np.iinfo(np.int32).max-limit)-(np.iinfo(np.int32).min+limit))/(MAX-MIN) #scale it
        for i in range(1,len(nonlin)-1):
            if nonlin[i]==-np.inf or nonlin[i]==np.inf or np.isnan(nonlin[i]):
                nonlin[i] = (nonlin[i-1]+nonlin[i+1])/2.0
        nonlin = np.array(np.round(nonlin,0),dtype='i4')
        if plot:
            plt.plot(nonlin)
            plt.show()
    else:
        print('length error')
        nonlin = np.zeros((l,),dtype='f4')
    return nonlin
    
#n is a fraction of the size 0.5 => 12E3 will have
def impulse_exp(mono):
    l = len(mono)
    r = np.random.choice([0.1,0.2,0.4])
    i = np.uint32(1)
    while i < len(mono):
        mono[i-1] = np.iinfo(np.int32).max                         #z is the scale of affect
        mono[i]   = np.iinfo(np.int32).min                         #with single impulses
        i += int(round(i*r,0)+1)
    return np.array(mono[1:l]+[0],dtype='i4')

#n is a fraction of the size 0.5 => 12E3 will have
def impulse_harm(mono,H):
    l = len(mono)
    w = np.random.choice([2,4,9,13,16,32,49,81,121])
    for i in range(1,l):
        for j in H:
            if i%j==0 and i+w/2+1<len(mono):
                mono[i-1:i+w/2]     = [np.iinfo(np.int32).max for k in range(w/2)]
                mono[i+w/2:i+w/2+1] = [np.iinfo(np.int32).min for k in range(w/2)]
                break
    return np.array(mono[1:l]+[0],dtype='i4')

#dense random cluster
def impulse_rand(mono):
    d = np.random.choice([i for i in range(len(mono)/10)])
    N = np.sort(np.random.normal(0,int(4*len(mono)),size=d))
    N = N[d/2:]
    a,b = np.min(N),np.max(N)
    N = sorted(list(set(np.array(np.round(((N-a)/(b-a))*(len(mono)-2),0),dtype='i4'))))
    for i in N:
        if i < len(mono):
            mono[i]   = np.iinfo(np.int32).max 
            mono[i+1] = np.iinfo(np.int32).min
    return np.array(mono[1:len(mono)]+[0],dtype='i4')
    
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

# (c) V Lazzarini, 2010, GPL
def pitch_shifter(mono, pitch, time):
    sigout = np.array(mono,dtype='f4')
    size  = time                         # delay time in samples
    delay = np.zeros((size,),dtype='f4') # delay line
    env = np.bartlett(size)              # fade envelope table
    tap1,tap2,wp = 0 ,size/2,0           #taps
    for i in range(len(mono)):
        delay = sigout[i]                                                 # fill the delay line
        frac = tap1 - int(tap1)                                         # first tap, linear interp readout
        if tap1 < size - 1 : delaynext = delay[tap1+1]                  # not at boundry
        else: delaynext = delay[0]                                      # wrap back to the begining
        sig1  =  delay[int(tap1)] + frac*(delaynext - delay[int(tap1)]) # invert and mix
        frac = tap2 - int(tap2)                                         # second tap, linear interp readout
        if tap2 < size - 1 : delaynext = delay[tap2+1]
        else: delaynext = delay[0]
        sig2  =  delay[int(tap2)] + frac*(delaynext - delay[int(tap2)])
        # fade envelope positions
        ep1 = tap1 - wp
        if ep1 <  0: ep1 += size
        ep2 = tap2 - wp
        if ep2 <  0: ep2 += size
        # combine tap signals
        sigout[i] = env[ep1]*sig1 + env[ep2]*sig2
        # increment tap pos according to pitch transposition
        tap1 += pitch
        tap2 = tap1 + size/2
        # keep tap pos within the delay memory bounds
        while tap1 >= size: tap1 -= size
        while tap1 < 0: tap1 += size
        while tap2 >= size: tap2 -= size
        while tap2 < 0: tap2 += size
        # increment write pos
        wp += 1
        if wp == size: wp = 0
    return np.array(sigout,dtype='int16')
    
    (sr,signalin) = wavfile.read(sys.argv[2])
    pitch = 2.**(float(sys.argv[1])/12.)
    signalout = zeros(len(signalin))
    
    fund = 131.
    dsize = int(sr/(fund*0.5))
    print dsize
    signalout = pitchshifter(signalin,signalout,pitch,dsize)
    wavfile.write(sys.argv[3],sr,array((signalout+signalin)/2., dtype='int16'))