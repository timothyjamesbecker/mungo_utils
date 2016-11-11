#interactive example
import mungo_utils as mu

audio_input_dir     = './test/'
audio_ext           = ['aif','wav']
target_mungo_module = 'C0'
mungo_out_dir       = './test/converted/'

mix                 = True
norm                = True
fade                = 256     #number of output samples to fade
rev                 = False   #reverse buffer
phase               = False
trim                = False

#will execute but then you will have interactive access to the audio arrays stored in data after
#data = mu.read_aifs_or_wavs(audio_input_dir,audio_ext,target_mungo_module,mix,norm,phase,fade)
#mu.write_mungo(mungo_out_dir,data,target_mungo_module)


#munfo_out_dir will create as many folders of IRs as needed
#IRs is the total number of random IRs to generate
#buffersize is like the target module, aka  how many samples
#rev_prob is the probabilty at each IR generation pass to reverse the IR
#H is the harmonic integer list to use for harmonic type IRs
#passes is the number of passes to randomly pick for each IR
#harmonics are the number of integers to use on a single harmonic pass
#mu.gen_c0_IRs(mungo_out_dir,
#              IRs=10,
#              buffersize=int(12E3),
#              rev_prob=0.5,
#              types=['HARM','EXP','RAND'],
#              H=[16,32,64,81],
#              passes=[1,2,8],
#              harmonics=[1,2,4])

mu.gen_W0_WTs(mungo_out_dir,
              WTs=10,
              buffersize=int(4E3),
              c_range=[0,10000],
              h_range=[-4,32],
              plot=True)