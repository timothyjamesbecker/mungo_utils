#interactive example

import mungo_utils as mu

audio_input_dir     = './test/Filesofweirdness/'
audio_ext           = ['aif','wav']
target_mungo_module = 'G0'
mungo_out_dir       = './test/converted/'

mix                 = True
norm                = True
fade                = 256     #number of output samples to fade
rev                 = False   #reverse buffer
phase               = False
trim                = False

#will execute but then you will have interactive access to the audio arrays stored in data after
data = mu.read_aifs_or_wavs(audio_input_dir,audio_ext,target_mungo_module,mix,norm,phase,fade)
mu.write_mungo(mungo_out_dir,data,target_mungo_module)