Mungo Enterprises Eurorack Module Focused Batch WAV Audio Processing and File Generation Utilities
(c) Timothy James Becker 2016

requires: python 2.7.10+, numpy, scipy

notes:
(1) 24 bit file handling is based on the python multiple bit depth converter tool wavio written by Warren Weckesser (c) 2015
(2) phase_vocoder written by Victor Lazzarini (c) 2010

./mungo_utils.py -h
usage: mungo_utils.py [-h] [-I AUDIO_INPUT_DIR] [-E AUDIO_EXT]
                      [-T TARGET_MUNGO_MODULE] [-O MUNGO_OUTPUT_DIR] [-m] [-n]
                      [-f FADE] [-l] [-p] [-t]

Automated Mungo Enterprises Eurorack Focused WAV Bit Depth and Sample Rate
Conversion Utilities Tested with WAV and AIFF audio files with 1|2 chnannels
and 16/24 bit depths provide a folder glob pattern to search and this utility
will match those full paths and one by one convert to the target Mungo Module
based on the published buffer sizes there for exampl the C0 has a buffer of
12000, so any audio file will be auto resampled to that size converted to a 16
bit WAV file format

optional arguments:
  -h, --help            show this help message and exit
  -I AUDIO_INPUT_DIR, --audio_input_dir AUDIO_INPUT_DIR
                        audio directory to search [required]
  -E AUDIO_EXT, --audio_ext AUDIO_EXT either or aif and wav audio extensions to convert [aif,wav]
  -T TARGET_MUNGO_MODULE, --target_mungo_module TARGET_MUNGO_MODULE the mungo target module to write to [G0=500K]
  -O MUNGO_OUTPUT_DIR, --mungo_output_dir MUNGO_OUTPUT_DIR mungo output directory [required]
  -m, --mix             mix multiple channels [False]
  -n, --norm            normalize audio and remove DC offset [False]
  -f FADE, --fade FADE  target buffer fade out in samples default is exponential fade [256]
  -r, --reverse         reverse the buffer [False]
  -p, --phase           apply phase vocoder timestretch [False]
  -t, --trim            trim begining and end of file based on amplitude[False]
  
  [EXAMPLE USING THE GPL TEST FILES]
  
./mungo_utils.py -I "./test/*" -O ./test/converted/ -T C0 -n
using audio input directory:
./test/*
processing ./test/aifs/Arp_2600.aif
1 audio channels detected
16 bit sample depth detected
44100Hz sample rate detected
resampled to 12000Hz
trimed to sample start 0
audio array of <type 'numpy.int32'>
amplitude center at 1 correcting for DC offset
correction of amplitude center now at 0 correcting for DC offset
processing fade out
---------------------------------------------------
processing ./test/aifs/GP CHH 3.aif
1 audio channels detected
16 bit sample depth detected
44100Hz sample rate detected
resampled to 12000Hz
trimed to sample start 13
audio array of <type 'numpy.int32'>
amplitude center at 0 correcting for DC offset
correction of amplitude center now at 0 correcting for DC offset
processing fade out
---------------------------------------------------
processing ./test/aifs/GP CR 3.aif
1 audio channels detected
16 bit sample depth detected
44100Hz sample rate detected
resampled to 12000Hz
trimed to sample start 28
audio array of <type 'numpy.int32'>
amplitude center at 0 correcting for DC offset
correction of amplitude center now at 0 correcting for DC offset
processing fade out
---------------------------------------------------
processing ./test/aifs/GP KD 2.aif
1 audio channels detected
16 bit sample depth detected
44100Hz sample rate detected
resampled to 12000Hz
trimed to sample start 60
audio array of <type 'numpy.int32'>
amplitude center at 0 correcting for DC offset
correction of amplitude center now at 0 correcting for DC offset
processing fade out
---------------------------------------------------
processing ./test/aifs/GP OHH 2.aif
1 audio channels detected
16 bit sample depth detected
44100Hz sample rate detected
resampled to 12000Hz
trimed to sample start 24
audio array of <type 'numpy.int32'>
amplitude center at 0 correcting for DC offset
correction of amplitude center now at 0 correcting for DC offset
processing fade out
---------------------------------------------------
processing ./test/aifs/GP RCB 3.aif
1 audio channels detected
16 bit sample depth detected
44100Hz sample rate detected
resampled to 12000Hz
trimed to sample start 0
audio array of <type 'numpy.int32'>
amplitude center at 0 correcting for DC offset
correction of amplitude center now at 0 correcting for DC offset
processing fade out
---------------------------------------------------
processing ./test/aifs/GP RS 1.aif
1 audio channels detected
16 bit sample depth detected
44100Hz sample rate detected
resampled to 12000Hz
trimed to sample start 15
audio array of <type 'numpy.int32'>
amplitude center at 0 correcting for DC offset
correction of amplitude center now at 0 correcting for DC offset
processing fade out
---------------------------------------------------
processing ./test/aifs/GP SN 2.aif
1 audio channels detected
16 bit sample depth detected
44100Hz sample rate detected
resampled to 12000Hz
trimed to sample start 9
audio array of <type 'numpy.int32'>
amplitude center at 0 correcting for DC offset
correction of amplitude center now at 0 correcting for DC offset
processing fade out
---------------------------------------------------
processing ./test/wavs/DC_Offset_Test.wav
1 audio channels detected
16 bit sample depth detected
44100Hz sample rate detected
resampled to 12000Hz
trimed to sample start 27
audio array of <type 'numpy.int32'>
amplitude center at 0 correcting for DC offset
correction of amplitude center now at 0 correcting for DC offset
processing fade out
---------------------------------------------------
processing ./test/wavs/High_Resolution_24b_Test.wav
2 audio channels detected
24 bit sample depth detected
96000Hz sample rate detected
resampled to 12000Hz
trimed to sample start 0
audio array of <type 'numpy.int32'>
amplitude center at 1 correcting for DC offset
correction of amplitude center now at 0 correcting for DC offset
processing fade out
---------------------------------------------------
