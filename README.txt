# mungo_utils
Automated Mungo Enterprises Eurorack Focused WAV Bit Depth and Sample Rate Conversion Utilities

./mungo_utils.py -h
#usage: mungo_utils.py [-h] [-i AUDIO_INPUT_DIR] [-e AUDIO_EXT] [-m]
                      [-t TARGET_MUNGO_MODULE] [-o MUNGO_OUTPUT_DIR]

Automated Mungo Enterprises Eurorack Focused WAV Bit Depth and Sample Rate
Conversion Utilities Tested with WAV and AIFF audio files with 1|2 chnannels
and 16/24 bit depths provide a folder glob pattern to search and this utility
will match those full paths and one by one convert to the target Mungo Module
based on the published buffer sizes there for exampl the C0 has a buffer of
12000, so any audio file will be auto resampled to that size converted to a 16
bit WAV file format

optional arguments:
  -h, --help            show this help message and exit
  -i AUDIO_INPUT_DIR, --audio_input_dir AUDIO_INPUT_DIR
                        audio directory to search [required]
  -e AUDIO_EXT, --audio_ext AUDIO_EXT
                        either or aif and wav audio extensions to convert
                        [aif,wav]
  -m, --mix             mix multiple channels [False]
  -t TARGET_MUNGO_MODULE, --target_mungo_module TARGET_MUNGO_MODULE
                        the mungo target module to write to [G0=500K]
  -o MUNGO_OUTPUT_DIR, --mungo_output_dir MUNGO_OUTPUT_DIR
                        mungo output directory [required]
  
  [EXAMPLE USING THE GPL TEST FILES]
  
./mungo_utils.py -i "./test/*" -o ./test/C0/converted/ -t C0
using audio input directory:
./test/*
processing ./test/aifs/GP CHH 3.aif
1 audio channels detected
16 bit sample depth detected
44100Hz sample rate detected
---------------------------------------------------
processing ./test/aifs/GP CR 3.aif
1 audio channels detected
16 bit sample depth detected
44100Hz sample rate detected
---------------------------------------------------
processing ./test/aifs/GP KD 2.aif
1 audio channels detected
16 bit sample depth detected
44100Hz sample rate detected
---------------------------------------------------
processing ./test/aifs/GP OHH 2.aif
1 audio channels detected
16 bit sample depth detected
44100Hz sample rate detected
---------------------------------------------------
processing ./test/aifs/GP RCB 3.aif
1 audio channels detected
16 bit sample depth detected
44100Hz sample rate detected
---------------------------------------------------
processing ./test/aifs/GP RS 1.aif
1 audio channels detected
16 bit sample depth detected
44100Hz sample rate detected
---------------------------------------------------
processing ./test/aifs/GP SN 2.aif
1 audio channels detected
16 bit sample depth detected
44100Hz sample rate detected
---------------------------------------------------
processing ./test/wavs/CoffeeBeansSeq1_Test.wav
2 audio channels detected
24 bit sample depth detected
96000Hz sample rate detected
---------------------------------------------------