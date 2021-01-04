from pydub import AudioSegment
from helper import denoise
import numpy as np
from scipy.io.wavfile import write, read
import matplotlib.pyplot as plt
import ffmpeg
import sys
import os
from os.path import dirname, realpath, join
import shutil
from multiprocessing import freeze_support

def denoising(sound_path):
    # read .wav original track
    sound = AudioSegment.from_file(sound_path)
    sampling_rate = sound.frame_rate

    # instantiate cache sound file paths
    denoised_path = sound_path.replace('_downsampled.wav', '_denoised.wav')

    # apply denoise algo on original track in .wav format
    denoised = denoise(sound_path)
    
    # save denoised .wav track
    write(denoised_path, rate=sampling_rate, data=denoised)

    return denoised_path


# rootdir = sys.executable.rsplit('/',1)[0]
rootdir = os.getcwd()
datadir = join(rootdir,'Samples')
exportdir = join(rootdir,'Exports')
if not os.path.exists(exportdir):
    os.mkdir(exportdir)
else:
    shutil.rmtree(exportdir)
    os.mkdir(exportdir)
original_files = os.listdir(datadir)
print(original_files)

for file in original_files:
    if file.endswith(".wav"):
        # file = <name>.wav
        # cwd = ~/Samples/<name>.wav
        cwd = join(datadir, file)
        # filename = ~/Samples/<name>
        filename = file.replace('.wav','')
        # newfolder_path = ~/Exports/<name>
        newfolder_path = join(exportdir,filename)
        os.mkdir(newfolder_path)

        # original_path = ~/Exports/<name>/<name>.wav
        original_path = join(newfolder_path,file)
        # copy original tracks to original_path
        shutil.copyfile(src=cwd, dst=original_path)

        # carry out downsampling, denoising and amplification in original_path
        downsampled_path = original_path.replace('.wav','_downsampled.wav')

        (ffmpeg
            .input(original_path).audio
            .output(downsampled_path, ar='8192')
            .run()
        )
        denoised_path = denoising(downsampled_path)
        denoised_amplified_path = denoised_path.replace('.wav','_amplified.wav')
        (ffmpeg
            .input(denoised_path).audio
            .filter('volume',50).output(denoised_amplified_path, ar='8192')
            .run()
        )
        sr1, wav1 = read(original_path)
        sr2, wav2 = read(downsampled_path)
        sr3, wav3 = read(denoised_path)
        fig, axs = plt.subplots(3, figsize=(25,10))
        fig.suptitle('Original, Downsampled and Denoised')
        time_length = len(wav3)/sr3
        axs[0].plot(wav1[:int(sr1*time_length)])
        axs[1].plot(wav2[:int(sr2*time_length)])
        axs[2].plot(wav3)
        plot_path = original_path.replace('.wav','.png')
        fig.savefig(plot_path)
        plt.close()

freeze_support()