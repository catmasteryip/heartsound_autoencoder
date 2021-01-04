from os.path import dirname, join
from scipy.io import wavfile

import numpy as np

import mel
# from sklearn.preprocessing import MinMaxScaler

# the model was trained in tf1
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.models import load_model
tf.disable_v2_behavior()

def load_sound(filename):
    '''
        Return sampling_rate and period for .wav file at filename
        Args:
            filename(str): directory of the .wav file
        Returns:
            sampling_rate(int): sampling rate
            wave.T(list(int)): lengths of sound wave in seconds
    '''
    sampling_rate, wave = wavfile.read(filename)
    # what is it for?
    assert (len(wave.T) > 4)
    return sampling_rate, wave.T


def divide_single_wave_into_smaller_chunk(output_duration=3, wave=None, sampling_rate=None, shift=0):
    '''
        Divide single wave into chunks of 3 seconds
        Args:
            output_duration(int): number of seconds of an output chunk
            wave(ndarray): sound wave
            sampling_rate(int): sampling rate in Hz
            shift(int): ?
        Returns:
            wave_chunks(ndarray): wave chunks with length of output_duration (s)
    '''
    shift_abs = int(sampling_rate * shift)
    chunk_length = sampling_rate*output_duration
    min_length = (output_duration)*sampling_rate - 2
    wave_chunks = []
    temp_wave = wave.copy()
    count = 0
    temp_wave = temp_wave[shift_abs:]
    while(len(temp_wave) >= min_length):
        count += 1
        new_chunk = temp_wave[0:chunk_length]
        wave_chunks.append(new_chunk)
        temp_wave = temp_wave[chunk_length*count:]
    return wave_chunks


# def minmax(wave):
#     scaler = MinMaxScaler()
#     scaler.fit(wave.reshape(-1, 1))
#     wave = scaler.transform(wave.reshape(-1, 1))
#     wave = wave.reshape(8192*3,)
#     return wave


def audio2spec(filename):
    sr, wav = load_sound(filename)

    assert(sr == 8192)
    chunks = divide_single_wave_into_smaller_chunk(3, wav, sr)
    specs = []
    for c in chunks:
        spec = mel.pretty_spectrogram(
            c.astype('float64'), fft_size=512, step_size=128, log=True)
        specs.append(spec)
    return specs


def transform(spectrogram):
    # this is for chaquopy to know where the binary(ires) is, relative to src/main/python
    model_path = join(dirname(__file__), '32-16-16-32.hdf5')

    model = load_model(model_path)
    sp_reshaped = np.expand_dims(spectrogram, -1)
    sp_reshaped = np.expand_dims(sp_reshaped, axis=0)

    pred = model.predict(sp_reshaped)

    pred_reshaped = np.squeeze(pred)
    return pred_reshaped


def back_to_audio(pred_spectrogram):
    
    recovered_audio_orig = mel.invert_pretty_spectrogram(
        pred_spectrogram, fft_size=512, step_size=128, log=True, n_iter=40)
    return recovered_audio_orig


def denoise(wav_path):
    specs = audio2spec(wav_path)
    denoised_audios = []
    for spec in specs:
        denoised_spec = transform(spec)
        denoised_audio = back_to_audio(denoised_spec)
        denoised_audios.append(denoised_audio)
    denoised = [item for sublist in denoised_audios for item in sublist]
    denoised = np.array(denoised,dtype='float64')
    return denoised
