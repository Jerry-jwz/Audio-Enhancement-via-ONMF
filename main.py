from scipy.io import wavfile
import scipy.io
import os.path
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from scipy.io.wavfile import write
import IPython.display as ipd

from dictionary_learner import *
from audio_enhancement import *
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import NMF


def Denoising(training_data, test_data, method, n_components=[50, 10], patch_length=1):
    if len(training_data) != len(n_components):
        raise ValueError("The number of training audio files is not equal to the number of dictionaries!")

    if method == 'ONMF':
        dicts = []
        n_components = [50, 10]
        for i, path in enumerate(training):
            reconstructor = ONMF_Dictionary_Learner(path=path,
                                                    n_components=n_components[i],
                                                    iterations=100,
                                                    sub_iterations=5,
                                                    patch_length=patch_length,
                                                    batch_size=50,
                                                    num_patches=100,
                                                    is_matrix=False,
                                                    is_color=True)

            print('A.spectrogram.shape', reconstructor.spectrogram.shape)
            reconstructor.train_dict()

            dictionary = reconstructor.W  # trained dictionary
            print("dictionary.shape = ", dictionary.shape)
            dicts.append(dictionary)

    elif method == 'NMF':
        dicts = []
        for i in range(len(training)):
            rate, src = wavfile.read(training[i])
            if src.ndim == 1:
                mono = src
            elif src.ndim == 2:
                mono = np.delete(src, 1, 1).ravel()

            f, t, stft = signal.stft(mono, rate, nperseg=2048)
            Spec = np.abs(stft)

            model = NMF(n_components=n_components[i], init='random', max_iter=9999, alpha=0.0,
                        solver='mu', beta_loss=2)
            dicts.append(model.fit_transform(Spec))

        normalizer = Normalizer().fit(dicts[0].T)  # fit does nothing.
        dicts[0] = normalizer.transform(dicts[0].T).T
        normalizer = Normalizer().fit(dicts[1].T)  # fit does nothing.
        dicts[1] = normalizer.transform(dicts[1].T).T

    else:
        raise ValueError("Invalid Method!")

    separator = Audio_Separation(input_dictionaries=dicts,
                                 audio_file=test_data,
                                 patch_length=patch_length,
                                 trunc_rate=1)
    specs = separator.separate_audio(recons_resolution=patch_length,
                                     alpha=100)  # alpha is the sparsity of the coding matrices

    noise_spec = specs[1].copy()
    voice_spec = specs[0].copy()
    voice_stft = topic_to_stft(separator.stft, noise_spec + voice_spec, voice_spec)
    noise_stft = topic_to_stft(separator.stft, noise_spec + voice_spec, noise_spec)

    return voice_stft, noise_stft, separator.rate


if __name__ == '__main__':
    patch_length = 1
    training = ["Data/audio/whitman.wav", "Data/audio/WhiteNoise.wav"]
    test = "Data/audio/Shakes_WhiteNoise.wav"
    n_components = [50, 10]

    # ONMF-based Denoising
    voice_stft, noise_stft, rate = Denoising(training, test, "ONMF", n_components=n_components,
                                             patch_length=patch_length)

    topic_1_audio = signal.istft(voice_stft)[1]
    write("Data/audio_output/shakes_whitenoise/voice_onmf_(50,10)_a=100.wav", rate,
          np.array(topic_1_audio, dtype=np.int16))
    topic_2_audio = signal.istft(noise_stft)[1]
    write("Data/audio_output/shakes_whitenoise/noise_onmf_(50,10)_a=100.wav", rate,
          np.array(topic_2_audio, dtype=np.int16))
    print("ONMF-based denoising is done!")


    # NMF-based Denoising
    voice_stft, noise_stft, rate = Denoising(training, test, "NMF", n_components=n_components)

    topic_1_audio = signal.istft(voice_stft)[1]
    write("Data/audio_output/shakes_whitenoise/voice_nmf_(50,10)_a=100.wav", rate,
          np.array(topic_1_audio, dtype=np.int16))
    topic_2_audio = signal.istft(noise_stft)[1]
    write("Data/audio_output/shakes_whitenoise/noise_nmf_(50,10)_a=100.wav", rate,
          np.array(topic_2_audio, dtype=np.int16))
    print("NMF-based denoising is done!")
