from onmf import update_code_within_radius
import itertools
import numpy as np
from scipy import signal
from scipy.io import wavfile

from time import time

DEBUG = False


class Audio_Separation:
    def __init__(self, input_dictionaries, audio_file, patch_length, trunc_rate=1):
        self.input_dictionaries = input_dictionaries
        self.audio_file = audio_file
        self.rate, self.audio = wavfile.read(self.audio_file)
        if (self.audio.ndim) == 2:
            self.audio = np.delete(self.audio, 1, 1).ravel()
            print(self.audio.shape)

        self.freq, self.time, self.stft = signal.stft(self.audio[0: len(self.audio) // trunc_rate], self.rate,
                                                      nperseg=2048)  # you can freely truncate in order to accelerate testing

        self.spectrogram = np.abs(self.stft)
        self.W = np.concatenate(input_dictionaries, axis=1)
        self.patch_length = patch_length

    def separate_audio(self, recons_resolution, alpha=0):
        print('reconstructing given network...')

        A = self.spectrogram
        A_matrix = A.reshape(-1, A.shape[1])
        [m, n] = A_matrix.shape
        separated_specs = [np.zeros(shape=A.shape) for x in self.input_dictionaries]
        zeroed_dicts = [np.zeros(shape=x.shape) for x in self.input_dictionaries]
        separated_dicts = []
        for i in range(len(self.input_dictionaries)):
            _temp = zeroed_dicts[i]
            zeroed_dicts[i] = self.input_dictionaries[i]
            separated_dicts.append(np.concatenate(zeroed_dicts, axis=1))
            zeroed_dicts[i] = _temp
        for x in zeroed_dicts:
            print("x.shape = ", x.shape)
        A_overlap_count = np.zeros(shape=(A.shape[0], A.shape[1]))
        k = self.patch_length
        t0 = time()
        c = 0
        num_rows = np.floor((A_overlap_count.shape[0] - k) / recons_resolution).astype(int)
        num_cols = np.floor((A_overlap_count.shape[1] - k) / recons_resolution).astype(int)

        for i in np.arange(0, A_overlap_count.shape[1] - k, recons_resolution):
            patch = A[:, i:i + k]
            patch = patch.reshape((-1, 1))
            code = update_code_within_radius(patch, self.W, H0=None, r=None, alpha=alpha, sub_iter=100,
                                             stopping_diff=0.01)
            patch_recons_list = [np.dot(D, code).reshape(len(A), k) for D in separated_dicts]

            # now paint the reconstruction canvases
            for x in itertools.product(np.arange(len(A)), np.arange(k)):
                c = A_overlap_count[x[0], i + x[1]]
                for spec in range(len(separated_specs)):
                    separated_specs[spec][x[0], i + x[1]] = (c * separated_specs[spec][x[0], i + x[1]] +
                                                             patch_recons_list[spec][x[0], x[1]]) / (c + 1)
                A_overlap_count[x[0], i + x[1]] += 1

            # progress status
            if (i // recons_resolution) % 10 == 0:
                print('reconstructing %ith patch out of %i' % (i / recons_resolution, num_cols))
        print('Reconstructed in %.2f seconds' % (time() - t0))
        # for i in range(len(separated_specs)):
        #     np.save('spec_arr_' + str(i), separated_specs[i])
        return separated_specs


def topic_to_stft(stft, NMF_Sxx, topic):
    '''
    Post-processing step to add phase information to the approximate spectrograms
    '''
    output = stft.copy()
    for i in range(len(output)):
        for j in range(len(output[0])):
            if NMF_Sxx[i][j] == 0:
                output[i][j] = 0
            else:
                output[i][j] *= topic[i][j]/NMF_Sxx[i][j]
    return output