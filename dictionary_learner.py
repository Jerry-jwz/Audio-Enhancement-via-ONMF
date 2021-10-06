from onmf import Online_NMF
import itertools
import numpy as np
from tqdm import trange
from scipy import signal
from scipy.io import wavfile


class ONMF_Dictionary_Learner():
    def __init__(self,
                 path,
                 n_components=100,
                 iterations=200,
                 sub_iterations=20,
                 num_patches=1000,
                 batch_size=20,
                 patch_length=7,
                 is_matrix=False,
                 is_color=True):
        '''
        batch_size = number of patches used for training dictionaries per ONMF iteration
        sources: array of filenames to make patches out of
        patches_array_filename: numpy array file which contains already read-in images
        '''
        self.path = path
        self.n_components = n_components
        self.iterations = iterations
        self.sub_iterations = sub_iterations
        self.num_patches = num_patches
        self.batch_size = batch_size
        self.patch_length = patch_length  # length in number of samples, not in seconds
        self.is_matrix = is_matrix
        self.is_color = is_color
        self.W = np.zeros(shape=(patch_length, n_components))
        self.code = np.zeros(shape=(n_components, iterations * batch_size))

        self.rate, self.audio = wavfile.read(self.path)
        if self.audio.ndim == 2:
            self.audio = np.delete(self.audio, 1, 1).ravel()
            print(self.audio.shape)
        self.freq, self.time, self.stft = signal.stft(self.audio, self.rate, nperseg=2048)
        self.spectrogram = np.abs(self.stft)

    def extract_random_patches(self):
        '''
        Extract 'num_patches' many random patches of given size
        '''
        x = self.spectrogram.shape
        k = self.patch_length
        X = np.zeros(shape=(len(self.freq) * k, 1, 1))
        for i in np.arange(self.num_patches):
            a = np.random.choice(x[1] - k)  # start time of the patch
            Y = self.spectrogram[:, a:a + k]
            # Y = Y.reshape(len(self.freq) * k, 1)  # size k*len(self.freq)
            if i == 0:
                X = Y
            else:
                X = np.append(X, Y, axis=1)  # X is class ndarray
        return X

    def train_dict(self):
        print('training dictionaries from patches...')
        '''
        Trains dictionary based on patches.
        '''
        W = self.W
        At = []
        Bt = []
        # code = self.code
        for t in trange(self.iterations):
            X = self.extract_random_patches()
            if t == 0:
                self.nmf = Online_NMF(X,
                                      n_components=self.n_components,
                                      iterations=self.sub_iterations,
                                      batch_size=self.batch_size,
                                      ini_dict=None,
                                      ini_A=None,
                                      ini_B=None,
                                      history=0,
                                      alpha=None)
                W, At, Bt, H = self.nmf.train_dict()
            else:
                self.nmf = Online_NMF(X,
                                      n_components=self.n_components,
                                      iterations=self.sub_iterations,
                                      batch_size=self.batch_size,
                                      ini_dict=W,
                                      ini_A=At,
                                      ini_B=Bt,
                                      history=self.nmf.history,
                                      alpha=None)
                # out of "sample_size" columns in the data matrix, sample "batch_size" randomly and train the dictionary
                # for "iterations" iterations
                W, At, Bt, H = self.nmf.train_dict()
                # code += H
        self.W = W
        print('dict_shape:', self.W.shape)
        # np.save('dictionary/dict_learned', self.W)
