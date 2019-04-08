import librosa
import numpy as np
import os
import code
import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import soundfile as sf
import sounddevice as sd
import queue

def extract_feature(file_name=None):
    if file_name:
        #print('Extracting', file_name)
        X, sample_rate = sf.read(file_name, dtype='float32')
    else:
        device_info = sd.query_devices(None, 'input')
        sample_rate = int(device_info['default_samplerate'])
        q = queue.Queue()
        def callback(i,f,t,s): q.put(i.copy())
        data = []
        with sd.InputStream(samplerate=sample_rate, callback=callback):
            while True:
                if len(data) < 100000: data.extend(q.get())
                else: break
        X = np.array(data)

    if X.ndim > 1: X = X[:,0]
    X = X.T

    # short term fourier transform
    stft = np.abs(librosa.stft(X))

    # mfcc (mel-frequency cepstrum)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz


# def extract_feature(file_name):
#     '''
#     extract_feature returns the features as specified by melspectrogram for each sound file
#
#     '''
#     Y, sample_rate = librosa.load(file_name, sr=2108)
#
#     # features = librosa.feature.mfcc(y=Y, sr=sample_rate, n_mfcc=120)
#     features = librosa.feature.melspectrogram(y=Y, sr=sample_rate, n_mels=120)
#
#     return features


root = "C:/Users/deyso/PycharmProjects/sound/mp3folder"

os.chdir(root + '/wavfile/')
filenames = os.listdir('.')
#print(len(filenames))
list=[]
features= np.empty((0,193))
for wav_file in filenames:
    lis1={}
    print(wav_file)
    mfccs, chroma, mel, contrast, tonnetz = extract_feature(wav_file)
    ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    list.append(ext_features)
    #dest = root + '/npy_files_TOTAL_train/' + str(os.path.splitext(wav_file)[0])
    #print(dest)
    features=np.vstack([features,ext_features])
dest = root + '/npy_files_TOTAL_train/' + 'features.npy'
#print(features)
np.save(dest, features)

