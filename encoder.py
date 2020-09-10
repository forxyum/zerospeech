import os
import math
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

folder_path = './shared/databases/english_small/train/unit/'
sound_names = os.listdir(folder_path)
max_len = 0
channels = 3


def load_sound_files():
	start_time = time.time()
	sounds = []
	for file in os.listdir(folder_path):
		sample, sample_rate = librosa.load(folder_path+file, sr = 16000)
		sounds.append(sample)
	print('Loading the files took {} seconds'.format(time.time()-start_time))
	return sounds

def get_mfccs(samples):
	start_time = time.time()
	mfccs = []
	for sample in samples:
		base = librosa.feature.mfcc(y=sample,sr=16000,n_mfcc=13)
		delta = librosa.feature.delta(base)
		delta2 = librosa.feature.delta(base,order=2)
		#sep = np.full((1,np.shape(base)[1]),600)
		mfccs.append(np.vstack([base,delta,delta2]))
		global max_len
		if np.shape(base)[1] > max_len:
			max_len = np.shape(base)[1]

	#making the vectors the same size
	for i,mfcc in enumerate(mfccs):
		if np.shape(mfcc)[1] < max_len:
			pad_width = max_len - np.shape(mfcc)[1]
			mfccs[i] = np.pad(mfcc, pad_width=((0,0),(0,pad_width)),mode='constant')
	print('Getting the mfccs took {} seconds'.format(time.time()-start_time))
	return mfccs

def plot_mfccs(mfcc):
	plt.subplot(1,3,1)
	librosa.display.specshow(mfcc[0:13],x_axis='time')
	plt.colorbar()

	plt.subplot(1,3,2)
	librosa.display.specshow(mfcc[13:26],x_axis='time')
	plt.colorbar()

	plt.subplot(1,3,3)
	librosa.display.specshow(mfcc[26:39],x_axis='time')
	plt.colorbar()

	plt.show()

sound_files = load_sound_files()
mfccs = get_mfccs(sound_files)
#plot_mfccs(mfccs[0])

mfccs = np.array(mfccs)
mfccs = mfccs.reshape(np.shape(mfccs)[0],39,max_len,1)


def get_conv_model():
	model = Sequential()
	print(channels)
	model.add(Conv2D(768,(3,3),activation='relu',input_shape=(39,max_len,channels,1)))
	model.add(Conv2D(768,(3,3),activation='relu'))
	model.summary()

model = get_conv_model()
print('-----------------------------------------------------------------------------')
