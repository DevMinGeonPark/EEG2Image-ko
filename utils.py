import os
from glob import glob
from natsort import natsorted
import tensorflow as tf
import skimage.io as skio
from skimage.transform import resize
from tifffile import imwrite
import numpy as np
import cv2
from save_figure import save_figure, save_figure_condition
import h5py
from functools import partial
import tensorflow_io as tfio
import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import style

style.use('seaborn') # matplotlib 기반 통계 데이터 시각화 라이브러리

def preprocess_data(X, Y, P, resolution=128): # X: eeg, Y: label, P: path (전처리 단계)
	X = tf.squeeze(X, axis=-1) #사이즈가 1인 차원을 제거
	max_val = tf.reduce_max(X)/2.0 #최대값을 구함
	X = (X - max_val) / max_val #최대값을 2로 나눔
	X = tf.transpose(X, [1, 0]) #전치행렬
	X = tf.cast(X, dtype=tf.float32) #데이터 타입을 float32로 변환
	Y = tf.argmax(Y) #Y의 최대값의 인덱스를 구함
	I = tf.image.decode_jpeg(tf.io.read_file(P), channels=3) #이미지를 읽어서 디코딩
	I = tf.image.resize(I, (resolution, resolution)) #이미지를 128*128로 리사이즈
	I = (tf.cast( I, dtype=tf.float32 ) - 127.5) / 127.5 #이미지를 -1~1로 정규화

	return X, Y, I

def load_complete_data(X, Y, P, batch_size=16, dataset_type='train'): # X: eeg, Y: label, P: path (데이터셋 로드 단계)
	if dataset_type == 'train':
		dataset = tf.data.Dataset.from_tensor_slices((X, Y, P)).map(preprocess_data).shuffle(buffer_size=2*batch_size).batch(batch_size, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE) 
	else:
		dataset = tf.data.Dataset.from_tensor_slices((X, Y, P)).map(preprocess_data).batch(batch_size, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
	return dataset


def show_batch_images(X, save_path, Y=None): # X: eeg, Y: label (이미지 시각화 단계)
	X = np.clip( np.uint8( ((X.numpy() * 0.5) + 0.5) * 255 ), 0, 255) #이미지를 0~255로 변환 (clip: 배열 값들을 지정된 범위 내로 제한하는 함수)
	# X = X[:16]
	col = 4
	row = X.shape[0] // col
	for r in range(row):
	    for c in range(col):
	        plt.subplot2grid((row, col), (r, c), rowspan=1, colspan=1) #그리드 형태로 이미지를 출력
	        plt.grid('off') #그리드를 끔
	        plt.axis('off') #축을 끔
	        if Y is not None: 
	        	plt.title('{}'.format(Y[r*col+c])) #이미지의 제목을 라벨로 설정
	        plt.imshow(X[r*col+c]) #이미지를 출력
	plt.tight_layout() #그래프의 여백을 최소화
	plt.savefig(save_path) #그래프를 저장
	plt.clf() #그래프를 초기화
	plt.close() #그래프를 닫음