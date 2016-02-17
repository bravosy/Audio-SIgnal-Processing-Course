#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
from numpy import cos, pi
from fractions import gcd
from scipy.fftpack import fft

#zero padding def
def zeroPadding(x,N):
	"""
	123456 -> 456000000123
	"""
	length = len(x)
	hm1 = np.floor((length+1.0)/2)
	hm2 = np.floor(length/2)
	dftbuffer = np.zeros(N)
	dftbuffer[:hm1] = x[hm2:]
	dftbuffer[-hm2:] = x[:hm2]
	return dftbuffer

def optimalZeroPad(x,fs,f):
	"""
	pad 0s so that length of signal x extends 
	to integer multiple cycles.
	"""
	length = len(x)
	#calculate 0s needed to be padded.
	sampleEveryPeriod = float(fs)/f
	integerCycle = np.ceil(length/sampleEveryPeriod)
	num = int(integerCycle * sampleEveryPeriod) - length
	y = np.zeros(int(num)+length)
	y[0:length] = x
	return num,y

def testRealEven(x):
	N = len(x)
	dftbuffer = zeroPadding(x,N)
	X = fft(dftbuffer)
	return dftbuffer,X

def suppressFreqDFTmodel(x,fs,N):
	"""
	suppress frequency less than 70Hz.
	"""
	length = len(x)
	window = np.hamming(length)
	x = x * window
	dftbuffer = zeroPadding(x,N)
	X = fft(dftbuffer)
	k = int(70 * N / fs)	# bins before k is attenuated
	X[0:k+1] = 0
	return X[0:N/2+1]


def zpFFTsizeExpt():
	f  = 110.0
	fs = 1000.0
	t = np.arange(0,1,1.0/fs)
	x = cos(2 * pi * f * t)
	xseg = x[0:256]
	w1 = np.hamming(256)
	w2 = np.hamming(512)
	X1 = fft(xseg * w1)
	X2 = fft(x[0:512] * w2)
	X3 = fft(xseg * w1, 512)
	mx1 = abs(X1)
	mx2 = abs(X2)
	mx3 = abs(X3)
	fx1 = fs * np.arange(256) / 256
	fx2 = fs * np.arange(512) / 512
	plt.xlim(0,150)
	plt.stem(fx1[0:80],mx1[0:80],'y')
	plt.stem(fx2[0:80],mx2[0:80],'r')
	plt.stem(fx2[0:80],mx3[0:80],'b')
	plt.show()

fs = 10000
f1 = 250
f2 = 10
f3 = 30
f4 = 1000
t = np.arange(0,1,1.0/fs)

x = cos(2*pi*f1*t) + cos(2*pi*f2*t) +cos(2*pi*f3*t) + cos(2*pi*f4*t) 
x = x[0:900]

