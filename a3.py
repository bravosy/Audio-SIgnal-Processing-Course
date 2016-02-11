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
	num0 = int(integerCycle) / float(f) * fs - length
	y = np.zeros(int(num0)+length)
	y[0:length] = x
	return num0,y


fs = 10000
f1 = 250
t = np.arange(0,1,1.0/fs)

x = cos(2*pi*f1*t) 
x = x[0:210]
n,y = optimalZeroPad(x,fs,f1)

plt.figure(1)
plt.subplot(211)
plt.stem(abs(fft(x)))

plt.subplot(212)
plt.stem(abs(fft(y)))
