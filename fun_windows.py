import numpy as np
from fun_features import find_peaks


''' 
  data: signal ECG
  N: window size
'''


def window_fun2(data, fs=500, m=5):
    peaks, t, t_peaks, data_peaks, data1 = find_peaks(data, fs)
    windows = []
    n = m-2  # peaks size: m+2
    for i in range(len(peaks)-(n+1)):
        if i == 0:
            windows.append(data[0:peaks[i+n+1]+round(0.5*fs)])
        else:
            windows.append(data[peaks[i]-round(0.5*fs):peaks[i+n+1]+round(0.5*fs)])
    return windows


def detrend_data(data, deg):
    detrend = [0]*deg
    for i in range(deg, len(data)-deg):
        diff = data[i-deg:i+deg]
        diff = sum(diff)/len(diff)
        detrend.append(data[i]-diff)
    return np.array(detrend+[0]*deg)
