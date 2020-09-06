#!/usr/bin/env python

import numpy as np
from scipy.signal import butter, lfilter
from scipy import stats
from scipy import signal
from scipy.misc import electrocardiogram
from scipy import signal
import scipy.integrate as integrate
import scipy.fftpack as ff

# Import features 
import fun_features
import fun_windows
import fun_cosEn
import numpy as np

def detect_peaks(ecg_measurements,signal_frequency,gain):

        """
        Method responsible for extracting peaks from loaded ECG measurements data through measurements processing.

        This implementation of a QRS Complex Detector is by no means a certified medical tool and should not be used in health monitoring. 
        It was created and used for experimental purposes in psychophysiology and psychology.
        You can find more information in module documentation:
        https://github.com/c-labpl/qrs_detector
        If you use these modules in a research project, please consider citing it:
        https://zenodo.org/record/583770
        If you use these modules in any other project, please refer to MIT open-source license.

        If you have any question on the implementation, please refer to:

        Michal Sznajder (Jagiellonian University) - technical contact (msznajder@gmail.com)
        Marta lukowska (Jagiellonian University)
        Janko Slavic peak detection algorithm and implementation.
        https://github.com/c-labpl/qrs_detector
        https://github.com/jankoslavic/py-tools/tree/master/findpeaks
        
        MIT License
        Copyright (c) 2017 Michal Sznajder, Marta Lukowska
    
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:
        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.
        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.

        """


        filter_lowcut = 0.001
        filter_highcut = 15.0
        filter_order = 1
        integration_window = 30  # Change proportionally when adjusting frequency (in samples).
        findpeaks_limit = 0.35
        findpeaks_spacing = 100  # Change proportionally when adjusting frequency (in samples).
        refractory_period = 240  # Change proportionally when adjusting frequency (in samples).
        qrs_peak_filtering_factor = 0.125
        noise_peak_filtering_factor = 0.125
        qrs_noise_diff_weight = 0.25


        # Detection results.
        qrs_peaks_indices = np.array([], dtype=int)
        noise_peaks_indices = np.array([], dtype=int)

 # Measurements filtering - 0-15 Hz band pass filter.
        fitered_Savitzky_Golay = signal.savgol_filter(ecg_measurements, 11, 3)
        filtered_ecg_measurements = bandpass_filter(ecg_measurements, lowcut=filter_lowcut, highcut=filter_highcut, signal_freq=signal_frequency, filter_order=filter_order)

        fitered_Savitzky_Golay[:5] = fitered_Savitzky_Golay[5]

        # Derivative - provides QRS slope information.
        differentiated_ecg_measurements = np.ediff1d(fitered_Savitzky_Golay)


        # Squaring - intensifies values received in derivative.
        squared_ecg_measurements = differentiated_ecg_measurements ** 2

        # Moving-window integration.
        integrated_ecg_measurements = np.convolve(squared_ecg_measurements, np.ones(integration_window)/integration_window)

        # Fiducial mark - peak detection on integrated measurements.
        detected_peaks_indices = findpeaks(data=integrated_ecg_measurements,
                                                     limit=findpeaks_limit,
                                                     spacing=findpeaks_spacing)

        detected_peaks_values = integrated_ecg_measurements[detected_peaks_indices]

        return detected_peaks_values,detected_peaks_indices

 
def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
        """
        Method responsible for creating and applying Butterworth filter.
        :param deque data: raw data
        :param float lowcut: filter lowcut frequency value
        :param float highcut: filter highcut frequency value
        :param int signal_freq: signal frequency in samples per second (Hz)
        :param int filter_order: filter order
        :return array: filtered data
        """
        nyquist_freq = 0.5 * signal_freq
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        b, a = butter(filter_order, [low, high], btype="band")
        y = lfilter(b, a, data)
        return y

def findpeaks(data, spacing=1, limit=None):
        """
        Janko Slavic peak detection algorithm and implementation.
        https://github.com/jankoslavic/py-tools/tree/master/findpeaks
        Finds peaks in `data` which are of `spacing` width and >=`limit`.
        :param ndarray data: data
        :param float spacing: minimum spacing to the next peak (should be 1 or more)
        :param float limit: peaks should have value greater or equal
        :return array: detected peaks indexes array
        """
        len = data.size
        x = np.zeros(len + 2 * spacing)
        x[:spacing] = data[0] - 1.e-6
        x[-spacing:] = data[-1] - 1.e-6
        x[spacing:spacing + len] = data
        peak_candidate = np.zeros(len)
        peak_candidate[:] = True
        for s in range(spacing):
            start = spacing - s - 1
            h_b = x[start: start + len]  # before
            start = spacing
            h_c = x[start: start + len]  # central
            start = spacing + s + 1
            h_a = x[start: start + len]  # after
            peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

        ind = np.argwhere(peak_candidate)
        ind = ind.reshape(ind.size)
        if limit is not None:
            ind = ind[data[ind] > limit]
        return ind
'''
def WaveletFeat(ecg_measurements,signal_frequency,gain):
 Wavelet extraction process is advanced  for trying to describe Premature
    Ventricular Complex Arrhythmia, seems like the best results were verified
    with db2 wavelet mother
    cA, cD = pywt.dwt(ecg_measurements,'db2')
    
   # print("1. Wavelet features are: ", cA[1:5])
    
    wavelet = pywt.Wavelet('db2')
    
    phi, psi, x = wavelet.wavefun(level=5)
    
    return gain*2  
'''  

## FUNCIONES PARA AGREGAR AL MODELO ####

'''
1. r_peak_mean: Pico R
2. h_rate: Heart rate (bpm)
3. rr_interval: RR interval
4. qrs_Interval: QRS interval
5. q_peak: Q peak
6. s_peak: S peak 
'''

def qrs_features(data, fs):
    min_peaks = 0.5

    # detrend data
    data_detrend = detrend_data(data, 10)

    # Find R peaks locations
    peaks, _ = signal.find_peaks(data_detrend, min_peaks)
    r_peaks = data_detrend[peaks]
    r_peak_mean = np.mean(r_peaks)  # 1. R peak

    # time
    step = len(data_detrend) / fs
    t = np.linspace(0, step, len(data_detrend), True)

    # Hearth rate
    h_rate = (len(peaks)/t[len(t)-1])*60  # 2. Hearth rate (bpm)

    # RR Interval
    t_peaks = t[peaks]
    t_diff = []
    for i in range(1, len(t_peaks)):
        diff_t = t_peaks[i]-t_peaks[i-1]
        t_diff.append(diff_t)

    rr_interval = np.mean(np.array(t_diff))  # 3. RR interval (seconds)

    # QRS Interval
    qrs_interval = 0.094
    look_sample = int((fs * qrs_interval) / 2)
    zeros = np.zeros(look_sample)
    data_concatenated = np.concatenate((zeros, data_detrend, zeros))
    r_loc = peaks + look_sample*np.ones(len(peaks))
    q_peak = []
    s_peak = []
    q_idx = []
    s_idx = []
    q_i = []
    s_i = []
    for i in range(0, len(r_loc)):
        init = int(r_loc[i] - look_sample)
        final = int(r_loc[i])
        q_peak.append(np.min(data_concatenated[init:final]))
        q_idx.append(np.min(np.array(np.where(data_concatenated[init:final] == q_peak[i])[0])))

    for i in range(0, len(r_loc)):
        init = int(r_loc[i])
        final = int(r_loc[i] + look_sample)
        s_peak.append(np.min(data_concatenated[init:final]))
        s_idx.append(np.min(np.array(np.where(data_concatenated[init:final] == s_peak[i])[0])))

    for i in range(0, len(r_loc)):
        s_i.append(r_loc[i] + s_idx[i] - 1)
        q_i.append(r_loc[i] - look_sample + q_idx[i] - 1)

    q_loc = q_i - look_sample * np.ones(len(q_i))
    s_loc = s_i - look_sample * np.ones(len(s_i))
    qrs_region = s_loc - q_loc
    qrs_interval = np.mean(qrs_region/fs)  # 4. QRS_interval

    # Q and S peak
    q_peak = np.mean(q_peak)  # 5. Q peak
    s_peak = np.mean(s_peak)  # 6. S peak

    return r_peak_mean, h_rate, rr_interval, qrs_interval, q_peak, s_peak

    
def detrend_data(data, deg):
    detrend = [0]*deg
    for i in range(deg, len(data)-deg):
        diff = data[i-deg:i+deg]
        diff = sum(diff)/len(diff)
        detrend.append(data[i]-diff)
    return np.array(detrend+[0]*deg)


'''
1. Energy: Signal Energy
2. Corr: Autocorrelation 
3. Ent_Shannon: Shannon Entropy
'''


def time_features(data, fs):
    data = data/sum(abs(data))
    data = detrend_data(data, 10)
    step = len(data) / fs
    t = np.linspace(0, step, len(data), True)

    # Energy
    energy_aux = ((np.abs(data)) ** 2)
    energy_aux = energy_aux / np.sum(energy_aux)
    energy = integrate.simps(energy_aux, t)  # 1. Signal Energy

    # Autocorrelation
    corr = float(np.correlate(data, data))  # 2. Autocorrelation

    # Shannon Entropy
    pro = [np.mean(data == valor) for valor in set(data)]
    # Cuando la division sea cero, cambiar ent_shannon por cero.
    ent_shannon = sum(-q*np.log2(q) for q in pro)  # 3. Shannon Entropy

    # Centroid
    centroid = np.dot(t, energy_aux)/sum(energy_aux)  # 4. Centroid

    return energy, corr, ent_shannon, centroid



'''
1. psd: Spectral density
2. ent_esp: Spectral Entropy
3. cent_f: Spectral Centroid
'''


def frecuency_features(data, fs):

    spectral_entropy = []
    _, psd_aux = signal.periodogram(data, fs)  # 1. Spectral density
    psd = np.mean(psd_aux)
    for i in range(0, len(psd_aux)):
        spectral_entropy.append(psd_aux[i]/np.sum(psd_aux))
    ent_esp = -np.sum(spectral_entropy*np.log10(spectral_entropy))  # 2. Spectral Entropy

    # Spectral Centroid
    data_f = ff.fft(data)
    data_f = data_f[0:round(len(data_f)/2)]
    f = ff.rfftfreq(len(data), 1/fs)
    f = f[0:round(len(f)/2)]
    cent_f = np.dot(f, abs(data_f))/sum(abs(data_f))  # 3. Spectral Centroid

    return psd, ent_esp, cent_f

'''
Return the Coefficient of Sample Entropy CosEn
Input: 
    data: Signal
    m: length of sequences to be compared 
    r: tolerance 
Output: 
    cosen: CosEn
'''


def COSen(data, m):
    cosen = 0
    n = len(data)
    b1 = 0
    b2 = 0
    r = 0.2*np.std(data)
    # Compute b1
    aux1 = np.array([data[i: i + m]for i in range(n-m)])
    aux2 = np.array([data[i: i + m] for i in range(n-m+1)])

    b1 = np.sum([np.sum(np.abs(i-aux2).max(axis=1) <= r)-1 for i in aux1])

    # Compute b2
    m = m+1
    aux3 = np.array([data[i: i+m] for i in range(n-m+1)])
    b2 = np.sum([np.sum(np.abs(i-aux3).max(axis=1) <= r)-1 for i in aux3])

    # Compute COSen
    cosen = -np.log(b2/b1) - np.log(2*r) - np.log(np.mean(data))
    return cosen



#### END OF ADDING FEATURES ####
def fun_extract_data(data):
    fs = 500
    features = []
    for i in data:
        windows = fun_windows.window_fun2(i, fs, m=5)  # m: Peaks number
        for j in windows:
            fea1, fea2, fea3, fea4, fea5, fea6 = fun_features.qrs_features(j, fs)
            fea7, fea8, fea9 = fun_features.frecuency_features(j, fs)
            #fea10 = fun_cosEn.cos_en(j)
            fea11, fea12, fea13, fea14 = fun_features.time_features(j, fs)
            #features_1 = [fea1, fea2, fea3, fea4, fea5, fea6, fea7, fea8, fea9, fea10, fea11, fea12, fea13, fea14]
            # for testing: ARRANGE FEAT 10!!!!
            features_1 = [fea1, fea2]
            features.append(features_1)
    features = np.array(features)
    feats_reshape = features.reshape(1, -1)
    return feats_reshape

def get_12ECG_features(data, header_data):

    tmp_hea = header_data[0].split(' ')
    ptID = tmp_hea[0]
    num_leads = int(tmp_hea[1])
    sample_Fs= int(tmp_hea[2])
    gain_lead = np.zeros(num_leads)
    
    for ii in range(num_leads):
        tmp_hea = header_data[ii+1].split(' ')
        gain_lead[ii] = int(tmp_hea[2].split('/')[0])

    # for testing, we included the mean age of 57 if the age is a NaN
    # This value will change as more data is being released
    for iline in header_data:
        if iline.startswith('#Age'):
            tmp_age = iline.split(': ')[1].strip()
            age = int(tmp_age if tmp_age != 'NaN' else 57)
        elif iline.startswith('#Sex'):
            tmp_sex = iline.split(': ')[1]
            if tmp_sex.strip()=='Female':
                sex =1
            else:
                sex=0
        elif iline.startswith('#Dx'):
            label = iline.split(': ')[1].split(',')[0]

#   Add windowing function
# 
#for i in range(data.shape[0]): # tiene tamaño 12
#        header = headers[i]
#        window = get_windows(recordings[i],fs,n)
#        energy, corr, ent_shannon, centroid =   time_features(window, sample_Fs)
#        feat = np.hstack([energy, corr, ent_shannon, centroid ...])
#        data.append(feat)
 #contiene windows para trainingset y c/una de las 12 señales
#'''
#   We are only using data from lead1
    peaks,idx = detect_peaks(data[0],sample_Fs,gain_lead[0])
 
#   1. Wavelet feature extraction - PCV
    #wavet_feat = WaveletFeat(data[0],sample_Fs,gain_lead[0])
    
#   2. QRS features - LBBB 
    #r_peak_mean, h_rate, rr_interval, qrs_interval, q_peak, s_peak = qrs_features(data[0],sample_Fs)

#   3. Time features 
    #energy, corr, ent_shannon, centroid =   time_features(data[0], sample_Fs)
    
#   4. Frequency features 
    #psd, ent_esp, cent_f =  frecuency_features(data[0], sample_Fs)
#   5. Entropy Sampled Coefficient 
    #''' m : embedded dimension; defined by MATLAB AS 30 '''
    #m = 30
   # cosen = COSen(data[0], m)
#   mean
    
    #fea1, fea2, fea3, fea4, fea5, fea6 = fun_features.qrs_features(data[0], sample_Fs)
    #fea7, fea8, fea9 = fun_features.frecuency_features(data[0], sample_Fs)
    #fea10 = fun_cosEn.cos_en(data[0])
    #fea11, fea12, fea13, fea14 = fun_features.time_features(data[0],sample_Fs)
    mean_RR = np.mean(idx/sample_Fs*1000)
    mean_Peaks = np.mean(peaks*gain_lead[0])

#   median
    median_RR = np.median(idx/sample_Fs*1000)
    median_Peaks = np.median(peaks*gain_lead[0])

#   standard deviation
    std_RR = np.std(idx/sample_Fs*1000)
    std_Peaks = np.std(peaks*gain_lead[0])

#   variance
    var_RR = stats.tvar(idx/sample_Fs*1000)
    var_Peaks = stats.tvar(peaks*gain_lead[0])

#   Skewness
    skew_RR = stats.skew(idx/sample_Fs*1000)
    skew_Peaks = stats.skew(peaks*gain_lead[0])

#   Kurtosis
    kurt_RR = stats.kurtosis(idx/sample_Fs*1000)
    kurt_Peaks = stats.kurtosis(peaks*gain_lead[0])

    #frame = [r_peak_mean, h_rate, rr_interval, qrs_interval, q_peak, s_peak,energy, corr, ent_shannon, centroid,psd, ent_esp, cent_f]



    #features = np.hstack([age,sex,mean_RR,mean_Peaks,median_RR,median_Peaks,std_RR,std_Peaks,var_RR,var_Peaks,skew_RR,skew_Peaks,kurt_RR,kurt_Peaks, fea1, fea2, fea3, fea4, fea5, fea6, fea7, fea8, fea9, fea11, fea12, fea13, fea14 ])

    features = np.hstack([age,sex,mean_RR,mean_Peaks,median_RR,median_Peaks,std_RR,std_Peaks,var_RR,var_Peaks,skew_RR,skew_Peaks,kurt_RR,kurt_Peaks])

    return features


