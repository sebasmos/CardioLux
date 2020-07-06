import numpy as np
from scipy.misc import electrocardiogram
from scipy import signal
import scipy.integrate as integrate
import scipy.fftpack as ff
import matplotlib.pyplot as plt

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


def detrend_data(data, deg):
    detrend = [0]*deg
    for i in range(deg, len(data)-deg):
        diff = data[i-deg:i+deg]
        diff = sum(diff)/len(diff)
        detrend.append(data[i]-diff)
    return np.array(detrend+[0]*deg)


'''
PRUEBAS
'''

fs = 360
y = electrocardiogram()
y_filtered = signal.savgol_filter(y, 11, 3)
c = COSen(y, 30, 1)
print(c)

'''#plt.plot(y_detrend)
#plt.show()'''