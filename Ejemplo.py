import numpy as np
import biosppy as Bp
import scipy as sp
import signal as sg
import scipy.integrate as integrate
import pywt

'''Extract ECG Features with BioSppy
@Misc{,
  author = {Carlos Carreiras and Ana Priscila Alves and Andr\'{e} Louren\c{c}o and Filipe Canento and Hugo Silva and Ana Fred and others},
  title = {{BioSPPy}: Biosignal Processing in {Python}},
  year = {2015--},
  url = "https://github.com/PIA-Group/BioSPPy/",
  note = {[Online; accessed <today>]}
}'''
def ECG_Extraction_time(data,Fs):
    Qrs_Interval = 0.094 
    Look_Sample = int((Fs*Qrs_Interval)/2);  
    ceros = np.zeros(Look_Sample)
    rpeaks, = Bp.ecg.hamilton_segmenter(data,Fs)
    rpeaks, = Bp.ecg.correct_rpeaks(data,rpeaks,Fs,0.05)
    template, rpeaks = Bp.ecg.extract_heartbeats(data,rpeaks,Fs,0.2,0.4)
    hr_i, hr = Bp.tools.get_heart_rate(rpeaks,Fs,True,3)
    size_signal = len(data)
    T = (size_signal-1)/Fs
    t = np.linspace(0,T,size_signal,True)
    t_hr = t[hr_i]
    t_temp = np.linspace(-0.2,0.4,template.shape[1],False)
    hr = np.mean(hr) #Heart rate (bpm)
    t_hr = np.mean(t_hr) #Heart rate (second)
    Rr_Region = []
    for i in range(0,len(t_temp)-1):
        Rr_Region.append(t_temp[i+1]-t_temp[i])    
    Rr_interval = np.mean(Rr_Region) #RR Interval (second)
    #QRS Interval------------------------------------------------------------------------
    data_Shifted = np.concatenate((ceros,data,ceros))
    R_loc = rpeaks+Look_Sample*np.ones(len(rpeaks))
    Q_peak = []
    S_peak = []
    Q_idx = []
    S_idx = []
    Q_i = []
    S_i = []
    for i in range(0,len(R_loc)):
        init = int(R_loc[i]-Look_Sample)
        Final = int(R_loc[i])
        Q_peak.append(np.min(data_Shifted[init:Final]))
        Q_idx.append(np.min(np.array(np.where(data_Shifted[init:Final] == Q_peak[i])[0])))
    
    for i in range(0,len(R_loc)):
        init = int(R_loc[i])
        Final = int(R_loc[i]+Look_Sample)
        S_peak.append(np.min(data_Shifted[init:Final]))
        S_idx.append(np.min(np.array(np.where(data_Shifted[init:Final] == S_peak[i])[0])))
 

    for i in range(0,len(R_loc)):
        S_i.append(R_loc[i]+S_idx[i]-1)
        Q_i.append(R_loc[i]-Look_Sample+Q_idx[i]-1)

    Q_loc = Q_i-Look_Sample*np.ones(len(Q_i))
    S_loc = S_i-Look_Sample*np.ones(len(S_i))
    Qrs_Region = S_loc-Q_loc 
    Qrs_Interval = np.mean(Qrs_Region/Fs) #QRS_interval
    #Q and S peak
    Q_peak = np.mean(Q_peak)
    S_peak = np.mean(S_peak)

    #Energy of signal
    Energy_aux = ((np.abs(data))**2)
    Energy_aux = Energy_aux/np.sum(Energy_aux)
    Energy = integrate.simps(Energy_aux,t) #Energy Signal
    #Centroid 
    t_e = np.dot(t,Energy_aux)
    if Energy == 0 or t_e == 0:
        Centroid = 0
    else:
        Centroid = t_e/Energy
    #auto-correlate
    data_aux = data/np.sum(data)
    Corr = float(np.correlate(data_aux,data_aux))

    mean_diff = np.mean(np.diff(data))
    # Entropy 
    std = np.std(data)
    if std == 0:
        Entropy = 0
    mean = np.mean(data)
    m = np.min(data)
    if np.max(data)<0:
        M = -np.max(data)
    else:
        M = np.max(data)
    
    probability = np.linspace(m,M,len(data))
    gauss_pdf = sp.stats.norm.pdf(probability, mean, std)
    gauss = gauss_pdf/np.sum(gauss_pdf)
    if np.sum(gauss) == 0:
        Entropy = 0

    gauss = gauss[np.where(gauss != 0)]
    Entropy = -np.sum(gauss*np.log2(gauss))/np.log2(len(data))

    return hr, t_hr,Rr_interval,Q_peak,S_peak,Qrs_Interval, Energy,Centroid,Corr, mean_diff, Entropy

def Frecuency_Features(Data, Fs):

    #Espectral density and Espectral Entropy 
    Espectral_Entropy = []
    _, Psd_aux = sp.signal.periodogram(Data,Fs) #Espectral density
    Psd = np.mean(Psd_aux)
    for i in range(0,len(Psd_aux)):
        Espectral_Entropy.append(Psd_aux[i]/np.sum(Psd_aux))
    E_h=-np.sum(Espectral_Entropy*np.log10(Espectral_Entropy)) #Espectral Entropy 
    return Psd, E_h 

def COSen(Data, m, r, Fs):
    x1 = np.matrix(np.zeros(m-1))
    x2 = np.matrix(np.zeros(m))
    N = len(Data)
    for i in range(0,N-m):
        x1 = np.vstack([x1, Data[i:i+m-1]])
        x2 = np.vstack([x2, Data[i:i+m]])
    x1 = np.delete(x1,0,0)
    x2 = np.delete(x2,0,0)
    return x2

def WaveletFeat(Data):
    ''' Wavelet extraction process is advanced  for trying to describe Premature
    Ventricular Complex Arrhythmia, seems like the best results were verified
    with db2 wavelet mother'''
    print("WaveletFeat seems to be workig")

fs = 360
y = sp.misc.electrocardiogram()
x1 = COSen(y,100, 30,360)
y = np.matrix(np.zeros(10))
print(len(x1))
print(type(x1))
