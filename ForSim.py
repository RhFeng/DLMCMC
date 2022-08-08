import numpy as np
import sys
sys.path.append("../input/mps-mcmc/")
from misc_inversion import *

def ForSim(input_data, nt_limit, n_theta, noise, snr):
    
    [nz,ncmp] = input_data.shape

    Rho = np.zeros(input_data.shape)
    Vp = np.zeros(input_data.shape)
    Vs = np.zeros(input_data.shape)
    
    Rho = np.where(input_data == 0, 2.3, 2.5)
    Vp = np.where(input_data == 0, 2.8, 3.2)
    Vs = np.where(input_data == 0, 1.6, 1.8)
    
    dz = 0.001 
    
    Depth = np.arange(nz) * 0.001
    
    # travel time
    dt = 0.001
    t0 = 0.0
    temp = np.cumsum(np.diff(Depth,axis=0)/Vp[1:,int(np.round(ncmp/2))-1])
    TimeLog = np.append(t0, t0+2*temp)
    Time = np.arange(TimeLog[0],TimeLog[-1],dt)
    
    # number of samples (seismic properties)
    nt = len(Time) - 1
    
    #%%
    #% Wavelet
    # wavelet 
    freq = 45 #45
    ntw = 64
    wavelet, tw = RickerWavelet(freq, dt, ntw)
    
    
    # reflection angles 
    ntheta = n_theta
    max_theta = 45
    theta = np.linspace(0,max_theta,ntheta)
    
    
    Seis = np.zeros((nt,ntheta * ncmp))
    
    Noise = noise
    SNR = snr
    
    
    Vp_t = np.zeros((nt+1,ncmp))
    Vs_t = np.zeros((nt+1,ncmp))
    Rho_t = np.zeros((nt+1,ncmp))
    
    for CMP in range(ncmp):
        
        # time-interpolated elastic log
        Vp_t[:,CMP] = np.interp(Time, TimeLog, np.squeeze(Vp[:,CMP]))
        Vs_t[:,CMP] = np.interp(Time, TimeLog, np.squeeze(Vs[:,CMP]))
        Rho_t[:,CMP] = np.interp(Time, TimeLog, np.squeeze(Rho[:,CMP]))
        
        #% Synthetic seismic data
        tmp_seis, _ = SeismicModel(Vp_t[:,CMP], Vs_t[:,CMP], Rho_t[:,CMP], Time, theta, wavelet)
        
        if Noise == 1:
            for i in range(tmp_seis.shape[1]):
                SignalRMS = np.sqrt(np.mean(np.power(tmp_seis,2)))
                # np.random.seed(0)
                NoiseTD = np.random.randn(len(tmp_seis[:,i]),1)
                NoiseRMS = np.sqrt(np.mean(np.power(NoiseTD,2)))
                New = np.reshape(tmp_seis[:,i],[-1,1]) + (SignalRMS/NoiseRMS) * np.power(10,-SNR/20) * NoiseTD
                tmp_seis[:,i] = New[:,0]
        
        Seis[:,CMP * ntheta:(CMP+1)*ntheta] = np.reshape(tmp_seis,[ntheta,nt]).T
        
    if nt < nt_limit:
        Seis_sim = np.zeros((nt_limit,ntheta*ncmp))
        Time_sim = np.zeros((nt_limit,))
        Seis_sim = np.append(Seis,np.tile(Seis[-1,:],(nt_limit-nt,1)),0)
        Time_sim = np.append(Time,(np.arange(nt_limit-nt)+1)*dt+Time[-1],0)
    else:
        Seis_sim = Seis
        Time_sim = Time
        
    return Seis_sim[:nt_limit,:], Time_sim[:nt_limit] 
