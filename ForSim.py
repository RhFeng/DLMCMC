import numpy as np
from scipy import signal
# from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter

def ForSim(input_data, k_cut, noise, snr):
    
    [nz,ncmp] = input_data.shape

    AI = np.zeros(input_data.shape)
    
    AI = np.where(input_data == 0, 6.0, 10.0)
    
    # nfilt = 3
    # cutofffr = k_cut
    # b, a = signal.butter(nfilt, cutofffr,btype='lowpass')
    
    AI_sm = np.zeros(AI.shape)
    
    # for i in range(ncmp):
    
        # Vp_sm[:,i] = signal.filtfilt(b, a, np.squeeze(Vp[:,i]))
        
        # Vp_sm[:,i] = gaussian_filter1d(np.squeeze(Vp[:,i]),3)
        
    AI_sm = gaussian_filter(np.squeeze(AI),7)  #7
    
    #%%   
    Noise = noise
    SNR = snr
      
    AI_noise = AI_sm
    
    if Noise == 1:
    
        for CMP in range(ncmp):
            
            tmp_AI = np.reshape(AI_noise[:,CMP],[-1,1])
            
            for i in range(tmp_AI.shape[1]):
                SignalRMS = np.sqrt(np.mean(np.power(tmp_AI,2)))
                # np.random.seed(0)
                NoiseTD = np.random.randn(len(tmp_AI[:,i]),1)
                NoiseRMS = np.sqrt(np.mean(np.power(NoiseTD,2)))
                New = np.reshape(tmp_AI[:,i],[-1,1]) + (SignalRMS/NoiseRMS) * np.power(10,-SNR/20) * NoiseTD
                # New = np.reshape(tmp_AI[:,i],[-1,1]) + np.sqrt(np.square(np.mean(tmp_AI))/SNR) * NoiseTD
                tmp_AI[:,i] = New[:,0]
            
            AI_noise[:,CMP] = tmp_AI.ravel()
        
        AI_noise = AI_sm + np.sqrt(np.square(np.mean(AI_sm))/SNR) * np.random.randn(nz,ncmp)  
        
    return AI_noise
