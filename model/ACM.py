import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from utils import gaussian, DOG, GainF_B,GainF_G
from nonlinearities import N
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit


    

class ACM(object):

    def __init__(self,
                 params,
                 inp,
                 filepath = None):

        self.filepath = filepath
        self.params = params
        self.VB = inp
        self.nb_cells = params['nb_cells']
        self.nb_GC_cells = params['nb_GC_cells']
        self.spacing = params['spacing']
        #self.rf_overlap = params['rf_size']
        self.rf_GC = params['rf_GC']
        self.rf_GC_surround = params['rf_GC_s']

        self.sig_c =params['std_GC']
        self.sig_s =params['std_GC_s']
        self.weight =params['wGB'] 
        
        self.dt = params['dt']
        self.duration = params['duration']
        self.speed = params['speed']
        self.time = np.arange(0,self.duration,self.dt)
        self.tps = params['tps']

        self.tauActB = params['tauActB']
        self.hB = params['hB']
        self.hG = params['hG']
        self.tauActG = params['tauActG']
        self.w = params['w_GC']
        
        self.pos_rf_mid = np.arange(0,self.nb_cells,1)*self.spacing
      
        self.tps_rf_mid = self.pos_rf_mid/self.speed #s
        self.time_to_cross_rf = self.rf_GC/self.speed
        self.time_to_cross_rf_surround = self.rf_GC_surround/self.speed




    def make_activity_kernelB(self): 

        self.acttime = np.arange(0,1,self.dt)
        self.activity_kernelB = np.exp(-self.acttime/self.tauActB)


        return self.activity_kernelB 
    
        
    def make_activity_kernelG(self): 

        self.acttime = np.arange(0,1,self.dt)
        self.activity_kernelG = np.exp(-self.acttime/self.tauActB)


        return self.activity_kernelG


    def make_GCL_weight_matrix_pooling(self):
        
        # make weight matrix
        self.W_pooling = np.zeros((self.nb_GC_cells,self.nb_cells))
    
        # gaussian weighting 
        for i in range(self.nb_GC_cells):

            self.W_pooling[i,:] = self.weight * DOG(self.pos_rf_mid,self.spacing*i,self.sig_c,self.sig_s,self.w)

        return self.W_pooling
   

    def BCL(self):

        self.NB = np.zeros((self.nb_cells,self.tps))
        self.AB = np.zeros((self.nb_cells,self.tps))
        self.GB = np.zeros((self.nb_cells,self.tps))
        self.RB = np.zeros((self.nb_cells,self.tps))

        for c in range(self.nb_cells):

            self.NB[c,:] = [N(v,self.params,'BC')for v in self.VB[c,:]]
            self.AB[c,:] = self.hB*convolve(self.NB[c,:],self.activity_kernelB, mode = 'full')[:-len(self.activity_kernelB)+1]
            self.GB[c,:] = [GainF_B(a) for a in self.AB[c,:]]
            self.RB[c,:] = self.NB[c,:]*self.GB[c,:]



    def GCL(self):

        self.VG = np.zeros((self.nb_GC_cells,self.tps))
        self.NG = np.zeros((self.nb_GC_cells,self.tps))
        self.AG = np.zeros((self.nb_cells,self.tps))
        self.GG = np.zeros((self.nb_cells,self.tps))
        self.RG = np.zeros((self.nb_cells,self.tps))

        for c in range(self.nb_GC_cells):
            self.VG[c,:] = self.W_pooling[c,:].T@self.RB[:,:]

            self.NG[c,:] = [N(v,self.params,'GC')for v in self.VG[c,:]]
            self.AG[c,:] = self.hG *convolve(self.NG[c,:],self.activity_kernelG, mode = 'full')[:-len(self.activity_kernelG)+1]
            self.GG[c,:] = [GainF_G(a) for a in self.AG[c,:]]
            self.RG[c,:] = self.NG[c,:]*self.GG[c,:]




    def collect_output(self):

        out = {
        'VB': self.VB,
        'AB' : self.AB,
        'NB' : self.NB,
        'GB' : self.GB,
        'RB' : self.RB,
        'VG' : self.VG,
        'AG': self.AG,
        'GG': self.GG,
        'NG': self.NG,
        'RG' : self.RG,
            }

        return out


    def calculate_anticipation(self):

        self.ant_time = np.zeros(self.nb_GC_cells)
        self.ant_space = np.zeros(self.nb_GC_cells)
        self.ant_time_drive = np.zeros(self.nb_GC_cells)
        self.ant_space_drive = np.zeros(self.nb_GC_cells)
        for i,GC in enumerate(self.RG):
            drive = self.VB
            self.ant_time_drive[i] = (np.argmax(drive)-np.argmax(GC))*self.dt
            self.ant_space_drive[i] = self.ant_time_drive[i] *self.speed
            self.ant_time[i] = self.tps_rf_mid[i]-np.argmax(GC)*self.dt
            self.ant_space[i] = self.ant_time[i] *self.speed


       #return[ant_time, ant_space,ant_time_drive,ant_space_drive]


    def add_params(self):

        self.params['pos_rf_mid'] = self.pos_rf_mid
        self.params['pos_rf_GC_mid'] = self.pos_rf_mid
        self.params['tps_rf_mid'] = self.tps_rf_mid
        self.params['tps_rf_GC_mid'] = self.tps_rf_mid
        self.params['time_to_cross_rf_GC'] = self.time_to_cross_rf
        self.params['ant_time'] = self.ant_time
        self.params['ant_space'] = self.ant_space
        self.params['ant_time_drive'] = self.ant_time_drive
        self.params['ant_space_drive'] = self.ant_space_drive
        #self.params['p'] = self.p 
        #self.params['roh'] = self.roh


        return self.params






