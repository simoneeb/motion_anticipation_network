import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from utils import gaussian, DOG, GainF_B,GainF_G, bar
from nonlinearities import N
from scipy.signal import convolve
import pandas as pd 
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit


def DOG(x, mu, sig_c,sig_s,w):

    kern =  np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig_c, 2.))) - w*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig_s, 2.)))
    kern = kern / kern.max()
    return kern



def spatial_kernel(stimulus,sig_c,sig_s,w=0):

    out_c = gaussian_filter1d(stimulus, sig_c, axis = 0)
    out_s = gaussian_filter1d(stimulus, sig_s,axis = 0)
    out_total = out_c - w*out_s
        
    return out_total

def biphasic_alpha(t,tauOPL,tauOPL2,SF):
    
    kern =  (t/tauOPL**2) * np.exp(-t/tauOPL) * np.heaviside(t,1) -  SF* (t/tauOPL2**2) * np.exp(-t/tauOPL2) * np.heaviside(t,1) 
    # kern = (t/tauOPL) * np.exp(-t/tauOPL) * np.heaviside(t,1) -  SF* (t/tauOPL2) * np.exp(-t/tauOPL2) * np.heaviside(t,1) 
    # kern = kern/(np.sum(kern)*0.001)
    #calculate integral
    return  kern

    
class stim_moving_object_for_2D_net(object):

    def __init__(self,
                 params,
                 filepath = None):

        self.filepath = filepath
        self.params = params

        self.nb_cells = params['nb_cells']
        self.nb_GC_cells = params['nb_GC_cells']
        self.spacing = params['spacing']
        self.tauOPL = params['tauOPL']
        self.tauOPL2 = params['tauOPL2']
        self.SF = params['SF']
        self.input_scale = params['input_scale']
        #self.rf_overlap = params['rf_size']
        self.rf_BC = params['rf_BC']
        self.rf_BC_surround = params['rf_BC_s']

        self.sig_c = self.rf_BC/6/self.spacing
        self.sig_s = self.rf_BC_surround/6/self.spacing
        self.speed = params['speed']
        self.distance = params['distance']
        self.dt = params['dt']
        self.stimulus_polarity = params['stimulus_polarity']
        self.bar_width = params['bar_width']

        self.stop_pos = params['stop_pos']
        self.start_cell = params['start_cell']
        self.start_pos = self.start_cell*self.spacing
        self.start_tp = params['start_tp']
        self.occluder_width = params['occluder_width']


        if self.stop_pos is not None: 
            self.stop_t =(self.stop_pos/self.speed)
            self.stop_idf = int(self.stop_t/self.dt)

        self.duration = params['duration']
        self.time = np.arange(0,self.duration,self.dt)
        self.tps = params['tps']
        self.tauB = params['tauB']
        self.tauActB = params['tauActB']
        self.tauActG = params['tauActG']
        self.w = params['w_BC']
        
        self.pos_rf_mid = np.arange(0,self.nb_cells,1)*self.spacing
      
        self.tps_rf_mid = self.pos_rf_mid/self.speed #s
        self.time_to_cross_rf = self.rf_BC/self.speed
        self.time_to_cross_rf_surround = self.rf_BC_surround/self.speed
        self.rf_sizes = np.zeros(self.nb_cells)  #mm, no overlap
        self.rf_sizes_surround = np.zeros(self.nb_cells)  #mm, no overlap
        for i in range(self.nb_cells):
            self.rf_sizes[i] = self.rf_BC# + np.random.normal(0,0.1)
            self.rf_sizes_surround[i] = self.rf_BC_surround# + np.random.normal(0,0.1)

        
        self.std = self.time_to_cross_rf/6
        self.std_surround = self.time_to_cross_rf_surround/6
        self.roh = (self.tauB*self.speed)/self.rf_BC
        self.kernel_template = None
    


    def alpha_kernel(self): 

        self.ftime = np.arange(0,1,self.dt)
        self.temporal_kernel = (self.ftime/self.tauOPL**2) * np.exp(-self.ftime/self.tauOPL) * np.heaviside(self.ftime,1) 

        return self.temporal_kernel



    def filter_biphasic_norm(self):

        self.ftime = np.arange(0,1,self.dt)
        self.temporal_kernel = biphasic_alpha(self.ftime,self.tauOPL,self.tauOPL2,self.SF) 
        #self.temporal_kernel =  (self.ftime/self.tauOPL**2) * np.exp(-self.ftime/self.tauOPL) * np.heaviside(self.ftime,1) -  self.SF* (self.ftime/self.tauOPL2**2) * np.exp(-self.ftime/self.tauOPL2) * np.heaviside(self.ftime,1) 

        return self.temporal_kernel
    

    def load_filter(self):

        fp = '/Users/simone/Documents/Simulations/chen_2013/chen_2013_fast_OFF_filter.csv'
        chen_data = pd.read_csv(fp)
        cols = chen_data.columns
        #chen_data.info()

        x = gaussian_filter(chen_data[cols[0]].dropna().values[1:].astype(float),sigma =5)
        y = chen_data[cols[1]].dropna().values[1:].astype(float)
        # plt.plot(x,y)
        # plt.show()
        # interpolate
        f = interp1d(x,y, fill_value='extrapolate')

        # resample
        self.ttime = np.arange(np.round(x.min()),np.round(x.max()),1)
        self.kernel_template =-1*f(self.ttime)[130:]
        self.ttime = self.ttime[130:]*-0.001
        #fit biphasic profile 
        popt,_ = curve_fit(biphasic_alpha,self.ttime, self.kernel_template, p0 = [0.05508169, 0.05730902, 0.99907681])
        #print(popt)

        # create kernel
        self.ftime = np.arange(0,1,self.dt)
        self.temporal_kernel =biphasic_alpha(self.ftime,*popt)

        return self.temporal_kernel




    def bar_smooth(self):

        # make stimulus
        self.barstim = np.zeros((self.nb_cells,self.tps))

        for c in range(self.nb_cells):
            #print(c*spacing)
            for i in range(self.tps):
                self.barstim[c,i] = bar(i*self.dt,c*self.spacing, v = self.speed,b = self.bar_width) 

        return self.barstim
    
    
    def bar_onset(self):

        self.barstim = np.zeros((self.nb_cells,self.tps))

        for c in range(self.nb_cells):
            #print(c*spacing)
                for i in range(self.tps):
                    if i*self.dt <= self.start_tp:
                        if c*self.spacing >= self.start_pos - self.bar_width and c*self.spacing <= self.start_pos + self.bar_width:
                            self.barstim[c,i] = 1
                    else:
                        self.barstim[c,i] = bar(i*self.dt - self.start_tp,c*self.spacing - self.start_pos, v = self.speed,b = self.bar_width) 

        return self.barstim


    def bar_reversing(self):

        self.barstim = np.zeros((self.nb_cells,self.tps))

        self.tr = self.start_pos/self.speed

        for c in range(self.nb_cells):
            #print(c*spacing)
            if c*self.spacing <= self.start_pos + self.bar_width/2:
                for i in range(self.tps):
                    if i*self.dt <= self.tr:
                        self.barstim[c,i] = bar(i*self.dt,c*self.spacing, v = self.speed,b = self.bar_width) 
                    else:
                        self.barstim[c,i] = bar(2*self.tr - i*self.dt,c*self.spacing, v = self.speed,b = self.bar_width) 


        return self.barstim
    
    #make impulse stimulus
    def impulse_stimulus(self,length = 1.,impulse_timepoint = 0, amplitude = 1):


        self.tps = int(length/self.dt)
        self.barstim = np.zeros((self.nb_cells,self.tps))

        impulse_idx = int(impulse_timepoint/self.dt)
        self.barstim[:,impulse_idx] = amplitude

        #timeline = np.arange(0,length,self.dt)

        return self.barstim




    def bar_interrupted(self):

        # make stimulus
        self.barstim = np.zeros((self.nb_cells,self.tps))

        for c in range(self.nb_cells):
            #print(c*spacing) 
            if c*self.spacing <= self.start_pos - self.occluder_width or c*self.spacing >= self.start_pos + self.occluder_width:
                for i in range(self.tps):
                    self.barstim[c,i] = bar(i*self.dt,c*self.spacing, v = self.speed,b = self.bar_width) 
    
        return self.barstim


    def OPL(self):

        #temporal = temporal_kernel(ftime, self.tauOPL)

        self.outs = spatial_kernel(self.barstim,self.sig_c,self.sig_s,self.w)
        self.outst = np.zeros((self.nb_cells,self.tps))
        for c in range(self.nb_cells):

            self.outst[c,:] = convolve(self.outs[c,:],self.temporal_kernel, mode = 'full')[:-len(self.temporal_kernel)+1]

        self.outst = self.outst*self.input_scale 


        return self.outs,self.outst
    

    def F(self):

        self.F_array = np.zeros((self.nb_cells,self.tps))
  
        for c in range(self.nb_cells):
            outst =  np.zeros(self.tps+1)
            outst[:-1] = self.outst[c,:].copy()
            outst_prime = [(outst[i]-outst[i-1])/self.dt for i in range(0,self.tps)]
            self.F_array[c,:] =self.outst[c,:]/self.tauB + outst_prime

        return self.F_array
    

    

    def plot_kernels(self,tosave = True):

        fig, ax = plt.subplots(1,2, figsize =  (16,12))
        if self.kernel_template is not None :
            ax[0].scatter(self.ttime,self.kernel_template, color = 'k', linewidth = 5)
        ax[0].plot(self.ftime,self.temporal_kernel, color = 'k', linewidth = 5)
        spatial = DOG(np.arange(self.nb_cells)*self.spacing,int(self.nb_cells/2)*self.spacing, self.sig_c*self.spacing,self.sig_s*self.spacing,self.w)
        ax[1].plot(np.arange(self.nb_cells)*self.spacing,spatial, color = 'k', linewidth = 5)
        ax[1].axhline(np.mean(spatial), label = 'mean', color = 'grey', linestyle = ':', linewidth = 5)
        ax[1].axhline(np.median(spatial), label = 'median', color = 'grey', linewidth = 5)
        ax[0].set_xlabel ('time [s]')
        ax[1].set_xlabel ('space [mm]')

        ax[0].set_title(f'{np.sum(self.temporal_kernel)*self.dt}')
        ax[1].set_title(f'{np.sum(spatial)*self.spacing}')
        ax[1].legend()
        #plt.show()
        if tosave == True:
            fig.savefig(f'{self.filepath}/plots/kernels.png')


    def plot_stim(self, tosave = True):

        # print(f'stimulus speed {self.speed} : mm/s')
        # print(f'rf size {self.rf_size} : mm ')
        # print(f'roh {self.roh} : 1 ')

        fig,ax = plt.subplots(4,1, sharex = True, figsize =  (16,12))
        ploti =int(self.nb_cells/2)
        item = ax[0].plot(self.barstim[int(self.nb_cells/2),:]/self.barstim[ploti,:].max(), label = 'bar')
        ax[0].plot(self.outs[int(self.nb_cells/2),:]/self.outs[int(self.nb_cells/2),:].max(), label = 'spatial')
        ax[0].plot(self.outst[int(self.nb_cells/2),:]/self.outst[int(self.nb_cells/2),:].max(), label = 'spatiotemporal')
        ax[0].axvline(self.spacing*100/self.speed,linestyle = ':', color = item[0].get_color())
        ax[0].set_xlabel('timesteps')
        ax[0].set_ylabel('inputs')
        ax[0].set_title('Example for one cell')

        ax[1].matshow(self.barstim, aspect = 'auto')
        ax[1].set_ylabel('cells')
        ax[1].set_title('Stimulus')

        ax[2].matshow(self.outs, aspect = 'auto')
        ax[2].set_ylabel('cells')
        ax[2].set_title('Spatial Convolution')

        ax[3].matshow(self.outst, aspect = 'auto')
        ax[3].set_ylabel('cells')
        ax[3].set_title('Spatiotemporal Convolution')

        ax[0].legend()
        if tosave == True:
            fig.savefig(f'{self.filepath}/plots/stimulus.png')



    # def smooth_motion(self):

    #     self.inp = np.zeros((self.nb_cells,self.tps))

    #     for i in range(self.nb_cells):
    #         tst = DOG(self.time,self.tps_rf_mid[i],self.std[i],self.std_surround[i],self.w)
    #         self.inp[i,:] = tst
                        
    #     return self.inp
    

#     def reversing_motion():


   

    def plot_stim_simple(self):

        # print(f'stimulus speed {self.speed} : mm/s')
        # print(f'rf size {self.rf_size} : mm ')
        # print(f'roh {self.roh} : 1 ')

        fig,ax = plt.subplots(2,1, sharex = True, figsize =  (16,12))

        for i in range(self.nb_cells):
            item = ax[0].plot(self.inp[i,:])
            ax[0].axvline(self.tps_rf_mid[i]/self.dt,linestyle = ':', color = item[0].get_color())
        ax[0].set_xlabel('timesteps')
        ax[0].set_ylabel('inputs')
        ax[0].set_title('Stimulus')
        ax[1].matshow(self.inp, aspect = 'auto')
        ax[1].set_ylabel('cells')
        ax[1].set_title('Stimulus')

        fig.savefig(f'{self.filepath}/plots/stimulus.png')
        



    def add_params(self):

     
        self.params['std'] = self.std 
        #self.params['p'] = self.p 
        self.params['roh'] = self.roh
        self.params['pos_rf_mid'] = self.pos_rf_mid
        self.params['tps_rf_mid'] = self.tps_rf_mid
        self.params['tps'] = self.tps
        if self.stop_pos is not None: 

            self.params['stop_t'] = self.stop_t
            self.params['stop_idx'] = self.stop_idf

        return self.params




# stimulation details


# appendix = f'speed_{speed}mms_newcode'
# print(f'rf size {rf_size} mm, no overlap')
# print(f'time to cross rf {time_to_cross_rf} s')
# print(f'roh  = {roh}')






