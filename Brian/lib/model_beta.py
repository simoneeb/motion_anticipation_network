from brian2 import *
import os
import numpy as np
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter1d
import pickle


# function that defines the temporal kernel
def biphasic_alpha(t,tau1,tau2,bifw):

    kern =  (t/(tau1**2)) * np.exp(-t/tau1) * np.heaviside(t,1) -  bifw* (t/(tau2**2)) * np.exp(-t/tau2) * np.heaviside(t,1)
    return  kern


# function for gaussialn pooling
def DOG(x, mu, sig_c):

    kern =  np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig_c, 2.)))
    kern = kern / kern.max() # normalize
    return kern


class Model(object):

    def __init__(self, netname):

        self.netname = netname
        self.dir = f'../output/{self.netname}'

        print(self.dir)

        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)


    def set_params(self,params):
        # set each parameter as instance of the model
        self.sig_c =  params['sig_c']
        self.tau1 =  params['tau1']
        self.tau2 =  params['tau2']
        self.bifw =  params['bifw']

        self.scale_mV =  params['scale_mV']
        self.tauA =  params['tauA']
        self.tauB =  params['tauB']
        self.tauG =  params['tauG']

        self.wAB =  params['wAB']
        self.wBA =  params['wBA'] 
        self.wGB =  params['wGB']
        self.wGA =  params['wGA']

        self.slope =  params['slope']
        self.threshold =  params['threshold']
        self.sig_pool =  params['sig_pool'] 


        self.spacing =  params['spacing']
        self.dt =  params['dt']
        self.N =  params['N']



        # save params
        with open(f'{self.dir}/params.pkl', 'wb') as f:
            pickle.dump(params, f)
                


        # idx = int(N/2)
        # b =  0.160         # half bar width [mm]
        # speed =  0.8       # speed [mm/s]  
        





    def set_connectivity(self, C):

        self.C = C
     

    def set_stimulus(self,input_matrix,stimname):

        self.N,self.tps = input_matrix.shape        # extract some parameter
        self.dur = self.tps*self.dt                 # duration of the siulation [s]
        self.stim  = input_matrix

        self.stimname = stimname

        self.stimdir = f'../output/{self.netname}/{self.stimname}'

        # print(self.stimdir)

        if not os.path.isdir(self.stimdir):
            os.mkdir(self.stimdir)




    def simulate(self, save = True):

        # spatiotemporal convolution
        self.spat = np.zeros((self.N,self.tps))
        self.temp = np.zeros((self.N,self.tps))
        # print(self.tps)

        self.spat = gaussian_filter1d(self.stim, self.sig_c/self.spacing, axis = 0)
        # self.spat = self.stim
        #spat = gaussian_filter1d(stim, sig_c/dt, axis = 1)
        # self.spat = self.spat/np.max(self.spat)


        ftime = np.arange(0,1,self.dt)
        self.temporal_kernel = biphasic_alpha(ftime,self.tau1,self.tau2,self.bifw)


        for n in range(self.N):

            # self.spat[n,:] = self.spat[n,:]/np.max(self.spat[n,:])
            # apply temporal filter
            self.temp[n,:] = convolve(self.spat[n,:],self.temporal_kernel, mode = 'full')[:-len(self.temporal_kernel)+1]*self.dt*self.scale_mV



        # transpofm OPL voltage response into input current (divided by capacitance) for bipolars
        self.F_array = np.zeros((self.N,self.tps))

        for c in range(self.N):
            outst =  np.zeros(self.tps)
            outst[:] = self.temp[c,:].copy()
            outst_prime = [(outst[i]-outst[i-1])/self.dt for i in range(0,self.tps)]
            self.F_array[c,:] = outst[:]/self.tauB + outst_prime


        # define equations for bipolar and amacrine cell

        #equation that defines the BC voltage, with input
        eqs_bc = """ dv/dt = -(1/tau) * v + vsyn + inp(t,i): volt

        tau : second
        vsyn : volt/second
        """


        #equation that defines the AC voltage
        eqs_ac = """ dv/dt = -(1/tau) * v + vsyn : volt

        tau : second
        vsyn : volt/second
        """


        # equation for synaptic coupling
        eqs_syn = """vsyn_post = wsyn*v_pre : volt/second (summed)

        wsyn: Hz

        """
        
        
        # equation for synaptic coupling
        eqs_synb = """vsynb_post = wsyn*v_pre : volt/second (summed)

        wsyn: Hz

        """
        
        
        # equation for synaptic coupling
        eqs_syna = """vsyna_post = wsyn*v_pre : volt/second (summed)

        wsyn: Hz

        """


        # equation that defines GC integration
        eqs_gc = """ dv/dt = -(1/tau) * v + vsynb + vsyna : volt

        tau : second
        vsyna : volt/second
        vsynb : volt/second
        """


        start_scope()                # start the stimulation environment
        defaultclock.dt = self.dt*second  # set integration time step


        # create neurons
        bc = NeuronGroup(self.N, eqs_bc,  method = 'rk4')
        ac = NeuronGroup(self.N, eqs_ac,  method = 'rk4')
        gc = NeuronGroup(self.N,eqs_gc,method = 'rk4')


        # create synapses
        synab = Synapses(bc,ac,eqs_syn)       # create synapse form BC to AC
        synba = Synapses(ac,bc,eqs_syn)       # create synapse form AC to BC


    
        sources, targets = self.C.nonzero()      # get cell indices for sources and targets
        synab.connect(i=sources, j=targets)      # connect sources and targets BC to AC
        synba.connect(i=sources, j=targets)      # connect sources and targets AC to BC


        syngb = Synapses(bc,gc,eqs_synb)     # create synapses BC to GC
        syngb.connect()                     # connect all to all


        synga = Synapses(ac,gc,eqs_syna)     # create synapses BC to GC
        synga.connect()                     # connect all to all


        # set parameter
        bc.tau = self.tauB * second
        synba.wsyn = self.wBA * Hz


        ac.tau = self.tauA * second
        synab.wsyn = self.wAB * Hz

        gc.tau = self.tauG * second

        # assign gaussian weighting to synapses from BC to GC
        for n in range(self.N):
            syngb.wsyn[n,:] =self.wGB * DOG(np.arange(self.N),n,self.sig_pool/self.spacing)*Hz
            synga.wsyn[n,:] =self.wGA * DOG(np.arange(self.N),n,self.sig_pool/self.spacing)*Hz


        # transform input as timed array for integration
        inp = TimedArray(self.F_array.T*self.scale_mV*mV/second,dt=defaultclock.dt)

        # set up monitors to record cell voltages
        monbc = StateMonitor(bc, ('v'), record = True)
        monac = StateMonitor(ac, ('v'), record = True)
        mongc = StateMonitor(gc, ('v'), record = True)

        run(self.dur*second) # run the simulation 

        # todo: pass GC voltage thorugh nonlinearity for firing rate transformation


        # save output 
        if save == True:

            np.save(f'{self.stimdir}/BC_grid_{self.stimname}.npy',monbc.v/mV)
            np.save(f'{self.stimdir}/AC_grid_{self.stimname}.npy',monac.v/mV)
            np.save(f'{self.stimdir}/GC_grid_{self.stimname}.npy',mongc.v/mV)
            np.save(f'{self.stimdir}/stim_grid_{self.stimname}.npy',self.stim)
            np.save(f'{self.stimdir}/stc_grid_{self.stimname}.npy',self.temp)

        else: 
            BCgrid = monbc.v/mV
            GCgrid = mongc.v/mV
            return BCgrid[self.N//2,:],GCgrid[self.N//2,:]










