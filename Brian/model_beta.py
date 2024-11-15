from brian2 import *
import os
import numpy as np
from scipy.signal import convolve


class Model(object):

    def __init__(self,name,dir):

        self.name = name
        self.dir = dir

        if not os.path.isdir(dir):
            os.mkdir(dir)

        
    def set_params(self,params):

        self.params = [params]


    # def set_connectivity(C):

    #     self.C = C
     

    # def set_stim(self,input_matrix):

    #     self.N,self.tps = input_matrix.shape   # extract some parameter
    #     self.dur = self.tps*self.dt                 # duration of the siulation [s]
    #     self.stim  = input_matrix


    # def OPL():

    #     self.spat = np.zeros((self.N,self.tps))
    #     self.temp = np.zeros((self.N,self.tps))

    #     spat = gaussian_filter1d(self.stim, self.sig_c/self.spacing, axis = 0)
    #     #spat = gaussian_filter1d(stim, sig_c/dt, axis = 1)
    #     self.spat = spat/np.max(spat)

    #     for n in range(N):
    #         spat[n,:] = spat[n,:]/np.max(spat[n,:])
    #         # apply temporal filter
    #         self.temp[n,:] = convolve(spat[n,:],temporal_kernel, mode = 'full')[:-len(temporal_kernel)+1]*dt*scale_mV


    # def F():

    # def IPL():

    #     # define equations for bipolar and amacrine cell

    #     #equation that defines the BC voltage, with input
    #     eqs_bc = """ dv/dt = -(1/tau) * v + vsyn + inp(t,i): volt

    #     tau : second
    #     vsyn : volt/second
    #     """


    #     #equation that defines the AC voltage
    #     eqs_ac = """ dv/dt = -(1/tau) * v + vsyn : volt

    #     tau : second
    #     vsyn : volt/second
    #     """


    #     # equation for synaptic coupling
    #     eqs_syn = """vsyn_post = wsyn*v_pre : volt/second (summed)

    #     wsyn: Hz

    #     """


    #     # equation that defines GC integration
    #     eqs_gc = """ dv/dt = -(1/tau) * v + vsyn : volt

    #     tau : second
    #     vsyn : volt/second
    #     """


    #     start_scope()                # start the stimulation environment
    #     defaultclock.dt = dt*second  # set integration time step


    #     # create neurons
    #     bc = NeuronGroup(N, eqs_bc,  method = 'euler')
    #     ac = NeuronGroup(N, eqs_ac,  method = 'euler')
    #     gc = NeuronGroup(N,eqs_gc,method = 'euler')


    #     # create synapses
    #     synab = Synapses(bc,ac,eqs_syn)       # create synapse form BC to AC
    #     synba = Synapses(ac,bc,eqs_syn)       # create synapse form AC to BC


    
    #     sources, targets = self.C.nonzero()      # get cell indices for sources and targets
    #     synab.connect(i=sources, j=targets)      # connect sources and targets BC to AC
    #     synba.connect(i=sources, j=targets)      # connect sources and targets AC to BC


    #     syngb = Synapses(bc,gc,eqs_syn)     # create synapses BC to GC
    #     syngb.connect()                     # connect all to all


    #     # set parameter
    #     bc.tau = tauB * second
    #     synba.wsyn = wBA * Hz


    #     ac.tau = tauA * second
    #     synab.wsyn = wAB * Hz

    #     gc.tau = tauG * second

    #     # assign gaussian weighting to synapses from BC to GC
    #     for n in range(N):
    #         syngb.wsyn[n,:] = wGB* DOG(np.arange(N),n,sig_pool/spacing)*Hz


    #     # info: there are no forward connections from AC to GC here, they could be added similar to syngb

    #     # transform input as timed array for integration
    #     inp = TimedArray(input_matrix.T*scale_mV*mV/second,dt=defaultclock.dt)

    #     # set up monitors to record cell voltages
    #     monbc = StateMonitor(bc, ('v'), record = True)
    #     monac = StateMonitor(ac, ('v'), record = True)
    #     mongc = StateMonitor(gc, ('v'), record = True)
    #     run(dur*second)



    # def simulate():

    # def save():


