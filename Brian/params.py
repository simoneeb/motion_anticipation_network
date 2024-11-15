



sig_c = 0.05      # bipolar receptive fied size ~ 1/5 of the actual size [mm]
tau1 = 0.04       # excitation time constant of the remporal filter [s]
tau2 = 0.0876     # rebound time constant [s]
bifw = 0.         # relative weight of rebound/excitation [1]
# scale_mV = 1.     # UPDATED 
scale_mV = 20.     # UPDATED 

tauA = 0.15   # time constant of amacrine cells [s]
tauB = 0.08   # time constant of bipolar cells [s]
tauG = 0.01   # time contant of ganglion cells [s]

wAB = 10.     # synaptic weight from bipolar to amacrine [Hz]
wBA = -10.    # synaptic weight from amacrine to bipolar [Hz]
wGB = 4.0     # synaptic weight from bipolar to gangion  [Hz].   #UPDATED
wGA = 0.      # synaptic weight from amacrine  to gangion  [Hz]

slope = 5  # slope for ganglion cell recritifation [Hz/mV].   #UPDATED
threshold = 0 # threshold for ganglion cell recritifation [Hz/mV]

sig_pool = 0.065   # sigma for gaussian pooling in ganlion gells [mm]
spacing = 0.005    # spacing of cells on the lattice [mm]

N = 256          # number of neurons in each layer [1]
idx = int(N/2)

b =  0.160          # half bar width [mm]
dt = 0.001          # integration time step [s]



# class params(object):

#     def __init__(self):
#         self.sig_c = 0.05      # bipolar receptive fied size ~ 1/5 of the actual size [mm]
#         self.tau1 = 0.04       # excitation time constant of the remporal filter [s]
#         self.tau2 = 0.0876     # rebound time constant [s]
#         bifw = 0.         # relative weight of rebound/excitation [1]
#         # scale_mV = 1.     # UPDATED 
#         scale_mV = 20.     # UPDATED 

#         tauA = 0.15   # time constant of amacrine cells [s]
#         tauB = 0.08   # time constant of bipolar cells [s]
#         tauG = 0.01   # time contant of ganglion cells [s]

#         wAB = 10.     # synaptic weight from bipolar to amacrine [Hz]
#         wBA = -10.    # synaptic weight from amacrine to bipolar [Hz]
#         wGB = 4.0     # synaptic weight from bipolar to gangion  [Hz].   #UPDATED
#         wGA = 0.      # synaptic weight from amacrine  to gangion  [Hz]

#         slope = 5  # slope for ganglion cell recritifation [Hz/mV].   #UPDATED
#         threshold = 0 # threshold for ganglion cell recritifation [Hz/mV]

#         sig_pool = 0.065   # sigma for gaussian pooling in ganlion gells [mm]
#         spacing = 0.005    # spacing of cells on the lattice [mm]

#         N = 256          # number of neurons in each layer [1]
#         idx = int(N/2)

#         b =  0.160          # half bar width [mm]
#         dt = 0.001          # integration time step [s]