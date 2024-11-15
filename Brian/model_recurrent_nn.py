from brian2 import *



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


# equation that defines GC integration
eqs_gc = """ dv/dt = -(1/tau) * v + vsyn : volt

tau : second
vsyn : volt/second
"""



def simulate(input_matrix,name = None, dir = None):

    start_scope()                # start the stimulation environment
    defaultclock.dt = dt*second  # set integration time step

    N,tps = input_matrix.shape   # extract some parameter
    dur = tps*dt                 # duration of the siulation [s]


    # create neurons
    bc = NeuronGroup(N, eqs_bc,  method = 'euler')
    ac = NeuronGroup(N, eqs_ac,  method = 'euler')
    gc = NeuronGroup(N,eqs_gc,method = 'euler')

    # create synapses
    synab = Synapses(bc,ac,eqs_syn)       # create synapse form BC to AC
    synba = Synapses(ac,bc,eqs_syn)       # create synapse form AC to BC


    # create connectivity matrix
    C = np.zeros((N,N))
    ii = np.arange(0,N-1,1).astype(int)  # list of source cells for rightward connections
    ji = ii + 1                          # target cells of rightward connections


    C[ii,ji] = 1

    ii = np.arange(1,N,1).astype(int)    # list of source cells for leftward connections
    ji = ii - 1                         # target cells of leftward connections

    C[ii,ji] = 1

    sources, targets = C.nonzero()      # get cell indices for sources and targets
    synab.connect(i=sources, j=targets) # connect sources and targets BC to AC
    synba.connect(i=sources, j=targets) # connect sources and targets AC to BC


    syngb = Synapses(bc,gc,eqs_syn)     # create synapses BC to GC
    syngb.connect()                     # connect all to all


    # set parameter
    bc.tau = tauB * second
    synba.wsyn = wBA * Hz


    ac.tau = tauA * second
    synab.wsyn = wAB * Hz

    gc.tau = tauG * second

    # assign gaussian weighting to synapses from BC to GC
    for n in range(N):
        syngb.wsyn[n,:] = wGB* DOG(np.arange(N),n,sig_pool/spacing)*Hz


    # info: there are no forward connections from AC to GC here, they could be added similar to syngb

    # transform input as timed array for integration
    inp = TimedArray(input_matrix.T*scale_mV*mV/second,dt=defaultclock.dt)

    # set up monitors to record cell voltages
    monbc = StateMonitor(bc, ('v'), record = True)
    monac = StateMonitor(ac, ('v'), record = True)
    mongc = StateMonitor(gc, ('v'), record = True)
    run(dur*second)


    # todo: pass GC voltage thorugh nonlinearity for firing rate transformation

    # todo: save output automatically ?
    if name is not False:
        np.save(f'{dir}/BC_grid_{name}.npy',monbc.v/mV)
        np.save(f'{dir}/GC_grid_{name}.npy',mongc.v/mV)


    return [monbc.v[int(N/2)]/mV, monac.v[int(N/2)]/mV, mongc.v[int(N/2)]/mV]   # info: here you can change the output to return the voltage of all cells


# function to set a different speed for the moving bar
def choose_speed(speed):

    # recalculate some params
    dur = L/speed
    tps = int(dur/dt)
    time = np.arange(0,tps)*dt

    # make stimulus
    barstim = np.zeros((N,tps))

    for n in range(N):
        for ti in range(tps):
            x = n*spacing
            barstim[n,ti] = bar(ti*dt,x,b,speed)


    # simulate OPL
    spat,temp = OPL(barstim)

    #simulate network
    name = f'bar_{speed}'
    res = simulate(temp, name = f'{netdir}/stim_{name}.npy')

    return [time,barstim[idx,:],spat[idx,:],temp[idx,:],res]



# info: you could write a similar function to set the number of cells as choose_speed


