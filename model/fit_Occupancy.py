import pickle
import numpy as np
import cma
from stimuli import stim_moving_object_for_2D_net
from connectivity import connectivity
from system import system
from utils import GainF_B,GainF_G
from nonlinearities import N
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.optimize import curve_fit


# evaluate erorrs
def compute_mse(y,prediction):
        
    error = np.sum((prediction-y)**2)
    error /= len(y)
    error = np.sqrt(error)
    return error


def simulate(inp,params):
       # run stimulation 
    #params = stimulus_maker.add_params()

    # create weight matrices

    connecter = connectivity(params,
                            filepath = None)

    W_BB = connecter.weight_matrix_i_to_i(-1/params['tauB'],params['nb_cells'])
    W_BA = connecter.weight_matrix_i_to_nn(-1*params['wBA'],params['nb_cells'])

    W_AA = connecter.weight_matrix_i_to_i(-1/params['tauA'],params['nb_cells'])
    W_AB = connecter.weight_matrix_i_to_nn(params['wAB'],params['nb_cells'])


    W_GG = connecter.weight_matrix_i_to_i(-1/params['tauG'],params['nb_GC_cells'])
    W_outB = connecter.weight_matrix_pooling(params['wGB'])
    W_outA = connecter.weight_matrix_pooling(params['wGA'])


    W_ActB = connecter.weight_matrix_i_to_i(-1/params['tauActB'],params['nb_cells'])
    W_BtoActB = connecter.weight_matrix_i_to_i(params['hB'],params['nb_cells'])

    W_ActA = connecter.weight_matrix_i_to_i(-1/params['tauActA'],params['nb_cells'])
    W_AtoActA = connecter.weight_matrix_i_to_i(params['hA'],params['nb_cells'])

    W_ActG = connecter.weight_matrix_i_to_i(-1/params['tauActG'],params['nb_GC_cells'])
    W_GtoActG = connecter.weight_matrix_i_to_i(params['hG'],params['nb_GC_cells'])

    W_krecB = connecter.weight_matrix_i_to_i(params['krecB'],params['nb_cells'])
    W_krelB = connecter.weight_matrix_i_to_i(params['krelB']*params['betaB'],params['nb_cells'])

    W_krecA = connecter.weight_matrix_i_to_i(params['krecA'],params['nb_cells'])
    W_krelA = connecter.weight_matrix_i_to_i(params['krelA']*params['betaA'],params['nb_cells'])



    W_connectivity_B = (W_BB,W_BA) 
    W_connectivity_A = (W_AB,W_AA)
    sys = system(params, W_GG, W_ActG, W_GtoActG)

    sys.create_layer([*W_connectivity_B],
                    W_ActB,W_BtoActB,
                    W_krecB,W_krelB,
                    W_outB,
                    params['rectification_BC'],
                    inp)


    sys.create_layer([*W_connectivity_A],
                    W_ActA,W_AtoActA,
                    W_krecA,W_krelA,
                    W_outA,
                    params['rectification_AC'],
                    np.zeros(inp.shape))


    #sys.dummy()
    test,test2,test3 = sys.solve_IPL_GainControl_Plasticity(GainF_B,N)
    Layers = sys.Layers_IPL
    #res,A = sys.solve_IPL_GainControl(N)

    VGsys,AGsys,NGsys = sys.solve_GC(N)
    RGsys, GGsys = sys.rectify(N,GainF_G)
    PVA = sys.PVA()


    nb_cells = params['nb_cells']
    tps = params['tps']

    VB = np.zeros((nb_cells,tps))
    NB = np.zeros((nb_cells,tps))
    AB = np.zeros((nb_cells,tps))
    GB = np.zeros((nb_cells,tps))
    RB = np.zeros((nb_cells,tps))

    for c in range(nb_cells):

        VB[c,:] = Layers[0]['X'][c]
        #NB[c,:] = [N(v,params,'BC')for v in Layers[0]['X'][c]]
        NB[c,:] = Layers[0]['X_rect'][c]
        AB[c,:] =  Layers[0]['A'][c]
        GB[c,:] = Layers[0]['G'][c] #[GainF_B(a) for a in AB[c,:]]
        RB[c,:] = NB[c,:]*GB[c,:]
        
        
    VA = np.zeros((nb_cells,tps))
    NA = np.zeros((nb_cells,tps))
    AA = np.zeros((nb_cells,tps))
    GA = np.zeros((nb_cells,tps))
    RA = np.zeros((nb_cells,tps))

    for c in range(nb_cells):

        VA[c,:] = Layers[1]['X'][c]
        #NB[c,:] = [N(v,params,'BC')for v in Layers[0]['X'][c]]
        NA[c,:] = Layers[1]['X_rect'][c]
        AA[c,:] =  Layers[1]['A'][c]
        GA[c,:] = Layers[1]['G'][c] #[GainF_B(a) for a in AB[c,:]]
        RA[c,:] = NA[c,:]*GA[c,:]

    [ant_time,ant_space] = sys.calculate_anticipation()



    VG = np.zeros((nb_cells,tps))
    NG = np.zeros((nb_cells,tps))
    AG = np.zeros((nb_cells,tps))
    GG = np.zeros((nb_cells,tps))
    RG = np.zeros((nb_cells,tps))

    for c in range(nb_cells):

        VG[c,:] =VGsys[c]
        NG[c,:] = NGsys[c]#[N(v,params,'GC')for v in VG[c,:]]
        AG[c,:] =  AGsys[c]
        GG[c,:] = GGsys[c]#[GainF_G(a) for a in AG[c,:]]
        RG[c,:] = NG[c,:]*GG[c,:]

    return [VB,RB,VA,RA,RG]


def slopef(x,a,b):
    return a*x + b

#hyperparameter 
sigma0 = 1
popsize = 10
nb_repeats = 5



# plt.plot(res)
# plt.show()

# load initial paramset
s = 0.81
fpp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/Reciporcal/Reciporcal_fitted_F_plastic/w_GC/w_GC_0/smooth_{s}'
with open(f'{fpp}/params', 'rb') as handle:
    params = pickle.load(handle)   


# define parameter to be fitted 
paramis = ['krecA', 'krelA', 'betaA']
#speeds = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
speeds = np.array([0.1,0.2,0.5,0.7,0.9,1.0])
# give initial conditions for k_ratio and beta, alos A and B to have different dynamics
# paramis_init = np.array([9,9,3,3])
# x0 = np.log(paramis_init)
# scales = np.array([.1,.1,.1,.1])

paramis_init = np.array([2,2,2])
x0 = np.log(paramis_init)
scales = np.array([1,1,1])



# get param suggestions
es = cma.CMAEvolutionStrategy(x0, sigma0, {'popsize' : popsize})

# load anticipation times of acm for reference 
data = []
cell = 300
for s in speeds: 
    print(s)
    fp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/ACM/ACM_slow/w_GC/w_GC_0.0/smooth_{s}'
    with open(f'{fp}/out', 'rb') as handle:
        out = pickle.load(handle)    
    

    duration = 3/s
    dt = 0.001
    #time = (np.arange(0,duration,dt)- params['tps_rf_GC_mid'][50])*1000
    time = np.arange(0,duration,dt) - (0.005*cell)/s

    data.append([time,out['RG'][cell,:]])


# ACM peak   
peak_tp_ACM_ours = []
for i in range(0,len(speeds),1):

    peak_tp_ACM_ours.append(-1*data[i][0][data[i][1].argmax()])

res = np.array(peak_tp_ACM_ours)*speeds


#fitting
errors_all = []
for n in range(nb_repeats):
    # test params
    errors = []
    paramsets = es.ask()           # draw paramsets
    for pset in paramsets: 


        pset = np.exp(pset) *scales
        # put values in params


        #make param_values 
        #pset_vals = [1,1,1/pset[0], 1/pset[1], pset[2], pset[3]]
        pset_vals = [pset[0], pset[1], pset[2]]
        for i,parami in enumerate(paramis):
            params[parami]=pset_vals[i]
            print(f'{parami} = {pset_vals[i]}')

        # get output 
        t_maxis = []
        preds = []
        err_R = 0
        for si,s in enumerate(speeds): 
            print(f'speed = {s} mm/s')
            params['speed'] = s
            duration = params['distance'] / s
            print(f'duration = {duration} s')
            params['duration'] =  duration 
            dt = params['dt']

            time = (np.arange(0,duration,dt) - (params['spacing']*300)/s)#+45
            tps = len(time)
            params['tps'] = tps


            # create stimulus
            stimulus_maker = stim_moving_object_for_2D_net(params,
                                                            filepath = None)
            # inp = stimulus_maker.smooth_motion()

            bar = stimulus_maker.bar_smooth()
            # tkern = stimulus_maker.alpha_kernel()

            _ = stimulus_maker.load_filter()
            tkern = stimulus_maker.filter_biphasic_norm()
            # plt.plot(tkern)
            # plt.show()
            _,_ = stimulus_maker.OPL()
            inp = stimulus_maker.F()

            out = simulate(inp,params)
            RG = out[-1][300,:]

            #time = (np.arange(0,duration,dt)- params['tps_rf_GC_mid'][50])*1000
            maxi =-1*time[RG.argmax()]
            t_maxis.append(maxi)
            preds.append([time,RG])
            
            err_R = err_R + compute_mse(data[si][1],RG)

        t_maxis = np.array(t_maxis)
        space_maxis = t_maxis*speeds
        # calsulate slope 
        popt,_ = curve_fit(slopef,xdata = speeds, ydata = space_maxis)
        sl = np.abs(popt[0])*1000
        diff = np.abs(np.mean(np.diff(space_maxis)))*1000

        #error for ganglion rates




        # error for anticipation times
        err = compute_mse(res,space_maxis)
        errors.append(err+err_R)




    errors = np.asarray(errors)
    es.tell(paramsets,errors)
    es.disp()
    errors_all.append(errors.mean()) 

    bestparams = es.result.xbest
    paramis_best = np.exp(bestparams) *scales
    #bestis_vals = [1,1,1/ paramis_best[0], 1/ paramis_best[1],  paramis_best[2], paramis_best[3]]
    bestis_vals = [pset[0], pset[1], pset[2]]

    # put values in params
    print('optimized parameter')
    for i,parami in enumerate(paramis):
        params[parami]=bestis_vals[i]
        print(f'{parami} = {bestis_vals[i]}')



    #simulate with best params so far 
    t_maxis = []
    preds = []
    for si,s in enumerate(speeds): 
        params['speed'] = s
        duration = params['distance'] / s
        params['duration'] =  duration 
        dt = params['dt']

        time = (np.arange(0,duration,dt) - (params['spacing']*300)/s)#+45
        tps = len(time)
        params['tps'] = tps
        print(params['duration'])


        # create stimulus
        stimulus_maker = stim_moving_object_for_2D_net(params,
                                                        filepath = None)
        # inp = stimulus_maker.smooth_motion()

        bar = stimulus_maker.bar_smooth()
        # tkern = stimulus_maker.alpha_kernel()

        _ = stimulus_maker.load_filter()
        tkern = stimulus_maker.filter_biphasic_norm()
        # plt.plot(tkern)
        # plt.show()
        _,_ = stimulus_maker.OPL()
        inp = stimulus_maker.F()

        out = simulate(inp,params)
        RG = out[-1][300,:]

        #time = (np.arange(0,duration,dt)- params['tps_rf_GC_mid'][50])*1000
        maxi =-1*time[RG.argmax()]
        t_maxis.append(maxi)
        preds.append([time,RG])
        
    t_maxis = np.array(t_maxis)
    space_maxis = t_maxis*speeds
    # calsulate slope 
    popt,_ = curve_fit(slopef,xdata = speeds, ydata = space_maxis)
    sl = np.abs(popt[0])*1000


 

    #show result 


    fig,ax = plt.subplots(len(speeds))

    for si,s in enumerate(speeds):
        ax[si].plot(data[si][0], data[si][1], label ='ACM')
        ax[si].plot(preds[si][0], preds[si][1], label = 'RAM-P')
        ax[si].set_title(f'speed = {s}', loc = 'left')
        ax[si].set_xlabel(f'time [s]')
        ax[si].set_ylabel(f'R(t)')
        ax[i].legend()

    plt.figure()
    plt.scatter(speeds,res, label = 'ACM')
    plt.scatter(speeds,space_maxis, label = 'RAM-P')
    plt.axhline(0, color = 'grey', linestyle = ':')
    plt.axvline(1, color = 'grey', linestyle = ':')
    plt.xscale('log')
    plt.legend()
    fig.savefig(f'{fpp}/fitted_output.png')



plt.plot(errors_all)
plt.title('mean error aftereach iteration')
plt.show()

with open(f'{fpp}/params_occupancy_opt3', 'wb') as handle:
    pickle.dump(params, handle)