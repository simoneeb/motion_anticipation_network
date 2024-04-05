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

# evaluate erorrs
def compute_mse(y,prediction):
        
    error = np.sum((prediction-y)**2)
    error /= len(y)
    error = np.sqrt(error)
    return error


def simulate(inp):
       # run stimulation 
    #params = stimulus_maker.add_params()

    # create weight matrices
    inp = inp*params['input_scale']
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

    predres = [RB[300,:],VG[300,:],RG[300,:]]

    return predres

#hyperparameter 
sigma0 = 1
popsize = 10
nb_repeats = 5


# get desired output 
# bipolar response after gain control
# ganglion response before gain control ??
# ganglion response after gain control ?? 
s = 0.81
fp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/ACM/ACM_slow/w_GC/w_GC_0.0/smooth_{s}'
with open(f'{fp}/out', 'rb') as handle:
    out = pickle.load(handle)   

res = [out['RB'][300,:],out['VG'][300,:],out['RG'][300,:]]


fig,ax = plt.subplots(3)
ax[0].plot(res[0])
ax[1].plot(res[1])
ax[2].plot(res[2])

ax[0].set_title('RB', loc = 'left')
ax[1].set_title('VG', loc = 'left')
ax[2].set_title('RG', loc = 'left')
plt.show()

# load initial paramset
s = 0.81
fp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/Reciporcal/GainControl_slow/w_GC/w_GC_0/smooth_{s}'
with open(f'{fp}/params', 'rb') as handle:
    params = pickle.load(handle)   


# define parameter to be fitted 
paramis = ['input_scale','wGB', 'tauG', 'tauB']

# give initial conditions
paramis_init = np.array([8,2,8,9])
x0 = np.log(paramis_init)
scales = np.array([100,0.1,0.01,0.01])

# get param suggestions
es = cma.CMAEvolutionStrategy(x0, sigma0, {'popsize' : popsize})


# create stimulus
stimulus_maker = stim_moving_object_for_2D_net(params,
                                                filepath = None)
# inp = stimulus_maker.smooth_motion()

bar = stimulus_maker.bar_smooth()
#tkern = stimulus_maker.alpha_kernel()

_ = stimulus_maker.load_filter()
tkern = stimulus_maker.filter_biphasic_norm()
# plt.plot(tkern)
# plt.show()
_,inp = stimulus_maker.OPL()


errors_all = []
for n in range(nb_repeats):
    # test params
    errors = []
    paramsets = es.ask()           # draw paramsets
    for pset in paramsets: 


        pset = np.exp(pset) *scales
        # put values in params
        for i,parami in enumerate(paramis):
            params[parami]=pset[i]
            # print(f'{parami} = {pset[i]}')

        # get output 
        out = simulate(inp)
        # compare to desired result 
        err = 0
        for i in range(len(out)):
            err = err + compute_mse(res[i],out[i])

        # save error
        errors.append(err)


    errors = np.asarray(errors)
    es.tell(paramsets,errors)
    es.disp()

    bestparams = es.result.xbest
    paramis_best = np.exp(bestparams) *scales
    # put values in params
    for i,parami in enumerate(paramis):
        params[parami]=paramis_best[i]
        print(f'{parami} = {paramis_best[i]}')
    bestout = simulate(inp)

    fig,ax = plt.subplots(3)
    ax[0].plot(res[0], label = 'ACM')
    ax[1].plot(res[1])
    ax[2].plot(res[2])

    ax[0].plot(bestout[0], label = 'ACM DS')
    ax[1].plot(bestout[1])
    ax[2].plot(bestout[2])

    ax[0].set_title('RB', loc = 'left')
    ax[1].set_title('VG', loc = 'left')
    ax[2].set_title('RG', loc = 'left')

    fig.legend()
    plt.show()

    errors_all.append(errors.mean())


# bestparams = es.result.xbest
# paramis_best = np.exp(bestparams) *scales
# # put values in params
# for i,parami in enumerate(paramis):
#     params[parami]=paramis_best[i]
#     print(f'{parami} = {paramis_best[i]}')
# bestout = simulate(inp)

plt.plot(errors_all)
plt.title('mean error aftereach iteration')
plt.show()
# plt.plot(res, label = 'ACM')
# plt.plot(bestout[300,:], label = 'Reciporcal')
# plt.legend()
# plt.show()

# save best params
with open(f'{fp}/params_opt', 'wb') as handle:
    pickle.dump(params, handle)
with open(f'{fp}/params_opt.json', 'wb') as handle:
    json.dump(params, handle)
# compare other outputs