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
from run_Reciporcal import run_Reciporcal
from params_Reciporcal import make_params   
from joblib import Parallel,delayed


# evaluate erorrs
def compute_mse(y,prediction):
        
    error = np.sum((prediction-y)**2)
    error /= len(y)
    error = np.sqrt(error)
    return error


#hyperparameter 
sigma0 = 1
popsize = 20
nb_repeats = 10


# get desired output 
# bipolar response after gain control

s = 0.81
fp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/Reciporcal/Reciporcal_fitted/noGCGainControl/wBA/wBA_31.0/smooth_{s}'
with open(f'{fp}/out', 'rb') as handle:
    out = pickle.load(handle)   

res = out['RB'] 
res = res[50,:]


# load initial paramset

with open(f'{fp}/params', 'rb') as handle:
    params = pickle.load(handle)   

params['speed'] = 0.81
# params['SF']: 0.
# params['hB'] = 0.
# params['nb_cells'] = 600
# params['nb_GC_cells'] = 600
# params['wAB'] = 10.0
# define parameter to be fitted 
paramis = ['wBA','wAB','tauB','tauA']

# give initial conditions
paramis_init = np.array([6,6,9,9])
x0 = np.log(paramis_init)
scales = np.array([10,10,0.01,0.01])

# get param suggestions
es = cma.CMAEvolutionStrategy(x0, sigma0, {'popsize' : popsize})


# create stimulus
stimulus_maker = stim_moving_object_for_2D_net(params,
                                                filepath = None)

bar = stimulus_maker.bar_smooth()

_ = stimulus_maker.load_filter()
tkern = stimulus_maker.filter_biphasic_norm()

_,_ = stimulus_maker.OPL()
inp = stimulus_maker.F()


def run_one_pset(pset):

        pset = np.exp(pset) *scales
        paramss = make_params(paramis,pset)

        simu = run_Reciporcal(paramss)




        err = compute_mse(res,simu[-1])

        return err


errors_all = []

for n in range(nb_repeats):
    # test params
    errors = []
    paramsets = es.ask()           # draw paramsets

    X = Parallel(n_jobs = 6, verbose=10)(delayed(run_one_pset)(i) for i in paramsets)



    errors = np.asarray(X)
    es.tell(paramsets,errors)
    es.disp()

    bestparams = es.result.xbest
    paramis_best = np.exp(bestparams) *scales


    # put values in params
    for i,parami in enumerate(paramis):
        params[parami]=paramis_best[i]
        print(f'{parami} = {paramis_best[i]}')


    errors_all.append(errors.mean())

# save best params
with open(f'{fp}/params_opt_mono_to_fitted', 'wb') as handle:
    pickle.dump(params, handle)
