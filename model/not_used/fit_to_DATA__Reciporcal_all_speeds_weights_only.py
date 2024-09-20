import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import cma

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from joblib import Parallel,delayed

from stimuli import stim_moving_object_for_2D_net
from run_Reciporcal import run_Reciporcal
from model.params_FB import make_params, modify_params



# TODO move functions to utils
def N(V):

    if V <= 0:
        return 0
    else:
        return V

def gauss(x, mu, sig_c):

    kern =  np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig_c, 2.))) 
    kern = kern / kern.max()
    return kern

# evaluate erorrs
def compute_mse(y,prediction):
        
    error = np.sum((prediction-y)**2)
    error /= len(y)
    error = np.sqrt(error)
    return error


def normalize01(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))




cell_nb = 431                         # cell to fit 
model_name  = 'reciprocal_linear_long_no_tauB'   # model name 
speeds = [0.14,0.42,0.7,0.98,1.96]       # speeds to simulate 
     

# hyperparameter for optimization
sigma0 = 2                 # STD for parameter draw
popsize = 15               # number of parametersets for each draw 
nb_repeats = 20            # number of repeats 


# get desired output 
fp = '/Users/simoneebert/Documents/Experiments/motion_anticipation/Simone/small_anticipation_dict.pickle'
with open(fp, 'rb') as handle:
    small_dict = pickle.load(handle)


# fit GC RF size to STA 
rf = small_dict[cell_nb]['rf']           # load data
rf_norm = (rf-rf.mean())/rf.std()        # normalize


rf_space = small_dict['rfspace']/1000                              # get x-rage of experiment
rf_space_highres = np.arange(rf_space.min(),rf_space.max(),0.01)   # increase resolution for plot
center = rf_space[np.argmax(rf)]                                   # get center coodrinate as initial param
popt,_ = curve_fit(gauss,rf_space,rf_norm, p0=(center,10))        # estimate gsussian fit 


# #test different options for RF optimization
# rf_smooth = gaussian_filter1d(rf,1)      # smoothen 
# rf_rect = np.array([N(r) for r in rf])   # rectify

# popt_norm,_ = curve_fit(gauss,rf_space,rf_norm, p0=(center,100))
# popt_smooth,_ = curve_fit(gauss,rf_space,rf_smooth, p0=(center,100))
# popt_rect,_ = curve_fit(gauss,rf_space,rf_rect, p0=(center,100))

# lin = plt.plot(rf_space,rf_smooth, label = 'data smooth')
# plt.plot(rf_space_highres,gauss(rf_space_highres,*popt_smooth), label = 'fit smooth', linestyle = ':', color = lin[0].get_color())
# lin = plt.plot(rf_space,rf_norm, label = 'data nrom')
# plt.plot(rf_space_highres,gauss(rf_space_highres,*popt_norm), label = 'fit nrom', linestyle = ':', color = lin[0].get_color())
# lin = plt.plot(rf_space,rf_rect, label = 'data nrom')
# plt.plot(rf_space_highres,gauss(rf_space_highres,*popt_rect), label = 'fit rect', linestyle = ':', color = lin[0].get_color())


params = make_params() # initialize params 

std_GC = popt[1] # use GC rf size from data as STD for pooling
dt = 0.01        # define timestep
dt_exp = 0.025   # set timestep of experimentls 
params = modify_params(params,['std_GC,dt'],[std_GC,dt])   # change parameter 

paramis = ['wBA','wAB','tauA']  # define parameter to be fitted 
paramis_init = np.array([2,2,2])     # give initial conditions, all need to be in the same oder of maginude
scales = np.array([1,1,0.1])       # define scales for params
#x0 = np.log(paramis_init)             # set inital conditions as log to avoid negative values 
x0 = paramis_init                      # set inital conditions as log to avoid negative values 


# generate network input 
stimulus_maker = stim_moving_object_for_2D_net(params, filepath = None)  # create stimulus
bar = stimulus_maker.bar_smooth()                # make bar

_ = stimulus_maker.load_filter()                 # set filter param from data 
tkern = stimulus_maker.filter_biphasic_norm()    # generate filter 

_,_ = stimulus_maker.OPL()                       # simulate OPL response
inp = stimulus_maker.F()                         # simulate input to dynamical system 



# run simulations of all speeds for one parameterset and retrun error 
def run_one_pset(pset):

    #pset = np.exp(pset) *scales                                            # create parameter values 
    pset = pset *scales                                            # create parameter values 
    paramss = modify_params(params,paramis,pset)                           # modify params 
    # TODO error temporal STA profile

    err_over_speeds = []
    for speed in speeds:                                                   # loop over speeds 

        paramss = modify_params(paramss,['speed'], [speed])                # modify speed in params
        simu = run_Reciporcal(paramss)                                     # run simulation 

        res = small_dict[cell_nb]['centered_bar_responses'][speed]         # experimental response centered arout time point where bar center is at RF cetner
        res_time = np.arange(0,len(res))*dt_exp                            # corresponding time with dt = bin_bize, but starting at 0  
        #res_time = small_dict['times'][speed]
        resfun = interp1d(res_time,res, fill_value='extrapolate')          # interpolation of response
        time_dt = np.arange(0,res_time[-1],params['dt'])                   # new time with dt of simulation 
        res_dt = resfun(time_dt)                                           # new response 

        tps_res = len(res_dt)                                              # get length of res_dt
        tps_half = int(np.floor(tps_res/2))                                # get middle point 

        if tps_res%2 != 0:                                                 # check if even numner of timesteps 
            print('!!! uneven numner of timepoints, RF middle crossing is inaccurate')

        tp_rf_crossing = np.round(simu[3],2)                               # time of bar center at rf center
        idx_rf_crossing = int(tp_rf_crossing/dt)                           # convert to index
        pred = simu[-2]                                                    # predicted response 
        pred =  pred[idx_rf_crossing - tps_half:idx_rf_crossing+tps_half]  # cernter around bar crossing and crop to length of experimental response

        res_dt = normalize01(res_dt)                 
        pred = normalize01(pred)
        err = compute_mse(res_dt,pred)
        err_over_speeds.append(err)
        
    err = np.mean(err_over_speeds)

    # if any(pset) <= 0:
    #     err = err + 1
    
    # if pset[2]>= 0.6:
    #     err = err + 1

    # if pset[3]>= 0.6:
    #     err =err + 1

    return err



# run the optimization

es = cma.CMAEvolutionStrategy(x0, sigma0, {'popsize' : popsize})                         # initialize fitting
errors_all = []                        

for n in range(nb_repeats):                                                              # loop over repeats 

    # test params
    errors = []
    paramsets = es.ask()           # draw paramsets

    X = Parallel(n_jobs = 2, verbose=10)(delayed(run_one_pset)(i) for i in paramsets)    # run simulations with paramsets in parallel

    errors = np.asarray(X)                                
    es.tell(paramsets,errors)                                                            # evaluate paramsets
    es.disp()                                                                           

    bestparams = es.result.xbest                                                         # export best paramset 
    #paramis_best = np.exp(bestparams) *scales                                            # transform into used values 
    paramis_best = bestparams *scales                                            # transform into used values 

    errors_all.append(errors.mean())                                                     # save error 




# create output directory
fpout = f'/Users/simone/Documents/Simulations/motion_anticipation_network/{model_name}_fitted_cell_{cell_nb}'
if not os.path.isdir(fpout):
    os.mkdir(fpout)


# print and save best params
for i,parami in enumerate(paramis):
    print(f'{parami} = {paramis_best[i]}')

params = modify_params(params,paramis,paramis_best)
with open(f'{fpout}/params_fitted_cell_{cell_nb}', 'wb') as handle:
    pickle.dump(params, handle)
    
# save intital params
# paramis_init = np.exp(paramis_init) * scales
paramis_init = paramis_init * scales
params_init = modify_params(params,paramis,paramis_init)

with open(f'{fpout}/params_init_cell_{cell_nb}', 'wb') as handle:
    pickle.dump(params_init, handle)
  

# plot RF fit 
fig_rf = plt.figure()
lin = plt.plot(rf_space,rf, label = 'data raw')
plt.plot(rf_space_highres,gauss(rf_space_highres,*popt), label = 'fit raw', linestyle = ':', color = lin[0].get_color())
plt.legend()
plt.title(f' mu = {popt[0]},sigma = {popt[1]}')

# plot error 
fig_err = plt.figure()
plt.plot(errors_all)
plt.xlabel('iteration')
plt.ylabel('MSE')


# save figs 
fig_rf.savefig(f'{fpout}/fit_RF.png')
fig_err.savefig(f'{fpout}/fit_MSE.png')

plt.close()

