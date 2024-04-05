import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from run_Reciporcal import run_Reciporcal
from params_Reciporcal import make_params, modify_params
from nonlinearities import N



# TODO move to utils
def N(V):

    if V <= 0:
        return 0
    else:
        return V

def gauss(x, mu, sig_c):

    kern =  np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig_c, 2.))) 
    kern = kern / kern.max()
    return kern


def normalize01(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))



cell_nb = 479                                  # cell to fit 
model_name  = 'reciprocal_linear_long'      # model name 
speeds = [0.14,0.42,0.7,0.98,1.96]             # speeds to simulate 
     

# get desired output 
fp = '/Users/simone/Documents/Experiments/motion_anticipation/Simone/small_anticipation_dict.pickle'
with open(fp, 'rb') as handle:
    small_dict = pickle.load(handle)


     

#TODO save in fiting and load 
rf = small_dict[cell_nb]['rf'] 
rf_norm = (rf-rf.mean())/rf.std()
rf_smooth = gaussian_filter1d(rf,1)

rf_space = small_dict['rfspace']/1000
rf_space_highres = np.arange(rf_space.min(),rf_space.max(),0.01)
rf_rect = np.array([N(r) for r in rf])

center = rf_space[np.argmax(rf)]

popt,_ = curve_fit(gauss,rf_space,rf_norm, p0=(0,10))

# params = make_params() 


# # use GC rf size from data
# std_GC = popt[1]
dt = 0.01
dt_exp = 0.025
# params = modify_params(params,['std_GC,dt'],[std_GC,dt])
# params['std_GC'] = std_GC




# # define parameter to be fitted 
paramis = ['wBA','wAB','tauB','tauA']

# # give initial conditions
# paramis_init = np.array([4,8,1,2])
# x0 = np.log(paramis_init)
# scales = np.array([1,1,0.1,0.1])


# load best params 
fpout = f'/Users/simone/Documents/Simulations/motion_anticipation_network/{model_name}_fitted_cell_{cell_nb}'
with open(f'{fpout}/params_fitted_cell_{cell_nb}', 'rb') as handle:
    params_best = pickle.load(handle)

# load initial params
with open(f'{fpout}/params_init_cell_{cell_nb}', 'rb') as handle:
    params_init = pickle.load(handle)




# vals = [6.779169845261268,11.363323753783916,-0.10676657442322597,0.15253688899199971]
#params_best = modify_params(params_best,['tauB'],[0.001])



# print inital and optimal params
for i,parami in enumerate(paramis):
    parami_best = params_best[f'{parami}']
    parami_init = params_init[f'{parami}']
    print(f'{parami} = {parami_best} (inital = {parami_init})')


# plot simulation and compare to data 
fig_res = plt.figure(figsize = (12,14) )
gs = fig_res.add_gridspec(nrows=len(speeds),ncols=2)


antis = []
onsets = []
sim_antis = []
sim_onsets_antis = []
sim_antis_init = []
sim_onsets_init = []
preds = []
times = []

for i,speed in enumerate(speeds): 
    print(f'speed: {speed}')
    # stim_name = f'smooth_{speed}'
    params = modify_params(params_best,['speed'], [speed])                                           # set speed
    params_init = modify_params(params_init,['speed'], [speed])                                     # set speed
    simu = run_Reciporcal(params = params, filepath = None, save_one = True,stim_type='smooth')    # run simulation best
    simu_init = run_Reciporcal(params = params_init, filepath = None, save_one = True,stim_type='smooth')   # run simulation init 

    res = small_dict[cell_nb]['centered_bar_responses'][speed]         # experimental response centered arout time point where bar center is at RF cetner
    #res_time = np.arange(0,len(res))*dt_exp                            # corresponding time with dt = bin_bize, but starting at 0  
    res_time = small_dict['times'][speed]
    resfun = interp1d(res_time,res, fill_value='extrapolate')          # interpolation of response
    time_dt = np.arange(res_time[0],res_time[-1],params['dt'])                   # new time with dt of simulation 
    res_dt = resfun(time_dt)                                           # new response 


    # get length of res_dt
    tps_res = len(res_dt)
    tps_half = int(np.floor(tps_res/2))
   
    tps_res = len(res_dt)                                              # get length of res_dt
    tps_half = int(np.floor(tps_res/2))                                # get middle point 

    if tps_res%2 != 0:                                                 # check if even numner of timesteps 
        print('!!! uneven number of timepoints, RF middle crossing is inaccurate')

    tp_rf_crossing = np.round(simu[3],2)                               # time of bar center at rf center
    idx_rf_crossing = int(tp_rf_crossing/dt)                           # convert to index
    pred = simu[-2]                                                    # predicted response 
    pred =  pred[idx_rf_crossing - tps_half:idx_rf_crossing+tps_half]  # cernter around bar crossing and crop to length of experimental response
    pred_init = simu_init[-2]                                                    # predicted response 
    pred_init =  pred_init[idx_rf_crossing - tps_half:idx_rf_crossing+tps_half]  # cernter around bar crossing and crop to length of experimental response


    ax = fig_res.add_subplot(gs[i,0])                                                                                 # create panel

    sim_anti = time_dt[np.argmax(pred)]                                                                               # get response peak in simulation best
    sim_anti_init = time_dt[np.argmax(pred_init)]                                                                     # get response peak in simulation initial
    anti = time_dt[np.argmax(res_dt)]                                                                                 # get response peak in data

    lin = ax.plot(res_time,normalize01(res), label = '_data', color = 'blue', alpha  = 0.8)
    ax.axvline(anti, color = lin[0].get_color(), linestyle = ':')

    lin = ax.plot(time_dt,normalize01(pred), label = '_simulation', color = 'orange')
    ax.axvline(sim_anti, color = lin[0].get_color(), linestyle = ':')

    lin = ax.plot(time_dt,normalize01(pred_init), label = '_simulation init', color = 'orange', alpha = 0.3)
    ax.axvline(sim_anti_init, color = lin[0].get_color(), linestyle = ':', alpha = 0.3)

    ax.set_ylabel('response')
    ax.set_xlabel('time [s]')
    ax.set_title(f'{speed} mm/s', loc = 'left')
   
    antis.append(anti)
    sim_antis.append(sim_anti)
    sim_antis_init.append(sim_anti_init)
    preds.append(pred)
    times.append(time_dt)

ax.set_xlabel('time [s]')



ax = fig_res.add_subplot(gs[3:,1])                                                                            # plot aticipation in space vs speed

antis = -1*np.array(antis)
sim_antis = -1*np.array(sim_antis)
sim_antis_init = -1*np.array(sim_antis_init)

plt.scatter(speeds,antis*speeds, label = '_experiment', color = 'blue')
plt.scatter(speeds,sim_antis*speeds, label = '_simulation', color = 'orange')
plt.scatter(speeds,sim_antis_init*speeds, label = '_simulation init', color = 'orange',alpha = 0.3)

plt.plot(speeds,antis*speeds,label = 'data', color = 'blue')
plt.plot(speeds,sim_antis*speeds, label = 'simulation', color = 'orange')
plt.plot(speeds,sim_antis_init*speeds, label = 'simulation init', color = 'orange',alpha = 0.3)

plt.axhline(0, color = 'k', linestyle = ':')
ax.set_xscale('log')
ax.set_ylabel(f'distance anticipated [\mm$]')
ax.set_xlabel('speed [mm/s]')

plt.xticks(speeds)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax.legend()



ax = fig_res.add_subplot(gs[3:,1])                                                                            # plot onset aticipation as response start in time  - time when leading bar edge enters RF 


plt.scatter(speeds,antis*speeds, label = '_experiment', color = 'blue')
plt.scatter(speeds,sim_antis*speeds, label = '_simulation', color = 'orange')
plt.scatter(speeds,sim_antis_init*speeds, label = '_simulation init', color = 'orange',alpha = 0.3)

plt.plot(speeds,antis*speeds,label = 'data', color = 'blue')
plt.plot(speeds,sim_antis*speeds, label = 'simulation', color = 'orange')
plt.plot(speeds,sim_antis_init*speeds, label = 'simulation init', color = 'orange',alpha = 0.3)

plt.axhline(0, color = 'k', linestyle = ':')
ax.set_xscale('log')
ax.set_ylabel(f'distance anticipated [\mm$]')
ax.set_xlabel('speed [mm/s]')

plt.xticks(speeds)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax.legend()


axe = fig_res.add_subplot(gs[0,1])                                                                          # plot all responses in space data 


ax2 = axe.twinx()                                                                                              # add the receptive field
ax2.fill_between(small_dict['rfspace'],y1 = small_dict[cell_nb]['rf'], color = 'r', linewidth = 4, alpha = 0.2, label = 'RF' )
ax2.legend()
ax2.set_yticks([])


axe.set_zorder(ax2.get_zorder()+1)                                  
axe.patch.set_visible(False)                                        # prevents ax1 from hiding ax2

for i,speed in enumerate(speeds):
    space = small_dict['spaces'][speed]
    res = small_dict[cell_nb]['centered_bar_responses'][speed]
    ant = space[np.argmax(res)]
    lin = axe.plot(space,res, label = f'{speed} mm/s', linewidth = 3)
    axe.axvline(ant, color = lin[0].get_color(), linestyle = ':')
axe.legend(bbox_to_anchor = (1.,.9))
print(space)

ax_ylims = axe.axes.get_ylim()           # Find y-axis limits set by the plotter
ax_yratio = ax_ylims[0] / ax_ylims[1]    # Calculate ratio of lowest limit to highest limit

ax2_ylims = ax2.axes.get_ylim()           # Find y-axis limits set by the plotter
ax2_yratio = ax2_ylims[0] / ax2_ylims[1]  # Calculate ratio of lowest limit to highest limit

if ax_yratio < ax2_yratio: 
    ax2.set_ylim(bottom = ax2_ylims[1]*ax_yratio)
else:
    axe.set_ylim(bottom = ax_ylims[1]*ax2_yratio)

axe.set_ylabel('firing rate [spikes/s]')
axe.set_xlabel('bar center distance to RF center [$\mu m$]')
axe.set_title('Experimental responses in space')


ax = fig_res.add_subplot(gs[1,1])                           # plot all responses in space simulations 


ax2 = ax.twinx()                                            # plot receptive field (GC pooling weights)
ax2.fill_between(rf_space_highres*1000,gauss(rf_space_highres,*popt), color = 'r', linewidth = 4, alpha = 0.2, label = 'RF' )
ax2.fill_between(rf_space_highres*1000,gauss(rf_space_highres,popt[0],popt[1]+params_best['rf_BC']/6), color = 'r', linewidth = 1, alpha = 0.1, label = 'RF with BC radius ' )
print(rf_space_highres*1000)
ax2.legend()
ax2.set_yticks([])


ax.set_zorder(ax2.get_zorder()+1)     # make responses be over RF
ax.patch.set_visible(False)           # prevents ax1 from hiding ax2

for i,speed in enumerate(speeds):
    print(len(times[i]),len(preds[i]))
    lin = ax.plot(times[i]*speed*1000,preds[i], label = f'{speed} mm/s', linewidth = 3)


#ax.legend(bbox_to_anchor = (.72,.9))
ax_ylims = ax.axes.get_ylim()             # Find y-axis limits set by the plotter
ax_yratio = ax_ylims[0] / ax_ylims[1]     # Calculate ratio of lowest limit to highest limit

ax2_ylims = ax2.axes.get_ylim()           # Find y-axis limits set by the plotter
ax2_yratio = ax2_ylims[0] / ax2_ylims[1]  # Calculate ratio of lowest limit to highest limit

if ax_yratio < ax2_yratio: 
    ax2.set_ylim(bottom = ax2_ylims[1]*ax_yratio)
else:
    ax.set_ylim(bottom = ax_ylims[1]*ax2_yratio)

ax.set_ylabel('R(t)')
ax.set_xlabel('bar center distance to RF center [$\mu m$]')
ax.set_title('Simulated responses in space')

fig_res.suptitle(f' {model_name} model fit to cell {cell_nb}')
fig_res.savefig(f'{fpout}/fit_RES.png')
plt.close()
