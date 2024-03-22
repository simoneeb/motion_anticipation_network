
import os
import numpy as np
from joblib import Parallel,delayed
from run_Reciporcal import run_Reciporcal
from params_Reciporcal import make_params
import time
import pandas as pd
import pickle
from utils import measure_onset_anticipation

# def measure_onset_anticipation(sim):

#     onset_sim =np.argmax(sim >=1)
#     #onset_ref =np.argmax(ref >=1)

#     return onset_sim
#     #return onset_ref - onset_sim

net_name = f'Reciporcal_fitted_mono_linear/noGCGainControl'

stim_type = 'smooth'

#loop over parameter
vals = np.arange(0,150,10)
vals =[20,10,0.0]
speeds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0]
speeds = [0.5,0.8]


# load ref for motion onset

refdict = {}

for si in speeds:
    fp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/Reciporcal/{net_name}/wBA/wBA_0.0/smooth_{si}'

    with open(f'{fp}/out', 'rb') as handle:
        out = pickle.load(handle)
    with open(f'{fp}/params', 'rb') as handle:
        params = pickle.load(handle)

    dt = params['dt']

    refdict[f'{si}'] = {}
    refdict[f'{si}']['RG'] = out['RG']
    refdict[f'{si}']['RB'] = out['RB'][50]
    refdict[f'{si}']['max_tp_RG'] = np.argmax(out['RG'])*dt
    refdict[f'{si}']['max_tp_RB'] = np.argmax(out['RB'][50])*dt

df = pd.DataFrame(columns=['wAB','wBA', 'speed', 'peak_RG','peak_RB', 'peak_drive', 'tp_rf_GC_mid', 'peak_RG_pooling', 'peak_RB_pooling', 'onset_RB', 'onset_RG'])
# df = pd.DataFrame(columns=['wAB','wBA', 'speed', 'ant_space', 'ant_time', 'ant_space_drive', 'ant_time_drive', 'onset_shift'])
# dfresRG = pd.DataFrame()
# dfresRB = pd.DataFrame()
grid = np.array(np.meshgrid(vals,vals,speeds)).T.reshape(-1,3)



def run(val1,val2,si):

    val1 = np.round(val1,2)
    val2 = np.round(val2,2)
    params = make_params(param_names = ['speed','wBA','wAB'], param_vals=[si,val1,val2])
    [peak_RG,peak_RB,peak_drive,tp_rf_GC_mid,onset_RG,onset_RB] = run_Reciporcal(params = params)   
    # [ant_space,ant_time,ant_space_drive,ant_time_drive,RG] = run_Reciporcal(params = params)   
    # onset_RG = measure_onset_anticipation(RG)
    # onset_RB = measure_onset_anticipation(RB)
    # peak_RG_pooling = refdict[f'{si}']['max_tp_RG']
    # peak_RB_pooling = refdict[f'{si}']['max_tp_RB']

    # RG = RG[0::10]
    # RB = RB[0::10]

    data = {'wBA': val1,
            'wAB':val2,
            'speed' : si,
            'peak_RG' : peak_RG,
            'peak_RB' : peak_RB,
            'peak_drive' : peak_drive,
            # 'peak_RG_pooling' : peak_RG_pooling,
            # 'peak_RB_pooling' : peak_RB_pooling,
            'tp_rf_GC_mid' : tp_rf_GC_mid,
            'onset_RG' : onset_RG,
            'onset_RB' : onset_RB}
    
    
  
    
    return [data,params]



start = time.time()

X = Parallel(n_jobs = 6, verbose=10)(delayed(run)(i[0],i[1],i[2]) for i in grid)

for i,xi in enumerate(X):
    data = xi[0]
    speed = data['speed']
    data['peak_RG_pooling'] =  refdict[f'{speed}']['max_tp_RG']
    data['peak_RB_pooling'] =  refdict[f'{speed}']['max_tp_RB']

    data['onset_RG_pooling'] =  measure_onset_anticipation(refdict[f'{speed}']['RG'])
    data['onset_RB_pooling'] =  measure_onset_anticipation(refdict[f'{speed}']['RB'])

    df = df.append(data, ignore_index = True)
    

    # add pooling peak 

    # add pooling onset


    # RG = xi[1]
    # RB = xi[2]
    # dfnRG = pd.DataFrame({ f'{i}' : RG})
    # dfnRB = pd.DataFrame({ f'{i}' : RB})
    # dfresRG = pd.concat([dfresRG,dfnRG], ignore_index=True, axis=1)
    # dfresRB = pd.concat([dfresRB,dfnRB], ignore_index=True, axis=1)

df.to_csv(f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/Reciporcal/{net_name}/anticipation_data.csv')
# dfresRG.to_csv(f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/Reciporcal/{net_name}/responses_RG.csv')
# dfresRB.to_csv(f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/Reciporcal/{net_name}/responses_RB.csv')

stop = time.time()
params = X[-1][1]
with open(f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/Reciporcal/{net_name}/params_grid', 'wb') as handle:
            pickle.dump(params, handle)


print('Elapsed time for the entire processing: {:.2f} s'
      .format(stop - start))


# for val in vals:
#     val = np.round(val,2)
#     params_name = f'{param}/{param}_{val}'
#     print(f'{param} = {val}')
#     # loop over speeds : 
#     for si in speeds:
#         stim_name = f'{stim_type}_{si}'
#         filepath = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/Reciporcal/{net_name}'
#         print(f'speed = {si}')
#         os.system(f'python params_Reciporcal.py {filepath}/{params_name}/{stim_name} speed {si} {param} {val}')
#         # # # # #os.system(f'python params.py {filepath}/{params_name}/{stim_name} speed {si} wBA {-1*val} wAB {val}')
#         os.system(f'python run_Reciporcal.py {filepath}/{params_name}/{stim_name} None')
#         # os.system(f'python plot_codes/plot_BC_GC_compare.py {filepath}/{param} {param} {val} {stim_name}')
#         os.system(f'python plot_codes/plot_Reciporcal_one.py {filepath}/{param} {param} {val} {stim_name}')
    
#     os.system(f'python plot_codes/plot_speeds_auto_one.py {filepath} {stim_type} {param} {val} ')
#     # os.system(f'python plot_speeds_auto.py {filepath} {stim_type} {param} {val} ')
#     # #os.system(f'python plot_pva.py {filepath} {stim_type} {param} {val} ')
#     # os.system(f'python plot_speeds_gaincontrol_mechanism.py {filepath} {stim_type} {param} {val}')
#     # #os.system(f'python plot_speeds_lateral_mechanism.py {filepath} {stim_type} {param} {val}')

