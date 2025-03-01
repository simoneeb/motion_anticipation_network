
import os
import numpy as np
from joblib import Parallel,delayed
from run_Reciporcal import run_Reciporcal
from params_FB import make_params
import time
import pandas as pd
import pickle
import sys


from utils import measure_onset_anticipation

net_name = f'fb_linear_512'

stim_type = 'smooth'

# loop over parameter
n_params = 50
vals2 = np.linspace(1,81,n_params)
vals2 = np.linspace(1,101,n_params) *46.0
vals2 = np.linspace(1,51,n_params) 

# speeds = [0.2,0.23,0.25,0.27,0.3,0.32,0.35,0.37,0.4,0.42,0.45,0.47,0.5]
# speeds = np.asarray([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9])
# speeds = np.asarray([0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9])
speeds = np.round(np.arange(0.1,4.0,0.05),2)

nb_jobs = len(vals2)*len(speeds)
dur = (nb_jobs*11.5)/60
print(f'nb jobs :  {nb_jobs}, takes {dur} mins')

# load ref for motion onset

refdict = {}

# for si in speeds:
#     fp = f'~/Documents/Simulations/motion_anticipation_network/Loops/{net_name}/wAB/wAB_0.0/smooth_{si}'
#     #fp = f'~/Documents/Simulations/motion_anticipation_network/Loops/{net_name}/wGA/wGA_0.0/smooth_{si}'

#     with open(f'{fp}/out', 'rb') as handle:
#         out = pickle.load(handle)
#     with open(f'{fp}/params', 'rb') as handle:
#         params = pickle.load(handle)

#     dt = params['dt']

#     refdict[f'{si}'] = {}
#     refdict[f'{si}']['RG'] = out['RG']
#     refdict[f'{si}']['RB'] = out['RB'][50]
#     refdict[f'{si}']['max_tp_RG'] = np.argmax(out['RG'])*dt
#     refdict[f'{si}']['max_tp_RB'] = np.argmax(out['RB'][50])*dt

df = pd.DataFrame(columns=['wTOT','tauTOT','wBA', 'wAB', 'tauA', 'tauB','mu', 'speed', 'peak_RG','peak_RB', 'peak_drive', 'tp_rf_GC_mid', 'peak_RG_pooling', 'peak_RB_pooling', 'onset_RB', 'onset_RG'])
# df = pd.DataFrame(columns=['wAB','wBA', 'speed', 'ant_space', 'ant_time', 'ant_space_drive', 'ant_time_drive', 'onset_shift'])
dfresRG = pd.DataFrame()
dfresRB = pd.DataFrame()
grid = np.array(np.meshgrid(vals2,speeds)).T.reshape(-1,2)


#TODO add output to dataframe durinng the parallel run to avoid blocking memory
def run(val2,si):

    params = make_params(['speed'],[si])
    params['wAB'] = 10.
    params['wGA'] = 0.#0.004
    
    wAB=  params['wAB']
    wGA=  params['wGA']

    # tauTOT = np.round(val1,2)
    # wTOT = np.round(val2,2)
    
    # tauA = 1/(tauTOT + 1/tauB)
    # wBA = wTOT/wAB

    wBA = np.round(val2,2)
    
    wTOT = wBA*wAB



    # print(tauA)
    # print(wBA)
    params['wBA'] = wBA
    #params = make_params(param_names = ['speed','wAB'], param_vals=[si,wAB])

    [peak_RG,peak_RB,peak_drive,tp_rf_GC_mid,amp_RB,onset_RG,onset_RB,RG,RB,VG] = run_Reciporcal(params = params)   


    # [ant_space,ant_time,ant_space_drive,ant_time_drive,RG] = run_Reciporcal(params = params)   
    # onset_RG = measure_onset_anticipation(RG)
    # onset_RB = measure_onset_anticipation(RB)
    # peak_RG_pooling = refdict[f'{si}']['max_tp_RG']
    # peak_RB_pooling = refdict[f'{si}']['max_tp_RB']

    RG = RG[0::10]
    RB = RB[0::10]

    data = {'wAB': wAB,
            'tauB': params['tauB'],
            'wTOT': wTOT,
            'wBA': wBA,
            'wGA': wGA,
            'speed' : si,
            'peak_RG' : peak_RG,
            'peak_RB' : peak_RB,
            'amp_RB' : amp_RB,
            'peak_drive' : peak_drive,
            # 'peak_RG_pooling' : peak_RG_pooling,
            # 'peak_RB_pooling' : peak_RB_pooling,
            'tp_rf_GC_mid' : tp_rf_GC_mid,
            'onset_RG' : onset_RG,
            'onset_RB' : onset_RB}
    
    
  
    
    return [data,RG,RB]



start = time.time()

X = Parallel(n_jobs = 10, verbose=10)(delayed(run)(i[0],i[1]) for i in grid)

print(sys.getsizeof(X))

for i,xi in enumerate(X):
    data = xi[0]
    speed = data['speed']
    # data['peak_RG_pooling'] =  refdict[f'{speed}']['max_tp_RG']
    # data['peak_RB_pooling'] =  refdict[f'{speed}']['max_tp_RB']

    # data['onset_RG_pooling'] =  measure_onset_anticipation(refdict[f'{speed}']['RG'])
    # data['onset_RB_pooling'] =  measure_onset_anticipation(refdict[f'{speed}']['RB'])

    df = df._append(data, ignore_index = True)
    # data = pd.DataFrame(data)
    # df = pd.concat([df,data], ignore_index = True, axis =1)
    

    RG = xi[1]
    RB = xi[2]
    dfnRG = pd.DataFrame({ f'{i}' : RG})
    dfnRB = pd.DataFrame({ f'{i}' : RB})
    dfresRG = pd.concat([dfresRG,dfnRG], ignore_index=True, axis=1)
    dfresRB = pd.concat([dfresRB,dfnRB], ignore_index=True, axis=1)




print(sys.getsizeof(df))

home = os.path.expanduser("~")
filepath = f'../output/{net_name}'
if not os.path.isdir(filepath):
    os.makedirs(filepath)

df.to_csv(f'{filepath}/anticipation_data_wAB.csv')
dfresRG.to_csv(f'{filepath}/responses_RG_wAB.csv')
dfresRB.to_csv(f'{filepath}/responses_RB_wAB.csv')

stop = time.time()
params = X[-1][-1]

with open(f'{filepath}/params_grid_wAB', 'wb') as handle:
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

