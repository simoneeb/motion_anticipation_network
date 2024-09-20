import numpy as np
import matplotlib.pyplot as plt



class plotting(object):


    def __init__(self,
                 params,
                 out,
                 stats = None,
                 filepath = None,
                 figsize =  (16,12)):
 
        #(params.keys())
        self.filepath = f'{filepath}/plots'
        self.figsize = figsize
        self.nb_cells = params['nb_cells']
        self.nb_GC_cells = params['nb_GC_cells']
        self.dt = params['dt']
        self.duration = params['duration']
        self.time = np.arange(0,self.duration,self.dt)    
        self.tps_rf_mid = params['tps_rf_mid']
        self.tps_rf_GC_mid = params['tps_rf_GC_mid']
        self.pos_rf_GC_mid = params['pos_rf_GC_mid']
        self.tps = params['tps']
        self.speed = params['speed']
        self.out = out
        self.stats = stats


    def plot_all_BC_responses(self,layer, response = 'RB'):

        # plot all BC responses
        fig,ax = plt.subplots(2,1,sharex =True, figsize = self.figsize)
        for i in range(0, self.nb_cells,1):
            if response == 'VB':
                item = ax[0].plot(self.out['res'][layer]['X'][i], label = f'BC {i}', alpha = 0.5)
            else: 
                item = ax[0].plot(self.out['RB'][i], label = f'BC {i}', alpha = 0.5)
            ax[0].axvline(self.tps_rf_mid[i]/self.dt,linestyle = ':', color = item[0].get_color(), alpha = 0.5)


        ax[0].set_xlabel('timesteps')
        ax[0].set_ylabel('V(t)')
        ax[0].set_title('BC Responses')

        fig.legend()
        ax[1].matshow(self.out['inp'], aspect = 'auto')
        ax[1].set_ylabel('cells')
        ax[1].set_title('Stimulus')

        fig.savefig(f'{ self.filepath}/resps_BC.png')


        

    def plot_all_AC_responses(self,layer):

    # plot all AC responses
        fig,ax = plt.subplots(2,1, sharex = True, figsize =  self.figsize)
        for i in range(0,self.nb_cells,1):
            item = ax[0].plot(self.out['res'][layer]['X'][i], label = f'AC {i}', alpha = 0.5)
            ax[0].axvline(self.tps_rf_mid[i]/self.dt,linestyle = ':', color = item[0].get_color(), alpha = 0.5)


        ax[0].set_xlabel('timesteps')
        ax[0].set_ylabel('V(t)')
        ax[0].set_title('AC Responses')

        fig.legend()
        ax[1].matshow(self.out['inp'], aspect = 'auto')
        ax[1].set_ylabel('cells')
        ax[1].set_title('Stimulus')

        fig.savefig(f'{self.filepath}/resps_AC.png')




    def plot_all_GC_responses(self,title = 'Pooled Response'):
        fig,ax = plt.subplots(3,1, sharex = True, figsize =self.figsize)

        ax[0].matshow(self.out['inp'], aspect = 'auto')
        ax[0].set_ylabel('cells')


        for i in range(0,self.nb_GC_cells,1):


            item = ax[1].plot(self.out['VG'][i], label = f'GC {i}')
            ax[2].plot(self.out['RG'][i], color = item[0].get_color())

            ax[1].axvline(self.tps_rf_GC_mid[i]/self.dt,linestyle = ':', alpha = 0.5, color = item[0].get_color())
            ax[2].axvline(self.tps_rf_GC_mid[i]/self.dt,linestyle = ':', alpha = 0.5, color = item[0].get_color())


        ax[0].set_title('Stimulus into Bipolars')
        ax[1].set_title('Galglion Voltage')
        ax[2].set_title('Ganglion Firing Rate')

        ax[2].set_xlabel('timesteps')

        ax[1].set_ylabel('V(t)')
        ax[2].set_ylabel('R(t)')


        fig.suptitle(title)

        fig.legend()
        fig.savefig(f'{self.filepath}/resps_GC.png')



    def plot_one_BC(self,layer,CELL,ax, label, alpha = 0.5,linewidth = 2,response = 'RB', color = 'k', middlecrossing_at_0 = False):

         # plot all BC responses
        if response == 'VB':

            r = self.out['res'][layer]['X'][CELL]
            anti_idx = r.argmax()
            anti_tp = self.time[anti_idx]
            anti = self.tps_rf_mid[CELL] -anti_tp
            maxi = r.max()
        else: 
            r =  self.out[response][CELL]  
            anti_idx = r.argmax()
            anti_tp = self.time[anti_idx]
            anti = self.tps_rf_mid[CELL] - anti_tp
            maxi = r.max()

        if middlecrossing_at_0 : 
                item = ax.plot((self.time-self.tps_rf_mid[CELL])*1000,r, label = label, alpha = alpha, color = color, linewidth = linewidth)
                ax.axvline(0,linestyle = ':', color = item[0].get_color(), alpha = alpha, linewidth = linewidth)

        else: 
                item = ax.plot(self.time*1000,r, label = label, alpha = alpha, color = color, linewidth = linewidth)
                ax.axvline(self.tps_rf_mid[CELL]*1000,linestyle = ':', color = item[0].get_color(), alpha = alpha, linewidth = linewidth)


        return anti,maxi




    def plot_one_AC(self,layer,CELL, ax, label, alpha = 0.5,color = 'k', middlecrossing_at_0 = False):


        r = self.out['res'][layer]['X'][CELL]
        anti_idx = r.argmax()
        anti_tp = self.time[anti_idx]
        anti = self.tps_rf_mid[CELL] -anti_tp
        maxi = r.max()
        
        if middlecrossing_at_0 : 
                item = ax.plot((self.time-self.tps_rf_mid[CELL])*1000,r, label = label, alpha = alpha, color = color)
                ax.axvline(0,linestyle = ':', color = item[0].get_color(), alpha = alpha)

        else: 
                item = ax.plot(self.time*1000,r, label = label, alpha = alpha, color = color)
                ax.axvline(self.tps_rf_mid[CELL]*1000,linestyle = ':', color = item[0].get_color(), alpha = alpha)

        return anti,maxi


    def plot_one_GC(self,CELL, ax,label = '',y = 'time',response = 'RG', alpha = 1, linestyle = '-', color = 'k', linewidth = 3,middlecrossing_at_0 = False):

        if y == 'time':
            if middlecrossing_at_0 == True:
                item = ax.plot((self.time - self.tps_rf_GC_mid[CELL])*1000,self.out[response], label = label, alpha = alpha, color = color, linestyle = linestyle, linewidth = linewidth)
                ax.axvline(0,linestyle = ':', color = item[0].get_color(), alpha = 1, linewidth = linewidth) 

            else:
                item = ax.plot(self.time*1000 ,self.out[response], label = label, alpha = alpha, color = color, linestyle = linestyle, linewidth = linewidth)
                ax.axvline(self.tps_rf_GC_mid[CELL]*1000,linestyle = ':', color = item[0].get_color(), alpha = 1, linewidth = linewidth) 


        if y == 'space':
            if middlecrossing_at_0 == True:
                item = ax.plot(np.asarray(self.time*self.speed) - self.pos_rf_GC_mid[CELL],self.out[response], label = label, alpha = alpha, color = color, linestyle = linestyle, linewidth = linewidth)
                ax.axvline(0,linestyle = ':', color = item[0].get_color(), alpha = 1, linewidth = linewidth) 

            else:
                item = ax.plot(np.asarray(self.time*self.speed) ,self.out[response], label = label, alpha = alpha, color = color, linestyle = linestyle, linewidth = linewidth)
                ax.axvline(self.pos_rf_GC_mid[CELL],linestyle = ':', color = item[0].get_color(), alpha = 1, linewidth = linewidth) 


        if y == 'neural image':

            res = np.flip(self.out[response])
            pos = (self.time*self.speed) - self.pos_rf_GC_mid[CELL]

            item = ax.plot(pos,res, label = label, alpha = alpha, linestyle = linestyle,color = color, linewidth = linewidth)
            ax.axvline(pos[res.argmax()],linestyle = linestyle, color = item[0].get_color(), alpha = alpha, linewidth = linewidth ) 

    

    
    def plot_one_stim(self,CELL,ax):

        ax.plot(self.time*1000,self.out['inp'][CELL,:], label = f'speed {self.speed}')

    def plot_mean_GC(self, ax,label = '',y = 'time',response = 'RG', alpha = 1, linestyle = '-', color = 'k', linewidth = 3,middlecrossing_at_0 = False):

        l = len(self.time)

        resps = []
        for i in range(self.nb_GC_cells):
            if i >=10 and i <=90:
                mid = int(self.tps_rf_GC_mid[i]/self.dt)+l
                r = np.asarray(self.out[response][i])
                r0 = np.concatenate((np.zeros(l),r,np.zeros(l)))
                resps.append(r0[int(mid-l/2):int(mid+l/2)])


        resps_mean = np.mean(resps, axis = 0)
        resps_std = np.std(resps, axis = 0)

        if y == 'time':
            if middlecrossing_at_0 == True:
                item = ax.plot(self.time - self.time[int(l/2)],resps_mean, label = label, alpha = alpha, color = color, linestyle = linestyle, linewidth = linewidth)
                ax.axvline(0,linestyle = ':', color = item[0].get_color(), alpha = 1, linewidth = linewidth) 
                ax.fill_between(self.time - self.time[int(l/2)],resps_mean-resps_std,resps_mean+resps_std, color = item[0].get_color(), alpha = 0.4, linewidth = linewidth) 

            else:
                item = ax.plot(self.time ,resps_mean, label = label, alpha = alpha, color = color, linestyle = linestyle, linewidth = linewidth)
                ax.axvline(self.time[int(l/2)],linestyle = ':', color = item[0].get_color(), alpha = 1, linewidth = linewidth) 
                ax.fill_between(self.time ,resps_mean-resps_std,resps_mean+resps_std, color = item[0].get_color(), alpha = 0.4, linewidth = linewidth) 


        if y == 'space':
            if middlecrossing_at_0 == True:
                item = ax.plot((self.time - self.time[int(l/2)])*self.speed,resps_mean, label = label, alpha = alpha, color = color, linestyle = linestyle, linewidth = linewidth)
                ax.axvline(0,linestyle = ':', color = item[0].get_color(), alpha = 1, linewidth = linewidth) 
                ax.fill_between((self.time - self.time[int(l/2)])*self.speed,resps_mean-resps_std,resps_mean+resps_std, color = item[0].get_color(), alpha = 0.4, linewidth = linewidth) 

            else:
                item = ax.plot(self.time*self.speed ,resps_mean, label = label, alpha = alpha, color = color, linestyle = linestyle, linewidth = linewidth)
                ax.axvline(self.time[int(l/2)]*self.speed,linestyle = ':', color = item[0].get_color(), alpha = 1, linewidth = linewidth) 
                ax.fill_between(self.time*self.speed,resps_mean-resps_std,resps_mean+resps_std, color = item[0].get_color(), alpha = 0.4, linewidth = linewidth) 


        # if y == 'neural image':

        #     res = np.flip(self.out[response][CELL])
        #     pos = (self.time*self.speed) - self.pos_rf_GC_mid[CELL]

        #     item = ax.plot(pos,res, label = label, alpha = alpha, linestyle = linestyle,color = color, linewidth = linewidth)
        #     ax.axvline(pos[res.argmax()],linestyle = linestyle, color = item[0].get_color(), alpha = alpha, linewidth = linewidth ) 
        # # axs[1].plot(self.out['RG'][CELL], alpha = 0.5, color = item[0].get_color())
        # axs[1].axvline(self.tps_rf_GC_mid[CELL]/self.dt,linestyle = ':', color = item[0].get_color(), alpha = 0.5)



    def plot_PVA_mean(self, ax,label = '', alpha = 1, linestyle = '-', color = 'k', linewidth = 3,middlecrossing_at_0 = False):

        # for every bar posititon, take the PVA at this timepoint, calculate mean and std 
        l = len(self.time)
        resps = []
        maxis = []
        for i in range(self.nb_GC_cells):
            if i >=10 and i <=90:
                t = int(self.tps_rf_GC_mid[i]/self.dt)
                r = np.asarray(self.out['PVA'][:,t])
                r0 = np.concatenate((np.zeros(self.nb_GC_cells),r,np.zeros(self.nb_GC_cells)))
                mid = self.pos_rf_GC_mid[i]
                resp = r0[int(i+self.nb_GC_cells/2):int(i+self.nb_GC_cells*1.5)]
                resps.append(np.flip(resp))
                maxis.append(np.argmax(np.flip(resp)))

        maxis = np.array(self.pos_rf_GC_mid[maxis])
        antis =  self.pos_rf_GC_mid[10:91] -maxis
        resps_mean = np.mean(resps, axis = 0)
        resps_std = np.std(resps, axis = 0)
        
        antis_mean = np.mean(antis, axis = 0)
        antis_std = np.std(antis, axis = 0)

        if middlecrossing_at_0 == True:
            item = ax.plot(self.pos_rf_GC_mid,resps_mean, label = label, alpha = alpha, color = color, linestyle = linestyle, linewidth = linewidth)
            ax.axvline(0,linestyle = ':', color = item[0].get_color(), alpha = 1, linewidth = linewidth) 
            ax.fill_between(self.pos_rf_GC_mid[int(self.pos_rf_GC_mid)],resps_mean-resps_std,resps_mean+resps_std, color = item[0].get_color(), alpha = 0.4, linewidth = linewidth) 

        else:
            item = ax.plot(self.pos_rf_GC_mid ,resps_mean, label = label, alpha = alpha, color = color, linestyle = linestyle, linewidth = linewidth)
            ax.axvline(self.pos_rf_GC_mid[int(self.nb_GC_cells/2)],linestyle = ':', color = item[0].get_color(), alpha = 1, linewidth = linewidth) 
            ax.fill_between(self.pos_rf_GC_mid,resps_mean-resps_std,resps_mean+resps_std, color = item[0].get_color(), alpha = 0.4, linewidth = linewidth) 

        return antis_mean,antis_std




    def plot_stats(self,CELL, ax,label = '',y = 'time', response = 'RG', alpha = 1, linestyle = '-', color = 'k', linewidth = 3,middlecrossing_at_0 = False):

        res = self.stats[response]['mean'][CELL,:,-1]
        std = np.array(self.stats[response][f'std'][CELL,:,-1])/10
    
        if y == 'time':
            if middlecrossing_at_0 == True:
                item = ax.plot(self.time - self.tps_rf_GC_mid[CELL],res, label = label, alpha = alpha, color = color, linestyle = linestyle, linewidth = linewidth)
                ax.axvline(0,linestyle = ':', color = item[0].get_color(), alpha = 1, linewidth = linewidth) 
                #ax.fill_between(self.time - self.tps_rf_GC_mid[CELL],res-std,res+std, color = item[0].get_color(), alpha = 0.4, linewidth = linewidth) 
            else:
                item = ax.plot(self.time, res, label = label, alpha = alpha, color = color, linestyle = linestyle, linewidth = linewidth)
                ax.axvline(self.tps_rf_GC_mid[CELL],linestyle = ':', color = item[0].get_color(), alpha = 1, linewidth = linewidth) 
                #ax.fill_between(self.time, res-std,res+std, color = item[0].get_color(), alpha = 0.4, linewidth = linewidth) 


        if y == 'space':
            if middlecrossing_at_0 == True:
                item = ax.plot(np.asarray(self.time*self.speed) - self.pos_rf_GC_mid[CELL],res, label = label, alpha = alpha, color = color, linestyle = linestyle, linewidth = linewidth)
                ax.axvline(0,linestyle = ':', color = item[0].get_color(), alpha = 1, linewidth = linewidth) 
                #ax.fill_between(np.asarray(self.time*self.speed) - self.pos_rf_GC_mid[CELL],res-std,res+std, color = item[0].get_color(), alpha = 0.4, linewidth = linewidth) 

            else:
                item = ax.plot(np.asarray(self.time*self.speed) ,res, label = label, alpha = alpha, color = color, linestyle = linestyle, linewidth = linewidth)
                ax.axvline(self.pos_rf_GC_mid[CELL],linestyle = ':', color = item[0].get_color(), alpha = 1, linewidth = linewidth) 
                #ax.fill_between(np.asarray(self.time*self.speed),res-std,res+std, color = item[0].get_color(), alpha = 0.4, linewidth = linewidth) 

        return


        # if y == 'neural image':

        #     res = np.flip(self.out[response][CELL])
        #     pos = (self.time*self.speed) - self.pos_rf_GC_mid[CELL]

        #     item = ax.plot(pos,res, label = label, alpha = alpha, linestyle = linestyle,color = color, linewidth = linewidth)
        #     ax.axvline(pos[res.argmax()],linestyle = linestyle, color = item[0].get_color(), alpha = alpha, linewidth = linewidth ) 
        # axs[1].plot(self.out['RG'][CELL], alpha = 0.5, color = item[0].get_color())
    # # big plot with all all responses 
    # fig,ax = plt.subplots(5,1, figsize = (16,24))


    # ax[0].matshow(inp, aspect = 'auto')
    # ax[0].set_ylabel('cells')
    # ax[0].set_title('Stimulus')


    # if stop is not None:
    #     ax[1].axvline(tps_rf_mid[C]/dt, color = 'k', linestyle = ':', alpha =0.3)
    #     ax[1].axvspan((tps_rf_mid[C]-time_to_cross_rf/2)/dt,(tps_rf_mid[C]+time_to_cross_rf/2)/dt, color = 'k', alpha = 0.1)    
    #     ax[2].axvline(tps_rf_mid[C]/dt, color = 'k', linestyle = ':', alpha =0.3)
    #     ax[2].axvspan((tps_rf_mid[C]-time_to_cross_rf/2)/dt,(tps_rf_mid[C]+time_to_cross_rf/2)/dt, color = 'k', alpha = 0.1)    
    #     ax[3].axvline(tps_rf_mid[C]/dt, color = 'k', linestyle = ':', alpha =0.3)
    #     ax[3].axvspan((tps_rf_mid[C]-time_to_cross_rf/2)/dt,(tps_rf_mid[C]+time_to_cross_rf/2)/dt, color = 'k', alpha = 0.1)    
    #     ax[4].axvline(tps_rf_mid[C]/dt, color = 'k', linestyle = ':', alpha =0.3)
    #     ax[4].axvspan((tps_rf_mid[C]-time_to_cross_rf/2)/dt,(tps_rf_mid[C]+time_to_cross_rf/2)/dt, color = 'k', alpha = 0.1)

    # for i in range(0,nb_cells,1):
    #     ax[1].plot(y[i], label = f'BC {i}', alpha = 0.5)

    # for i in range(nb_cells,nb_cells*2,1):
    #     ax[2].plot(y[i], label = f'AC {i}', alpha = 0.5)



    # ax[3].plot(bpsum, label = 'bipolar sum')
    # ax[3].plot(acsum, label = 'amacrine sum')
    # ax[3].plot(bpsum-acsum, label = 'all sum')

    # ax[4].plot([N(x,params) for x in bpsum], label = 'bipolar sum')
    # ax[4].plot([N(x,params) for x in acsum], label = 'amacrine sum')
    # ax[4].plot(np.array([N(x,params) for x in bpsum])-np.array([N(x,params) for x in acsum]), label = 'all sum')


    # ax[1].set_ylabel('V(t)')
    # ax[1].set_title('BC Responses')

    # ax[2].set_ylabel('V(t)')
    # ax[2].set_title('AC Responses')


    # ax[3].set_ylabel('V(t)')
    # ax[3].set_title('Linear Summation')


    # ax[4].set_ylabel('V(t)')
    # ax[4].set_title('Nonlinear Summation')

    # ax[1].legend()
    # ax[2].legend()
    # ax[3].legend()
    # ax[4].legend()
    # ax[-1].set_xlabel('timesteps')
    # fig.savefig(f'{filepath}/resps_all.png')



    # # plot response of one BC cell 

    # # TODO add trace trom other simulation


    # fig,ax = plt.subplots(2,1,sharex = True, figsize = figsize)
    # ax[0].matshow(inp, aspect = 'auto')
    # ax[0].set_ylabel('cells')
    # ax[0].set_title('Stimulus')

    # h = ax[1].plot(y[C*2], label = f'BC # {C*2} interrupted')
    # #ax[1].plot(sol.y[(C-1)*2], label = f'BC # {C-1}')
    # ax[1].plot(y[0], label = f'BC # {0}')
    # ax[1].axvline(tps_rf_mid[C]/dt, color = 'k', linestyle = ':')
    # ax[1].axvspan((tps_rf_mid[C]-time_to_cross_rf/2)/dt,(tps_rf_mid[C]+time_to_cross_rf/2)/dt, color = 'k', alpha = 0.3)
    # ax[1].set_xlabel('timesteps')
    # ax[1].set_ylabel('V(t)')
    # ax[1].legend()
    # fig.savefig(f'{filepath}/resp_cell_{C}.png')


    # # TODO add trace trom other simulation
    # if compare_with : 

    #     # load other sim
    #     with open(compare_with, "rb") as handle:   #Pickling
    #         sol_smooth = pickle.load(handle)

    #     fig,ax = plt.subplots(2,1,sharex = True, figsize = figsize)
    #     ax[0].matshow(inp, aspect = 'auto')
    #     ax[0].set_ylabel('cells')
    #     ax[0].set_title('Stimulus')

    #     h = ax[1].plot(y[C*2], label = f'BC # {C*2} interrupted')
    #     ax[1].plot(sol_smooth.y[C*2], label = f'BC # {C*2} smooth', color = h[0].get_color(), alpha = 0.5)
    #     #ax[1].plot(sol.y[(C-1)*2], label = f'BC # {C-1}')
    #     ax[1].plot(y[0], label = f'BC # {0}')
    #     ax[1].axvline(tps_rf_mid[C]/dt, color = 'k', linestyle = ':')
    #     ax[1].axvspan((tps_rf_mid[C]-time_to_cross_rf/2)/dt,(tps_rf_mid[C]+time_to_cross_rf/2)/dt, color = 'k', alpha = 0.3)
    #     ax[1].set_xlabel('timesteps')
    #     ax[1].set_ylabel('V(t)')
    #     ax[1].legend()
    #     fig.savefig(f'{filepath}/resp_cell_{C}_compare.png')
