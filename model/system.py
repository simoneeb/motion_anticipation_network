
import numpy as np


class system(object):



    def __init__(self, 
             params,
             W_GG,
             W_ActG,
             W_GtoActG
             ):
        
        self.Layers_IPL = list()


        self.params = params
        self.nb_cells = params['nb_cells']
        self.nb_GC_cells = params['nb_GC_cells']
        self.pos_rf_GC_mid = params['pos_rf_GC_mid'] 
        self.tps_rf_GC_mid = params['tps_rf_GC_mid'] 
        self.speed = params['speed'] 
        self.W_GG =W_GG
        self.W_ActG =W_ActG
        self.W_GtoActG =W_GtoActG

        self.plastic_to_G = params['plastic_to_G']
        self.plastic_to_A = params['plastic_to_A']

        self.tps = params['tps']
        self.X0 = np.ones(self.nb_cells) * params['X0']
        self.G0 = np.ones(self.nb_GC_cells) * params['X0']
        self.A0 = np.ones(self.nb_cells) * params['X0']
        self.n0 = np.ones(self.nb_cells)

        self.dt = params['dt']
        self.duration = params['duration']
        self.time = np.arange(0,self.duration,self.dt)

        self.layer = dict()


    def create_layer(self,W_connectivity,
                     W_intra_Act,W_inter_Act,
                     W_krec,W_krel,
                     W_out,
                     rectification,
                     F):



        layer = {
                 'W':  W_connectivity,
                 'WA': [W_intra_Act,W_inter_Act],
                 'Wn': [W_krec,W_krel],
                 'W_out': [W_out],
                 
                 'X': np.zeros((self.nb_cells, self.tps)),                 
                 'X_rect': np.zeros((self.nb_cells, self.tps)),                 
                 'X_rect_n': np.zeros((self.nb_cells, self.tps)),                 
                 'A': np.zeros((self.nb_cells, self.tps)),                 
                 'G': np.ones((self.nb_cells, self.tps)),                 
                 'n': np.zeros((self.nb_cells, self.tps)), 

                 'dX' : np.zeros((self.nb_cells, self.tps)),   
                 'dA' : np.zeros((self.nb_cells, self.tps)),   
                 'dn' : np.zeros((self.nb_cells, self.tps)),

                 'rectification' : rectification,

                 'F' : F
                 }
        
        self.Layers_IPL.append(layer)
        

    def solve_IPL_GainControl_Plasticity(self,GainF,N):

        nb_layers = len(self.Layers_IPL)

        for layer in self.Layers_IPL:     # set inital condition for all cells 

            layer['X'][:,0] = self.X0      
            layer['A'][:,0] = self.A0
            layer['n'][:,0] = self.n0

        for t in range(1,self.tps-1):   # loop over timesteps 

            for l in range(nb_layers):  # Rectification


                if self.Layers_IPL[l]['rectification'] is True :                                                                 # if rectification is True, copy variable and rectify previous time step 
                    self.Layers_IPL[l]['X_rect'][:,t-1] = self.Layers_IPL[l]['X'][:,t-1].copy()                                        
                    self.Layers_IPL[l]['X_rect_n'][:,t-1] = self.Layers_IPL[l]['X'][:,t-1].copy()

                    for i in range(self.nb_cells):
                        self.Layers_IPL[l]['X_rect'][i,t-1] = N(self.Layers_IPL[l]['X_rect'][i,t-1],self.params,'BC') 
                        self.Layers_IPL[l]['X_rect_n'][i,t-1] = N(self.Layers_IPL[l]['X_rect_n'][i,t-1],self.params,'n') 

                else : 
                    self.Layers_IPL[l]['X_rect'][:,t-1] = self.Layers_IPL[l]['X'][:,t-1].copy()
                    self.Layers_IPL[l]['X_rect_n'][:,t-1] = self.Layers_IPL[l]['X'][:,t-1].copy()

                    for i in range(self.nb_cells):
                        self.Layers_IPL[l]['X_rect_n'][i,t-1] = N(self.Layers_IPL[l]['X_rect_n'][i,t-1],self.params,'n')            # if rectification is False, rectify n anyway





            for l in range(nb_layers): # Gradient 

                
                # add decay to gradient for plasticity, activity and voltage
                self.Layers_IPL[l]['dn'][:,t-1] = np.dot(self.Layers_IPL[l]['Wn'][0],(1-self.Layers_IPL[l]['n'][:,t-1])) - np.dot(self.Layers_IPL[l]['Wn'][1],self.Layers_IPL[l]['X_rect_n'][:,t-1])*self.Layers_IPL[l]['n'][:,t-1]
                self.Layers_IPL[l]['dA'][:,t-1] = np.dot(self.Layers_IPL[l]['WA'][0],self.Layers_IPL[l]['A'][:,t-1]) + np.dot(self.Layers_IPL[l]['WA'][1],self.Layers_IPL[l]['X_rect_n'][:,t-1])
                self.Layers_IPL[l]['dX'][:,t-1] += self.Layers_IPL[l]['F'][:,t-1]    


                for x in range(nb_layers):                                    # include input from other layers 

                    for i in range(self.nb_cells):                            # calculate Gain from input layer x based on activity at time t-1
                        self.Layers_IPL[x]['G'][i,t-1] = GainF(self.Layers_IPL[x]['A'][i,t-1])

                    if x == l :     # if the same layer, add decay 
                        self.Layers_IPL[l]['dX'][:,t-1] += np.dot(self.Layers_IPL[l]['W'][x],self.Layers_IPL[x]['X'][:,t-1]) 
                    
                    else :        # if other layer, add input scaled by plasticty and gain 
                        self.Layers_IPL[l]['dX'][:,t-1] += np.dot(self.Layers_IPL[l]['W'][x],self.Layers_IPL[x]['X_rect'][:,t-1]) * self.Layers_IPL[x]['n'][:,t-1] * self.Layers_IPL[x]['G'][:,t-1]
                        
                # update state vector at time t
                self.Layers_IPL[l]['n'][:,t] =  self.Layers_IPL[l]['n'][:,t-1]+ self.Layers_IPL[l]['dn'][:,t-1]*self.dt
                self.Layers_IPL[l]['X'][:,t] =  self.Layers_IPL[l]['X'][:,t-1]+ self.Layers_IPL[l]['dX'][:,t-1]*self.dt
                self.Layers_IPL[l]['A'][:,t] =  self.Layers_IPL[l]['A'][:,t-1]+ self.Layers_IPL[l]['dA'][:,t-1]*self.dt
        

    

    def solve_GC(self,N, rectification = False):

        self.G = np.zeros((self.nb_GC_cells,self.tps))       # initialize state vector 
        self.G[:,0] = self.G0                                # set inital condition
        
        self.G_rect = np.zeros((self.nb_GC_cells,self.tps))  # rectified state vector

        dG = np.zeros((self.nb_GC_cells,self.tps))           # gradient 

        self.AG = np.zeros((self.nb_GC_cells,self.tps))      # activity vector
        self.AG[:,0] = self.G0                               # set inital condition 

        dAG = np.zeros((self.nb_GC_cells,self.tps))          # gradient

        
        for t in range(1,self.tps-1):

            #TODO see if this can be done better
                    
            for i in range(self.nb_GC_cells):
                self.G_rect[i,t-1] = N(self.G[i,t-1],self.params,'GC')


            dG[:,t-1] = np.dot(self.W_GG,self.G[:,t-1]) 
            dAG[:,t-1] = np.dot(self.W_ActG,self.AG[:,t-1]) + np.dot(self.W_GtoActG,self.G_rect[:,t-1])


            for layer in self.Layers_IPL:
                dG[:,t-1] +=  np.dot(layer['W_out'],layer['X_rect'][:,t-1]*layer['G'][:,t-1]).flatten()

            self.G[:,t] = self.G[:,t-1]+ dG[:,t-1]*self.dt
            self.AG[:,t] = self.AG[:,t-1]+ dAG[:,t-1]*self.dt


        return self.G,self.AG,self.G_rect



    def rectify(self,N,GainF):
        self.GG = np.array([np.array([GainF(x) for x in ai]) for ai in self.AG])
        self.RG = np.array([np.array([self.G_rect[i,t]*self.GG[i,t] for t,x in enumerate(gi)]) for i,gi in enumerate(self.G)])

        return self.RG, self.GG




    def calculate_anticipation(self):

        ant_time = np.zeros(self.nb_GC_cells)
        ant_space = np.zeros(self.nb_GC_cells)
        ant_time_drive = np.zeros(self.nb_GC_cells)
        ant_space_drive = np.zeros(self.nb_GC_cells)
        for i,GC in enumerate(self.RG):
            drive = self.Layers_IPL[0]['F']
            ant_time_drive[i] = (np.argmax(drive)-np.argmax(GC))*self.dt
            ant_space_drive[i] = ant_time_drive[i] *self.speed
            ant_time[i] = self.tps_rf_GC_mid[i]-np.argmax(GC)*self.dt
            ant_space[i] = ant_time[i] *self.speed


        return[ant_time, ant_space,ant_time_drive,ant_space_drive]




    def PVA(self):

        PVA = np.zeros((self.nb_GC_cells,self.tps))

        for t in range(self.tps):
            if np.sum(self.RG[:,t]) > 0:
               PVA[:,t] =  self.RG[:,t] * self.pos_rf_GC_mid / np.sum(self.RG[:,t]) 
            else:
                PVA[:,t] =  self.RG[:,t] * self.pos_rf_GC_mid 
        return PVA




