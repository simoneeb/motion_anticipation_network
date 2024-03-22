
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


    
    # @property
    # def layer(self):

    #     return self._layer


    # @layer.setter
    # def layer(self,L):

    #     self.layer = L



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

        for layer in self.Layers_IPL:

            layer['X'][:,0] = self.X0
            layer['A'][:,0] = self.A0
            layer['n'][:,0] = self.n0

        # test = np.zeros(self.tps)
        # test2 = np.zeros(self.tps)
        # test3 = np.zeros(self.tps)
        
        for t in range(0,self.tps-1):

            # add extrernal stimulus
            for l in range(nb_layers):

                if self.Layers_IPL[l]['rectification'] is True : 
                    self.Layers_IPL[l]['X_rect'][:,t] = self.Layers_IPL[l]['X'][:,t].copy()
                    self.Layers_IPL[l]['X_rect_n'][:,t] = self.Layers_IPL[l]['X'][:,t].copy()
                    for i in range(self.nb_cells):
                        self.Layers_IPL[l]['X_rect'][i,t] = N(self.Layers_IPL[l]['X_rect'][i,t],self.params,'BC') 
                        self.Layers_IPL[l]['X_rect_n'][i,t] = N(self.Layers_IPL[l]['X_rect_n'][i,t],self.params,'n') 
                else : 
                    self.Layers_IPL[l]['X_rect'][:,t] = self.Layers_IPL[l]['X'][:,t].copy()
                    self.Layers_IPL[l]['X_rect_n'][:,t] = self.Layers_IPL[l]['X'][:,t].copy()
                    for i in range(self.nb_cells):
                        self.Layers_IPL[l]['X_rect_n'][i,t] = N(self.Layers_IPL[l]['X_rect_n'][i,t],self.params,'n') 


            for l in range(nb_layers):

                self.Layers_IPL[l]['dn'][:,t] = np.dot(self.Layers_IPL[l]['Wn'][0],(1-self.Layers_IPL[l]['n'][:,t])) - np.dot(self.Layers_IPL[l]['Wn'][1],self.Layers_IPL[l]['X_rect_n'][:,t])*self.Layers_IPL[l]['n'][:,t]
                self.Layers_IPL[l]['dA'][:,t] = np.dot(self.Layers_IPL[l]['WA'][0],self.Layers_IPL[l]['A'][:,t]) + np.dot(self.Layers_IPL[l]['WA'][1],self.Layers_IPL[l]['X_rect_n'][:,t])
                self.Layers_IPL[l]['dX'][:,t] += self.Layers_IPL[l]['F'][:,t]


                for x in range(nb_layers):

                    # if self.Layers_IPL[x]['rectification'] is True : 
                    #     #X_rect = self.Layers_IPL[x]['X'][:,t].copy()

                    for i in range(self.nb_cells):
                    #         #X_rect[i] = N(X_rect[i],self.params,'BC') 
                        self.Layers_IPL[x]['G'][i,t] = GainF(self.Layers_IPL[x]['A'][i,t])


                    # else : 
                    #     #X_rect = self.Layers_IPL[x]['X'][:,t].copy()
                    #     for i in range(self.nb_cells):
                    #         Gain[i] = GainF(self.Layers_IPL[x]['A'][i,t])

                    if x == l :
                        self.Layers_IPL[l]['dX'][:,t] += np.dot(self.Layers_IPL[l]['W'][x],self.Layers_IPL[x]['X'][:,t]) 
                        # if l ==0:
                        #     test2[t]=np.dot(self.Layers_IPL[l]['W'][x],self.Layers_IPL[x]['X'][:,t])[300] *self.dt

                    else :
                        if self.plastic_to_A is False and x == 0:
                            self.Layers_IPL[l]['dX'][:,t] += np.dot(self.Layers_IPL[l]['W'][x],self.Layers_IPL[x]['X_rect'][:,t])* self.Layers_IPL[x]['G'][:,t]
                        else:
                            if t ==0:
                                print(f' Layer {l}, is plastic ')
                            self.Layers_IPL[l]['dX'][:,t] += np.dot(self.Layers_IPL[l]['W'][x],self.Layers_IPL[x]['X_rect'][:,t])  * self.Layers_IPL[x]['n'][:,t] * self.Layers_IPL[x]['G'][:,t]

                        # if l ==0 : 
                        #     test[t]=np.array(np.dot(self.Layers_IPL[l]['W'][x],self.Layers_IPL[x]['X_rect'][:,t]) )[300]*self.dt
                        #     test3[t]=self.Layers_IPL[x]['X_rect'][300,t]

           
                self.Layers_IPL[l]['n'][:,t+1] =  self.Layers_IPL[l]['n'][:,t]+ self.Layers_IPL[l]['dn'][:,t]*self.dt
                self.Layers_IPL[l]['X'][:,t+1] =  self.Layers_IPL[l]['X'][:,t]+ self.Layers_IPL[l]['dX'][:,t]*self.dt
                self.Layers_IPL[l]['A'][:,t+1] =  self.Layers_IPL[l]['A'][:,t]+ self.Layers_IPL[l]['dA'][:,t]*self.dt
        

    

    def solve_GC(self,N, rectification= False):

        self.G = np.zeros((self.nb_GC_cells,self.tps))
        self.G[:,0] = self.G0        
        
        self.G_rect = np.zeros((self.nb_GC_cells,self.tps))

        dG = np.zeros((self.nb_GC_cells,self.tps))

        self.AG = np.zeros((self.nb_GC_cells,self.tps))
        self.AG[:,0] = self.G0

        dAG = np.zeros((self.nb_GC_cells,self.tps))

        #t = 0
        
        #while t < simt:
        for t in range(0,self.tps-1):

            #TODO see if this can be done better
                    
            
            #self.G_rect[:,t] = self.G[:,t].copy()
            for i in range(self.nb_GC_cells):
                self.G_rect[i,t] = N(self.G[i,t],self.params,'GC')


            dG[:,t] = np.dot(self.W_GG,self.G[:,t]) 
            dAG[:,t] = np.dot(self.W_ActG,self.AG[:,t]) + np.dot(self.W_GtoActG,self.G_rect[:,t])



            for layer in self.Layers_IPL:

                # if layer['rectification'] is True:
                #     X_rect = layer['X'][:,t].copy()
                #     for i in range(self.nb_cells):
                #         X_rect[i] = N(X_rect[i],self.params,'BC') 
                # else: 
                #     X_rect = layer['X'][:,t].copy()
                    

                # Gain = np.zeros(self.nb_cells)
                # for i in  range(self.nb_cells):
                #         Gain[i] = GainF(layer['A'][i,t])
                if self.plastic_to_G is True:
                    dG[:,t] +=  np.dot(layer['W_out'],layer['X_rect'][:,t]*layer['n'][:,t]*layer['G'][:,t]).flatten()
                else:
                    dG[:,t] +=  np.dot(layer['W_out'],layer['X_rect'][:,t]*layer['G'][:,t]).flatten()

            self.G[:,t+1] = self.G[:,t]+ dG[:,t]*self.dt
            self.AG[:,t+1] = self.AG[:,t]+ dAG[:,t]*self.dt

  
    
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






    # # SYNAPSE DYNAMICS
    # def solve_IPL(self):

    #     self.X = np.zeros((self.nb_cells*2,self.tps))
    #     self.X[:,0] = self.X0
    #     #t = 0
        
        
    #     for t in range(0,self.tps-1):

    #         #dP = np.dot(P[:,t].transpose(), W) + F(t,params)
      
    #         dX = np.dot(self.W,self.X[:,t]) + self.F[:,t]
    #         self.X[:,t+1] = self.X[:,t]+ dX*self.dt


        
    #     return self.X


    # def solve_IPL_GainControl(self,N):

    #     self.X = np.zeros((self.nb_cells*2,self.tps))
    #     self.X[:,0] = self.X0

    #     self.A = np.zeros((self.nb_cells*2,self.tps))
    #     self.A[:,0] = self.A0
    #     #t = 0
        
        
    #     for t in range(0,self.tps-1):

    #         #dP = np.dot(P[:,t].transpose(), W) + F(t,params)

    
    #         X = self.X[:,t].copy()  

    #         for i in range(self.nb_cells*2):
    #             X[i] = N(X[i],self.params,'BC')
      
    #         dX = np.dot(self.W,X[:,t]) + self.F[:,t]
    #         dA = np.dot(self.W_Act,self.A[:,t]) + np.dot(self.W_BtoAct,X)
    #         self.A[:,t+1] = self.A[:,t]+ dA*self.dt
    #         self.X[:,t+1] = self.X[:,t]+ dX*self.dt


    #     return self.X,self.A