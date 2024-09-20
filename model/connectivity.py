import numpy as np
import matplotlib.pyplot as plt

from utils import gaussian,DOG



class connectivity(object):

    def __init__(self,
                 params,
                 filepath = None):

        self.params = params

        self.nb_cells = params['nb_cells']
        self.pos_rf_mid = params['pos_rf_mid']
        self.dt = params['dt']
        self.duration = params['duration']
        self.time = np.arange(0,self.duration,self.dt)
        self.nb_GC_cells = params['nb_GC_cells']

        self.w = params['w_GC']

        self.distance = params['distance']
        self.speed = params['speed']
       
        self.filepath = filepath

        self.W = None


        self.pos_rf_GC_mid = np.linspace(0,self.distance,self.nb_GC_cells+2) #mm
        self.pos_rf_GC_mid = self.pos_rf_GC_mid[1:-1] #mm
        self.std_GC = params['std_GC']
        self.std_GC_surround = params['std_GC_s']
        self.tps_rf_GC_mid = self.pos_rf_GC_mid/self.speed #s


    def weight_matrix_i_to_i(self,tau,nb_cells):
        
        # make weight matrix
        W = np.zeros((nb_cells,nb_cells))

        # fill diagonal
        np.fill_diagonal(W, (np.ones(nb_cells)*tau))

        return W
    

   
    

    def weight_matrix_i_to_nn(self,w,nb_cells):
        
        # make weight matrix
        W = np.zeros((nb_cells,nb_cells))

    
        # fill weights

        # backward connection
        # from amacrine to bipolar
        ib = np.arange(0,nb_cells-1,1).astype(int)
        ja = ib+1

        W[ib,ja] = w


        # forward connection
        # from amacrine to bipolar
        ib = np.arange(1,nb_cells,1).astype(int)
        ja = ib-1

        W[ib,ja] = w



        return W
    
    def weight_matrix_i_to_nnplusd(self,w,nb_cells,d = 2):
        
        # make weight matrix
        W = np.zeros((nb_cells,nb_cells))

    
        # fill weights

        # backward connection
        # from amacrine to bipolar
        ib = np.arange(0,nb_cells-d,1).astype(int)
        ja = ib+d

        W[ib,ja] = w

        # forward connection
        # from amacrine to bipolar
        ib = np.arange(d,nb_cells,1).astype(int)
        ja = ib-d

        W[ib,ja] = w

        return W
    


    def weight_matrix_pooling(self,weight):
        
        # make weight matrix
        W_pooling = np.zeros((self.nb_GC_cells,self.nb_cells))

        # gaussian weighting 
        for i in range(self.nb_GC_cells):

            W_pooling[i,:] = weight * DOG(self.pos_rf_mid,self.pos_rf_GC_mid[i],self.std_GC,self.std_GC_surround,self.w)


        return W_pooling



    def assemble_matrix_IPL(self,W):


        self.L =np.concatenate(np.array([np.concatenate(ti,axis = 1) for ti in W]),axis = 0)

        return self.L


    def plot_weight_matrix_IPL(self):

        fig = plt.figure()
        plt.imshow(self.L)
        plt.colorbar()
        plt.title('Weight Matrix')
        plt.xlabel('sending')
        plt.ylabel('recieving')

        #TODO add labels 

        fig.savefig(f'{self.filepath}/plots/weights_IPL.png')




    def plot_weight_matrix_pooling(self,W_pooling):

        fig = plt.figure()
        plt.imshow(W_pooling)
        plt.colorbar()
        plt.title('Weight Matrix')
        plt.xlabel('BC')
        plt.ylabel('GC')

        #TODO add labels 

        fig.savefig(f'{self.filepath}/plots/weights_pooling.png')



    def get_eig(self):

        [lam,P] = np.linalg.eig(self.L)
        resonance_freq = lam[0].imag/(2*np.pi)
        # print(f'time constant of the solution = {-1/lam[0].real} s')

        self.params['lam'] = lam
        self.params['tau_res'] = -1/lam[0].real
        self.params['P'] = P
        self.params['resonance_freq'] = resonance_freq

        return self.params


    def add_params(self):

        self.params['pos_rf_GC_mid'] =self.pos_rf_GC_mid
        self.params['tps_rf_GC_mid'] = self.tps_rf_GC_mid
        self.params['std_GC'] = self.std_GC
       

        return self.params
    




