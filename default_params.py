
import numpy as np
from utils import  Gaussian

def double_cartpole_params():
    # setup learner parameters
    # initial state mean ( x, dx, dtheta1, dtheta2, theta1, theta2)
    
    plant_params = {}    
    plant_params['s0_m']= np.float32(np.array([0, 0, 0, 0, np.pi, np.pi]))
    plant_params['s0_v']= np.float32(np.eye(len(plant_params['s0_m']))*(1e-3**2))
    p0= Gaussian(plant_params['s0_m'], plant_params['s0_v'])
    
    # plant parameters

    
    plant_params['s_dim']=6
    plant_params['a_dim']=1
    plant_params['dt'] = 0.075
    plant_params['link1_length'] = 0.6
    plant_params['link2_length'] = 0.6
    plant_params['link1_mass'] = 0.5
    plant_params['link2_mass'] = 0.5
    plant_params['cart_mass'] = 0.5
    plant_params['friction'] = 0.1
    plant_params['gravity'] = 9.82
    plant_params['angle_dims'] = 2
    plant_params['s0_dist'] = p0
    plant_params['noise_dist'] = Gaussian(
        np.zeros((p0.dim, )),
        np.eye(p0.dim)*1e-6)
    
    plant_params['cw'] = 0.5
    

    return plant_params


def PRL_double_cartpole_params(plant_params):
    # optimizer params


    PRL_params = {}
    PRL_params['angi']=2       # number of angle in state
    

    # general parameters
    PRL_params['s_dim']=plant_params['s_dim']
    PRL_params['a_dim']=plant_params['a_dim']
    
    PRL_params['horizon']=3
    PRL_params['s0_m'] = plant_params['s0_m']      # init state mean
    PRL_params['s0_v']= plant_params['s0_v']       # init state variance
    PRL_params['max_steps'] = int(PRL_params['horizon']/plant_params['dt'])  # control horizon
    PRL_params['discount'] = 1.0      # discount factor
    PRL_params['max_episodes'] = 50

    # policy parameters

    
    PRL_params['Pin_dim']=PRL_params['s_dim']+PRL_params['angi']
    PRL_params['Pou_dim']=PRL_params['a_dim']
    PRL_params['Phidden_dim']=[400,400]
    PRL_params['Plr']=1e-3     # learning rate for policy training
    PRL_params['Pou_max']=20
    
    
    
    # dynamics model parameters


    PRL_params['Din_dim']=PRL_params['angi']+PRL_params['s_dim']+PRL_params['a_dim']
    PRL_params['Dou_dim']=PRL_params['s_dim']+2*(PRL_params['s_dim']+PRL_params['a_dim'])
    PRL_params['Dhidden_dim']=[800,800,800]     
    PRL_params['Dlr']=1e-3     #learning rate for dynamic training
    PRL_params['Dds']=500      #declay_step for dynamic training
    PRL_params['Ddr']=.5       #declay_rate for dynamic training
    
    
    
    PRL_params['SC_init']=np.float32(np.tile([2.6,0.21,1],[PRL_params['s_dim'],1]).T)
    PRL_params['SClr']=1e-3
    PRL_params['coefficient']=np.array([1,1,7,15,.7,3,.4])     # coefficient for normalizing state when build the append label
    PRL_params['Memory_trans_size']=20*PRL_params['max_steps']
    PRL_params['Memory_prob_size']=4*PRL_params['max_steps']
    
    
    #PRL parameters
    PRL_params['sample_num']=200
    
    
    return PRL_params


def cartpole_params():
    plant_params = {}   
    
    plant_params['s0_m'] = np.float32(np.array([0, 0, 0, 0]))
    plant_params['s0_v'] = np.float32(np.eye(len(plant_params['s0_m']))*(0.2**2))
    p0 = Gaussian(plant_params['s0_m'], plant_params['s0_v'])
    plant_params['s_dim']=4
    plant_params['a_dim']=1
    plant_params['dt'] = 0.1

    plant_params['pole_length'] = 0.6
    plant_params['pole_mass'] = 0.5
    plant_params['cart_mass'] = 0.5
    plant_params['friction'] = 0.1
    plant_params['gravity'] = 9.82

    plant_params['angle_dims'] = 2
    plant_params['s0_dist'] = p0
    plant_params['noise_dist'] = Gaussian(np.zeros((p0.dim, )),
                                            np.eye(p0.dim)*1e-6)
                                        
    
    plant_params['cw'] = 0.25
    return plant_params

def PRL_cartpole_params(plant_params):

    PRL_params = {}
    PRL_params['angi']=1       # number of angle in state
    
    PRL_params['s_dim']=plant_params['s_dim']
    PRL_params['a_dim']=plant_params['a_dim']
    
    PRL_params['horizon']=2.5
    PRL_params['s0_m'] = plant_params['s0_m']      # init state mean
    PRL_params['s0_v']= plant_params['s0_v']       # init state variance
    PRL_params['max_steps'] = int(PRL_params['horizon']/plant_params['dt'])  # control horizon
    PRL_params['discount'] = 1.0      # discount factor
    PRL_params['max_episodes'] = 25

    # policy parameters

    
    PRL_params['Pin_dim']=PRL_params['s_dim']+PRL_params['angi']
    PRL_params['Pou_dim']=PRL_params['a_dim']
    PRL_params['Phidden_dim']=[200,200]
    PRL_params['Plr']=1e-3     # learning rate for policy training
    PRL_params['Pou_max']=10
    
    
    
    # dynamics model parameters


    PRL_params['Din_dim']=PRL_params['angi']+PRL_params['s_dim']+PRL_params['a_dim']
    PRL_params['Dou_dim']=PRL_params['s_dim']+2*(PRL_params['s_dim']+PRL_params['a_dim'])
    PRL_params['Dhidden_dim']=[400,400]     
    PRL_params['Dlr']=1e-3     #learning rate for dynamic training
    PRL_params['Dds']=500      #declay_step for dynamic training
    PRL_params['Ddr']=.5       #declay_rate for dynamic training
    
    
    
    PRL_params['SC_init']=np.float32(np.tile([.3,-2.1,.3],[PRL_params['s_dim'],1]).T)
    PRL_params['SClr']=1e-3
    PRL_params['coefficient']=np.array([1,4,1,1,0.3])    # coefficient for normalizing state when build the append label
    PRL_params['Memory_trans_size']=20*PRL_params['max_steps']
    PRL_params['Memory_prob_size']=4*PRL_params['max_steps']
    
    
    #PRL parameters
    PRL_params['sample_num']=200

    
    
    return PRL_params