import gym
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.widgets import Cursor
from matplotlib.colors import cnames
from scipy.integrate import ode
from time import time, sleep
from threading import Thread
from multiprocessing import Process, Pipe, Event
import default_params
color_generator = iter(cnames.items())

class Plant(gym.Env):

    def __init__(self,plant_params=None, name='Plant'):


        self.noise_dist = plant_params['noise_dist']
        self.angle_dims =plant_params['angle_dims']        
        self.dt = plant_params['dt']       
            
        self.name = name
        self.state = None
        self.u = None
        self.t = 0
        self.done = False
        self.renderer = None

    def apply_control(self, u):
        self.u = np.array(u, dtype=np.float64)

        if len(self.u.shape) < 2:
            self.u = self.u[:, None]

    def get_state(self, noisy=True):
        state = self.state

        if noisy and self.noise_dist is not None:
            # noisy state measurement
            state += self.noise_dist.sample(1).flatten()


        return state.flatten(), self.t

    def set_state(self, state):
        self.state = state

    def stop(self):
      
        self.close()


class ODEPlant(Plant):
    def __init__(self,plant_params, name='ODEPlant', integrator='dopri5'):
        super(ODEPlant, self).__init__(name,plant_params)
        
        integrator = 'dopri5'
        atol=1e-12
        rtol=1e-12
        
        # initialize ode solver
        self.solver = ode(self.dynamics).set_integrator(integrator,
                                                        atol=atol,
                                                        rtol=rtol)
        

    def set_state(self, state):
        if self.state is None or\
           np.linalg.norm(np.array(state)-np.array(self.state)) > 1e-12:
            # float64 required for the ode integrator
            self.state = np.array(state, dtype=np.float64).flatten()
        # set solver internal state
        self.solver = self.solver.set_initial_value(self.state)
        # get time from solver
        self.t = self.solver.t

    def step(self, action):
        self.apply_control(action)
        
        t1 = self.solver.t + self.dt
        while self.solver.successful and self.solver.t < t1:
            self.solver.integrate(t1)
        self.state = np.array(self.solver.y)
        self.t = self.solver.t


        state, t = self.get_state()
 
        cost = self.get_cost(state)
        if np.abs(state[0])>10:
            done=True
        else:
            done=False
        return state, cost, done, dict(t=t)
    
    def get_cost(self,state):
        # cost = 1-np.exp(-(np.power(state[0]-self.plant_params['link1_length']*np.sin(state[4])
        #                            -self.plant_params['link2_length']*np.sin(state[5]),2)
        #                   +np.power(self.plant_params['link1_length']+self.plant_params['link2_length']
        #                             -self.plant_params['link1_length']*np.cos(state[4])-self.plant_params['link2_length']*np.cos(state[5]),2))/2/.25)
        cost=1-np.exp(-(np.power(self.state[0]+self.plant_params['pole_length']*np.sin(self.state[3]),2)
                          +np.power(self.plant_params['pole_length']+self.plant_params['pole_length']*np.cos(self.state[3]),2))/2/self.plant_params['cw']**2)

        return cost



class Cartpole(ODEPlant):

    metadata = {
        'render.modes': ['human']
    }

    def __init__(self,plant_params,name='DoubleCartpole'):
        super().__init__(name,plant_params)
        # cartpole system parameters
        self.l = plant_params['pole_length']
        self.m = plant_params['pole_mass']
        self.M = plant_params['cart_mass']
        self.b = plant_params['friction']
        self.g = plant_params['gravity']
        self.plant_params=plant_params
        # initial state
        self.state0_dist = plant_params['s0_dist']


        # pointer to the class that will draw the state of the carpotle system
        self.renderer = None



    def dynamics(self, t, z):
        l, m, M, b, g = self.l, self.m, self.M, self.b, self.g
        f = self.u if self.u is not None else np.array([0])

        sz, cz = np.sin(z[3]), np.cos(z[3])
        cz2 = cz*cz
        a0 = m*l*z[2]*z[2]*sz
        a1 = g*sz
        a2 = f[0] - b*z[1]
        a3 = 4*(M+m) - 3*m*cz2

        dz = np.zeros((4, 1))
        dz[0] = z[1]                                      # x
        dz[1] = (2*a0 + 3*m*a1*cz + 4*a2)/a3              # dx/dt
        dz[2] = -3*(a0*cz + 2*((M+m)*a1 + a2*cz))/(l*a3)  # dtheta/dt
        dz[3] = z[2]                                      # theta

        return dz

    def reset(self):
        state0 = self.state0_dist()
        self.set_state(state0)
        return self.state

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


    def state_change(self,state):
        state_train=np.hstack([state[3],state[2],state[0],state[1]])
        return state_train



if __name__ =="__main__":
    Eparams=default_params.cartpole_params()
    env=Cartpole(Eparams)





