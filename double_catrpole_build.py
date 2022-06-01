# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 00:00:25 2022

@author: 98040
"""
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
        cost = 1-np.exp(-(np.power(state[0]-self.plant_params['link1_length']*np.sin(state[4])
                                   -self.plant_params['link2_length']*np.sin(state[5]),2)
                          +np.power(self.plant_params['link1_length']+self.plant_params['link2_length']
                                    -self.plant_params['link1_length']*np.cos(state[4])-self.plant_params['link2_length']*np.cos(state[5]),2))/2/.25)
        return cost



class PlantDraw(object):
    def __init__(self, plant, refresh_period=(1.0/240),
                 name='PlantDraw'):
        super(PlantDraw, self).__init__()
        self.name = name
        self.plant = plant
        self.drawing_thread = None
        self.polling_thread = None

        self.dt = refresh_period
        self.exec_time = time()
        self.scale = 150  # pixels per meter

        self.center_x = 0
        self.center_y = 0
        self.running = Event()

        self.polling_pipe, self.drawing_pipe = Pipe()

    def init_ui(self):
        plt.close(self.name)
        self.fig = plt.figure(self.name)
        self.ax = plt.gca()
        self.ax.set_xlim([-1.5, 1.5])
        self.ax.set_ylim([-1.5, 1.5])
        self.ax.set_aspect('equal', 'datalim')
        self.ax.grid(True)
        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=2)

        # plt.ion()
        plt.show(False)

    def drawing_loop(self, drawing_pipe):
        # start the matplotlib plotting
        self.init_ui()

        while self.running.is_set():
            exec_time = time()
            # get any data from the polling loop
            updts = None
            while drawing_pipe.poll():
                data_from_plant = drawing_pipe.recv()
                if data_from_plant is None:
                    self.running.clear()
                    break

                # get the visuzlization updates from the latest state
                state, t = data_from_plant
                updts = self.update(state, t)
                self.update_canvas(updts)

            # sleep to guarantee the desired frame rate
            exec_time = time() - exec_time
            plt.waitforbuttonpress(max(self.dt-exec_time, 1e-9))
        self.close()

    def close(self):
        # close the matplotlib windows, clean up
        # plt.ioff()
        plt.close(self.fig)

    def update(self):
        plt.figure(self.name)



    def update_canvas(self, updts):
        if updts is not None:
            # update the drawing from the plant state
            self.fig.canvas.restore_region(self.bg)
            for artist in updts:
                self.ax.draw_artist(artist)
            self.fig.canvas.draw()
            # sleep to guarantee the desired frame rate
            exec_time = time() - self.exec_time
            plt.waitforbuttonpress(max(self.dt-exec_time, 1e-9))
        self.exec_time = time()

    def polling_loop(self, polling_pipe):
        current_t = -1
        while self.running.is_set():
            exec_time = time()
            state, t = self.plant.get_state(noisy=False)
            if t != current_t:
                polling_pipe.send((state, t))

            # sleep to guarantee the desired frame rate
            exec_time = time() - exec_time
            sleep(max(self.dt-exec_time, 0))

    def start(self):

        self.drawing_thread = Process(target=self.drawing_loop,
                                      args=(self.drawing_pipe, ))
        self.drawing_thread.daemon = True
        self.polling_thread = Thread(target=self.polling_loop,
                                     args=(self.polling_pipe, ))
        self.polling_thread.daemon = True
        # self.drawing_thread = Process(target=self.run)
        self.running.set()
        self.polling_thread.start()
        self.drawing_thread.start()

    def stop(self):
        self.running.clear()

        if self.drawing_thread is not None and self.drawing_thread.is_alive():
            # wait until thread stops
            self.drawing_thread.join(10)

        if self.polling_thread is not None and self.polling_thread.is_alive():
            # wait until thread stops
            self.polling_thread.join(10)




# an example that plots lines
class LivePlot(PlantDraw):
    def __init__(self, plant, refresh_period=1.0,
                 name='Serial Data', H=5.0, angi=[]):
        super(LivePlot, self).__init__(plant, refresh_period, name)
        self.H = H
        self.angi = angi
        # get first measurement
        state, t = plant.get_state(noisy=False)
        self.data = np.array([state])
        self.t_labels = np.array([t])

        # keep track of latest time stamp and state
        self.current_t = t
        self.previous_update_time = time()
        self.update_period = refresh_period

    def init_artists(self):
        plt.figure(self.name)
        self.lines = [plt.Line2D(self.t_labels, self.data[:, i],
                                 c=next(color_generator)[0])
                      for i in range(self.data.shape[1])]
        self.ax.set_aspect('auto', 'datalim')
        for line in self.lines:
            self.ax.add_line(line)
        self.previous_update_time = time()

    def _update(self, state, t):
        if t != self.current_t:
            if len(self.data) <= 1:
                self.data = np.array([state]*2)
                self.t_labels = np.array([t]*2)

            if len(self.angi) > 0:
                state[self.angi] = (state[self.angi]+np.pi) % (2*np.pi) - np.pi

            self.current_t = t
            # only keep enough data points to fill the window to avoid using
            # up too much memory
            curr_time = time()
            self.update_period = 0.95*self.update_period + \
                0.05*(curr_time - self.previous_update_time)
            self.previous_update_time = curr_time
            history_size = int(1.5*self.H/self.update_period)
            self.data = np.vstack((self.data, state))[-history_size:, :]
            self.t_labels = np.append(self.t_labels, t)[-history_size:]

            # update the lines
            for i in range(len(self.lines)):
                self.lines[i].set_data(self.t_labels, self.data[:, i])

            # update the plot limits
            plt.xlim([self.t_labels.min(), self.t_labels.max()])
            plt.xlim([t-self.H, t])
            mm = self.data.mean()
            ll = 1.05*np.abs(self.data[:, :]).max()
            plt.ylim([mm-ll, mm+ll])
            self.ax.autoscale_view(tight=True, scalex=True, scaley=True)

        return self.lines

class DoubleCartpole(ODEPlant):

    metadata = {
        'render.modes': ['human']
    }

    def __init__(self,plant_params,name='DoubleCartpole'):
        super().__init__(name,plant_params)
        # double cartpole system parameters
        self.plant_params=plant_params
        self.l1 = plant_params['link1_length']
        self.l2 = plant_params['link2_length']
        self.m1 = plant_params['link1_mass']
        self.m2 = plant_params['link2_mass']
        self.M = plant_params['cart_mass']
        self.b = plant_params['friction']
        self.g = plant_params['gravity']         
        self.state0_dist = plant_params['s0_dist']

            

        # pointer to the class that will draw the state of the carpotle system
        self.renderer = None

        # 6 state dims (x, dx, dtheta1, dtheta2, theta1, theta2)
        # o_lims = np.array([np.finfo(np.float).max for i in range(6)])
        # self.observation_space = spaces.Box(-o_lims, o_lims)
        # # 1 action dim (x_force)
        # a_lims = np.array([np.finfo(np.float).max for i in range(1)])
        # self.action_space = spaces.Box(-a_lims, a_lims)

    def dynamics(self, t, z):

        m1, m2, M, l1, l2, b, g = self.m1, self.m2, self.M,\
                                  self.l1, self.l2, self.b,\
                                  self.g
                                  
        f = self.u if self.u is not None else np.array([0])
        f = f.flatten()

        sz4 = np.sin(z[4])
        cz4 = np.cos(z[4])
        sz5 = np.sin(z[5])
        cz5 = np.cos(z[5])
        cz4m5 = np.cos(z[4] - z[5])
        sz4m5 = np.sin(z[4] - z[5])
        a0 = m2+2*M
        a1 = M*l2
        a2 = l1*(z[2]*z[2])
        a3 = a1*(z[3]*z[3])

        A = np.array([[2*(m1+m2+M), -a0*l1*cz4,     -a1*cz5],
                      [-3*a0*cz4,    (2*a0+2*M)*l1,  3*a1*cz4m5],
                      [-3*cz5,       3*l1*cz4m5,     2*l2]])
        b = np.array([2*f[0]-2*b*z[1]-a0*a2*sz4-a3*sz5,
                      3*a0*g*sz4 - 3*a3*sz4m5,
                      3*a2*sz4m5 + 3*g*sz5]).flatten()
        
        x = np.linalg.solve(A, b)

        dz = np.zeros((6,))
        dz[0] = z[1]
        dz[1] = x[0]
        dz[2] = x[1]
        dz[3] = x[2]
        dz[4] = z[2]
        dz[5] = z[3]

        return dz

    def reset(self):
        state0 = self.state0_dist()
        self.set_state(state0)
        return self.state

    def render(self, mode='human', close=False):
        if self.renderer is None:
            self.renderer = DoubleCartpoleDraw(self)
            self.renderer.init_ui()
        self.renderer.update(*self.get_state(noisy=False))

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
    def state_change(self,state):
        state_train=np.hstack([state[4],state[5],state[2],state[3],state[0],state[1]])
        return state_train

class DoubleCartpoleDraw(PlantDraw):
    def __init__(self, double_cartpole_plant, refresh_period=(1.0/240),
                 name='DoubleCartpoleDraw'):
        super(DoubleCartpoleDraw, self).__init__(double_cartpole_plant,
                                                 refresh_period, name)
        m1 = self.plant.m1
        m2 = self.plant.m2
        M = self.plant.M
        l1 = self.plant.l1
        l2 = self.plant.l2

        self.body_h = 0.5*np.sqrt(m1)
        self.mass_r1 = 0.05*np.sqrt(m2)  # distance to corner of bounding box
        self.mass_r2 = 0.05*np.sqrt(M)   # distance to corner of bounding box

        self.center_x = 0
        self.center_y = 0

        # initialize the patches to draw the cartpole
        self.body_rect = plt.Rectangle((self.center_x - 0.5*self.body_h,
                                       self.center_y - 0.125*self.body_h),
                                       self.body_h, 0.25*self.body_h,
                                       facecolor='black')
        self.pole_line1 = plt.Line2D((self.center_x, 0),
                                     (self.center_y, l1), lw=2, c='r')
        self.mass_circle1 = plt.Circle((0, l1), self.mass_r1, fc='y')
        self.pole_line2 = plt.Line2D((self.center_x, 0),
                                     (l1, l2), lw=2, c='r')
        self.mass_circle2 = plt.Circle((0, l1+l2), self.mass_r2, fc='y')

    def init_artists(self):
        self.ax.add_patch(self.body_rect)
        self.ax.add_patch(self.mass_circle1)
        self.ax.add_line(self.pole_line1)
        self.ax.add_patch(self.mass_circle2)
        self.ax.add_line(self.pole_line2)

    def _update(self, state, t):
        l1 = self.plant.l1
        l2 = self.plant.l2

        body_x = self.center_x + state[0]
        body_y = self.center_y
        mass1_x = -l1*np.sin(state[4]) + body_x
        mass1_y = l1*np.cos(state[4]) + body_y
        mass2_x = -l2*np.sin(state[5]) + mass1_x
        mass2_y = l2*np.cos(state[5]) + mass1_y

        self.body_rect.set_xy((body_x-0.5*self.body_h,
                               body_y-0.125*self.body_h))
        self.pole_line1.set_xdata(np.array([body_x, mass1_x]))
        self.pole_line1.set_ydata(np.array([body_y, mass1_y]))
        self.pole_line2.set_xdata(np.array([mass1_x, mass2_x]))
        self.pole_line2.set_ydata(np.array([mass1_y, mass2_y]))
        self.mass_circle1.center = (mass1_x, mass1_y)
        self.mass_circle2.center = (mass2_x, mass2_y)

        return (self.body_rect, self.pole_line1, self.mass_circle1,
                self.pole_line2, self.mass_circle2)
    


if __name__ =="__main__":
    Eparams=default_params.double_cartpole_params()
    env=DoubleCartpole(Eparams)