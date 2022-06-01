# -*- coding: utf-8 -*-
"""
Created on Sun May  1 21:10:13 2022

@author: 98040
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
import default_params
from long_term_predict import PRL
from double_catrpole_build import DoubleCartpole
import tqdm


tf.reset_default_graph()
def cost_func(s):
    c=1-tf.exp(-(tf.pow(s[:,4]-0.6*tf.sin(s[:,0])-0.6*tf.sin(s[:,1]),2)+tf.pow(0.6*2-0.6*tf.cos(s[:,0])-0.6*tf.cos(s[:,1]),2))/2/.25)
    return c

Eparams=default_params.double_cartpole_params()
env=DoubleCartpole(Eparams)

Pparams=default_params.PRL_double_cartpole_params(Eparams)
sess=tf.Session()
PRL=PRL(Pparams,sess,cost_func=cost_func)
sess.run(tf.global_variables_initializer())


for j in range(Pparams['max_episodes']):   
    
    real_loss=0   
    state=env.reset()
    state_t=env.state_change(state)
    for i in range(Pparams['max_steps']):
        state_c=utils.complex_represent(state_t, Pparams['angi'])
        if j<=2:
            action=np.array([np.random.uniform(-1,1)])
        else:
            action=PRL.policy_forward(state_c)
        next_state,loss,done,_=env.step(action*Pparams['Pou_max'])
        if done:
            break
        real_loss=real_loss+loss
        next_state_t=env.state_change(next_state)
        dstate_t=next_state_t-state_t
        data=np.concatenate([state_t,action.flatten(),dstate_t])
        PRL.push_trans(data)
        state_t=next_state_t
    
    print('##############   Current episode : %d, Total step:%d, real reward %2f   #############'%(j,i,real_loss))
    if j>0:

        PRL.push_prob()
        SC_lall=[]
        SC_loss=0
        pbar = tqdm.tqdm(total=300)
        pbar.desc='SC_learning  '
        for k in range(300):
            
            SC_loss=PRL.train_SC()
            SC_lall.append(SC_loss)
            pbar.update(1)
            pbar.postfix={'SC_loss  ':format(SC_loss, '.1f')}
        pbar.close()
        
    PINN_lall=[]
    pbar = tqdm.tqdm(total=1000)
    pbar.desc='PINN_learning'
    for k in range(1000):  
        PINN_loss=PRL.train_dynamic(k,np.max([.3-0.06*j,0.]))
        PINN_lall.append(PINN_loss)

        pbar.update(1)
        pbar.postfix={'PINN_loss':PINN_loss}
        if PINN_loss<1e-6:  
            break     
    pbar.close()
        
    

    if j>1:
        pbar = tqdm.tqdm(total=200)
        pbar.desc='PRL_learning '
        RPL_lall=[]
        for k in range(400):
            PRL_loss=PRL.train_PRL(env.state_change(Eparams['s0_m']))
            RPL_lall.append(PRL_loss)
            if k>200 and RPL_lall[-1]<RPL_lall[-2]: 
               break
            pbar.update(1)
            pbar.postfix={'RPL_loss ':PRL_loss}
        pbar.close()
        
sess.close()
        
        