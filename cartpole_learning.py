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
from cartpole_build import Cartpole
import tqdm


tf.reset_default_graph()


Eparams=default_params.cartpole_params()
env=Cartpole(Eparams)

def cost_func(s):
    c=1-tf.exp(-(tf.pow(s[:,2]+Eparams['pole_length']*tf.sin(s[:,0]),2)
                 +tf.pow(Eparams['pole_length']+Eparams['pole_length']*tf.cos(s[:,0]),2))/2/Eparams['cw']**2)
    return c

Pparams=default_params.PRL_cartpole_params(Eparams)
sess=tf.Session()
PRL=PRL(Pparams,sess,cost_func=cost_func)
sess.run(tf.global_variables_initializer())

real_loss_all=[]
init_state_t_all=[]
for j in range(Pparams['max_episodes']):   
    
    real_loss=0   
    state=env.reset()
    state_t=env.state_change(state)
    init_state_t_all.append(state_t)
    for i in range(Pparams['max_steps']):
        state_c=utils.complex_represent(state_t, Pparams['angi'])
        if j<=1:
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
    real_loss_all.append(real_loss)
    print('##############   Current episode : %d, Total step:%d, real loss %2f   #############'%(j,i,real_loss))
    if j>0:

        PRL.push_prob()
        SC_lall=[]
        SC_loss=0
        pbar = tqdm.tqdm(total=200)
        pbar.desc='SC_learning  '
        for k in range(200):
            
            SC_loss=PRL.train_SC()
            SC_lall.append(SC_loss)
            pbar.update(1)
            pbar.postfix={'SC_loss  ':format(SC_loss, '.1f')}
        pbar.close()
        
    PINN_lall=[]
    pbar = tqdm.tqdm(total=1000)
    pbar.desc='PINN_learning'
    for k in range(1000):  
        PINN_loss=PRL.train_dynamic(k,np.max([.3-0.1*j,0.]))
        PINN_lall.append(PINN_loss)

        pbar.update(1)
        pbar.postfix={'PINN_loss':format(PINN_loss,'.6f')}
        if PINN_loss<1e-5:  
            break     
    pbar.close()
        
    

    if j>1:
        pbar = tqdm.tqdm(total=200)
        pbar.desc='PRL_learning '
        RPL_lall=[]
        for k in range(400):
            PRL_loss=PRL.train_PRL(env.state_change(np.mean(init_state_t_all,0)),
                                   np.diag(env.state_change(np.var(init_state_t_all,0))))
            RPL_lall.append(PRL_loss)
            if k>200 and RPL_lall[-1]<RPL_lall[-2]: 
               break
            pbar.update(1)
            pbar.postfix={'RPL_loss':format(PRL_loss,'.2f')}
        pbar.close()
        
sess.close()
plt.plot(real_loss_all)
        