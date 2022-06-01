import tensorflow as tf
import numpy as np
import utils
import default_params
from policy_network_build import policy
from dynamic_model_build import PINN

class PRL(PINN,policy):
    def __init__(self,PRL_params,sess,policy_scope_name='policy',dynamic_scope_name='dynamic',SC_scope_name='SC',cost_func=None):
        PINN.__init__(self,PRL_params,sess,dynamic_scope_name,SC_scope_name)
        policy.__init__(self,PRL_params,sess,policy_scope_name)
        self.sample_num=PRL_params['sample_num']
        self.max_steps=PRL_params['max_steps']
        self.s0_m=tf.placeholder(tf.float32,PRL_params['s_dim'],'s0_m')
        # self.s0_v=PRL_params['s0_v']
        self.s0_v=tf.placeholder(tf.float32,[PRL_params['s_dim'],PRL_params['s_dim']],'s0_v')
        self.s_dist = tf.contrib.distributions.MultivariateNormalFullCovariance(
                        loc=self.s0_m, covariance_matrix=self.s0_v)
        
        self.cost_func=cost_func
        self.angi = PRL_params['angi']  
        self.loss_predict=[]

        self.PRLloss=self.build_PRL()
        Poptimizer = tf.train.AdamOptimizer(PRL_params['Plr'])
        
        
        Pgrads,Pvalue = zip(*Poptimizer.compute_gradients(self.PRLloss, var_list=self.policy_variables))
        
        Pgrads_cor=[]
        
        for g,v in zip(Pgrads,Pvalue):
            acg=g/tf.maximum(tf.global_norm(Pgrads)/(0.16*tf.maximum(tf.global_norm(Pvalue),.1)),1)
            Pgrads_cor.append(acg)
        self.Ptrain = Poptimizer.apply_gradients(zip(Pgrads_cor, Pvalue))
        

        
    def build_PRL(self):
        for k in range(self.max_steps):
            s=self.s_dist.sample(self.sample_num)   

            s_c=utils.complex_represent(s, self.angi)

            a=self.build_policy(s_c)
            sa_c=tf.concat([s_c,a],axis=1)
            dsn_m,dsn_v=self.build_PINN(sa_c)
            dsn_m=s+dsn_m[:,:self.s_dim]
            self.s_m=tf.reduce_mean(dsn_m,axis=0)
            w1=tf.multiply(tf.matmul(tf.transpose(dsn_v),dsn_v),
                           tf.matrix_diag(np.float32(np.ones(self.s_dim))))/(self.sample_num)
            w2=tf.matmul(tf.transpose(dsn_m-self.s_m), (dsn_m-self.s_m))/(self.sample_num)
            self.s_v=w1+w2
            if self.cost_func==None:
                c=0
            else:
                    
                c=self.cost_func(s)
            # r=1-tf.exp(-(tf.pow(s[:,4]-0.6*tf.sin(s[:,0])-0.6*tf.sin(s[:,1]),2)+tf.pow(0.6*2-0.6*tf.cos(s[:,0])-0.6*tf.cos(s[:,1]),2))/2/.25)
            self.loss_predict.append(tf.reduce_mean(c))
            
            self.s_dist= tf.contrib.distributions.MultivariateNormalFullCovariance(
                            loc=self.s_m, covariance_matrix=self.s_v)
        return tf.reduce_sum(self.loss_predict)
    def train_PRL(self,s0_m,s0_v):
        _,loss=self.sess.run([self.Ptrain,self.PRLloss],{self.drop_prob:0,self.s0_m:s0_m,self.s0_v:s0_v})
        return loss
if __name__=='__main__':

    tf.reset_default_graph()
    Eparams=default_params.double_cartpole_params()
    Pparams=default_params.PRL_params(Eparams)
    sess=tf.Session()
    PRL=PRL(Pparams,sess)