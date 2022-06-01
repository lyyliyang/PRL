import tensorflow as tf
import numpy as np
import default_params


class policy():
    def __init__(self,PRL_params,sess,policy_scope_name='policy'):
        self.PRL_params=PRL_params
        self.policy_scope_name=policy_scope_name
        self.sess=sess
        self.Pdim=np.concatenate([[self.PRL_params['Pin_dim']],self.PRL_params['Phidden_dim'],[self.PRL_params['Pou_dim']]])

        with tf.variable_scope(self.policy_scope_name): 
            self.policy_w=[]
            self.policy_b=[]
            for i, size in enumerate(self.Pdim[1:]):
                self.policy_w.append(tf.get_variable("policy_w_{}".format(i), [self.Pdim[i], size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(0.,.1/self.Pdim[i])))
                self.policy_b.append(tf.get_variable("policy_b_{}".format(i), [1, size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(0., 0.001)))
               
        self.policy_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.policy_scope_name)       
        
        self.Pin=tf.placeholder(tf.float32,[None,PRL_params['Pin_dim']],'Pin')
        self.action=self.build_policy(self.Pin)
        

    def build_policy(self,Input):
        policy_net=[]
        for i in range(len(self.Pdim)-2):
            if i==0:
                policy_net.append(tf.nn.relu(tf.matmul(Input, self.policy_w[i])+self.policy_b[i],name='policy_net_{}'.format(i)))
            else:
                policy_net.append(tf.nn.relu(tf.matmul(policy_net[i-1], self.policy_w[i])+self.policy_b[i],name='policy_net_{}'.format(i)))
        policy_net.append(tf.tanh(tf.matmul(policy_net[len(self.Pdim)-3], self.policy_w[len(self.Pdim)-2])+self.policy_b[len(self.Pdim)-2]))

        return policy_net[-1]
    
    def policy_forward(self,state):
        action=self.sess.run(self.action,feed_dict={self.Pin:state.reshape([1,-1])})
        return action
        
if __name__=='__main__':
    tf.reset_default_graph()
    Eparams=default_params.double_cartpole_params()
    Pparams=default_params.PRL_params(Eparams)
    sess=tf.Session()
    PINN=policy(Pparams,sess)
        