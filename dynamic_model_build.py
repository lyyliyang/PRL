import tensorflow as tf
import numpy as np
import utils
import default_params

class Memory(object):
    def __init__(self,PRL_params):
        self.Memory_trans_size=PRL_params['Memory_trans_size']
        
        self.Memory_prob_size=PRL_params['Memory_prob_size']
        self.max_steps=PRL_params['max_steps'] 
        self.angi=PRL_params['angi']     
        
        self.s_dim=PRL_params['s_dim']
        self.a_dim=PRL_params['a_dim']
        self.Din_dim=PRL_params['Din_dim']
        self.coefficient=PRL_params['coefficient']
        self.Memory_trans=np.zeros([0,self.s_dim*2+2*(self.s_dim+self.a_dim)+self.a_dim])      
        self.Memory_prob=np.zeros([0,self.s_dim*2+2*(self.s_dim+self.a_dim)+self.s_dim])    
        self.Memory_prob_label=np.zeros([0,self.Din_dim])    
        

    def push_trans(self,data):
        
        data=self._build_piror_output(data)
        if len(self.Memory_trans)<self.Memory_trans_size:
            self.Memory_trans=np.vstack([self.Memory_trans,data])
        else:
            self.Memory_trans=np.roll(self.Memory_trans,-1,axis=0)
            self.Memory_trans[-1,:]=data
            
    def _build_piror_output(self,data):
        sa=data[:self.s_dim+self.a_dim]
        s_=data[-self.s_dim:]
        s_piror=s_
        for i in range(self.s_dim+self.a_dim):
            s_piror=np.hstack([s_piror,np.cos(sa[i]/self.coefficient[i]),np.sin(sa[i]/self.coefficient[i])])    
        data=np.concatenate([sa,s_piror])
        return data
            
    def push_prob(self):
        data=self.Memory_trans[-2*self.max_steps:,:]
        data_c=utils.complex_represent(data[:,:self.s_dim+self.a_dim], self.angi)
        l_mu,l_sig=self.dynamic_forward(data_c[:,:])

        if len(self.Memory_prob)<self.Memory_prob_size:
            self.Memory_prob=np.vstack([self.Memory_prob,np.hstack([l_mu,l_sig,data[:,self.s_dim+self.a_dim:self.s_dim*2+self.a_dim]])])
            self.Memory_prob_label=np.vstack([self.Memory_prob_label,data_c[:,:self.Din_dim]])
        else:
            self.Memory_prob=np.roll(self.Memory_prob,-2*self.max_steps,axis=0)
            self.Memory_prob_label=np.roll(self.Memory_prob_label,-2*self.max_steps,axis=0)
            self.Memory_prob[-2*self.max_steps:,:]=np.hstack([l_mu,l_sig,data[-2*self.max_steps:,self.s_dim+self.a_dim:self.s_dim*2+self.a_dim]])
            self.Memory_prob_label[-2*self.max_steps:,:]=data_c[:,:self.Din_dim]
            
    
class PINN(Memory):
    def __init__(self,PRL_params,sess,dynamic_scope_name='dynamic',SC_scope_name='SC'):
        super().__init__(PRL_params)
        self.sess=sess
        self.PRL_params=PRL_params
        

        
        self.Ddim=np.concatenate([[self.PRL_params['Din_dim']],self.PRL_params['Dhidden_dim'],[self.PRL_params['Dou_dim']]])

        with tf.variable_scope(dynamic_scope_name): 
            self.dynamic_w=[]
            self.dynamic_b=[]
            for i, size in enumerate(self.Ddim[1:]):

                self.dynamic_w.append(tf.get_variable("dynamic_w_{}".format(i), [self.Ddim[i], size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(0.,.1/self.Ddim[i])))
                self.dynamic_b.append(tf.get_variable("dynamic_b_{}".format(i), [1, size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(0., 0.001)))
        self.dynamic_varibles = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=dynamic_scope_name)   
        
        with tf.variable_scope(SC_scope_name):
            self.SC_params=tf.get_variable('SC_params',initializer=self.PRL_params['SC_init'])
        self.SC_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=SC_scope_name)
        
        self.Din=tf.placeholder(tf.float32,[None,PRL_params['Din_dim']],'Din')
        self.Dlabel=tf.placeholder(tf.float32,[None,self.s_dim+2*(self.s_dim+self.a_dim)],'Dlabel')
        self.SClabel=tf.placeholder(tf.float32,[None,self.s_dim],'SClabel')
        self.drop_prob=tf.placeholder(tf.float32)
        self.Dprediction,self.Dsigma=self.build_PINN(self.Din)
        
        self.Dstep=tf.Variable(0, trainable=False)  
        self.Dloss= tf.reduce_mean(tf.squared_difference(self.Dlabel, self.Dprediction))
        

        Dlr = tf.train.exponential_decay(PRL_params['Dlr'],
                                         global_step=self.Dstep,
                                         decay_steps=PRL_params['Dds'],
                                         decay_rate=PRL_params['Ddr'])
        
        Doptimizer = tf.train.AdamOptimizer(Dlr)
        Dgrads = Doptimizer.compute_gradients(self.Dloss, var_list=self.dynamic_varibles)
        self.Dtrain = Doptimizer.apply_gradients(Dgrads)
        
        
        
        SCdist=tf.distributions.Normal(self.Dprediction[:,:self.s_dim],tf.abs(self.Dsigma))
         
        self.SCnlp=-tf.reduce_sum(tf.log(tf.maximum(SCdist.prob(self.SClabel),0.01))-0.1*tf.nn.softplus(self.SC_params[2,:]))
        
        SCoptimizer = tf.train.AdamOptimizer(PRL_params['SClr'])
        
        SCgrads = SCoptimizer.compute_gradients(self.SCnlp, var_list=self.SC_variables)
        
        self.SCtrain = SCoptimizer.apply_gradients(SCgrads)
        
        
    def build_PINN(self,Din):
        coefficient=self.PRL_params['coefficient']
        dynamic_net=[]
        for i in range(len(self.Ddim)-2):
            if i==0:
                dynamic_net.append(tf.nn.relu(tf.matmul(Din, self.dynamic_w[i])+self.dynamic_b[i],name='dynamic_net_{}'.format(i)))
            else:
                dynamic_net.append(tf.nn.relu(tf.matmul(dynamic_net[i-1], self.dynamic_w[i])+self.dynamic_b[i],name='dynamic_net_{}'.format(i)))
            dynamic_net[-1]=tf.nn.dropout(dynamic_net[-1],rate=self.drop_prob)
        dynamic_net.append(tf.matmul(dynamic_net[len(self.Ddim)-3], self.dynamic_w[len(self.Ddim)-2])+self.dynamic_b[len(self.Ddim)-2])

        SC2=0
        for i in range(self.s_dim+self.a_dim):
  
            if i<self.angi:
                SC2+=tf.reshape(tf.square((Din[:,2*i])-dynamic_net[-1][:,self.s_dim+2*i])
                                +tf.square((Din[:,2*i+1])-dynamic_net[-1][:,self.s_dim+2*i+1]),[-1,1])
            else:
                SC2+=tf.reshape(tf.square((tf.cos(Din[:,self.angi+i]/coefficient[i]))-dynamic_net[-1][:,self.s_dim+2*i])
                                +tf.square((tf.sin(Din[:,self.angi+i]/coefficient[i]))-dynamic_net[-1][:,self.s_dim+2*i+1]),[-1,1])

        sigma=tf.nn.softplus(self.SC_params[1,:])+(1-tf.exp(-tf.nn.softplus(self.SC_params[0,:])*tf.sqrt(1e-8+SC2)))*tf.nn.softplus(self.SC_params[2,:])

        
        return dynamic_net[-1],sigma
    def dynamic_forward(self,sa):
        if  sa.ndim<2:
            sa=sa.reshape([1,-1])
        s_mean,s_sigma=self.sess.run([self.Dprediction,self.Dsigma],feed_dict={self.Din:sa,self.drop_prob:0})
        return s_mean,s_sigma
    def train_dynamic(self,step,drop_prob):
        data=self.Memory_trans
        data_c=utils.complex_represent(self.Memory_trans, self.angi)
        _,loss=self.sess.run([self.Dtrain,self.Dloss],feed_dict={
                self.Din:data_c[:,:self.s_dim+self.a_dim+self.angi],
                self.Dlabel:data[:,self.s_dim+self.a_dim:],
                self.Dstep:step,
                self.drop_prob:drop_prob})
        return loss
    def train_SC(self):
        _,loss=self.sess.run([self.SCtrain,self.SCnlp],feed_dict={
                self.Dprediction:self.Memory_prob[:,:self.s_dim+2*(self.s_dim+self.a_dim)],
                self.SClabel:self.Memory_prob[:,-self.s_dim:],
                self.Din:self.Memory_prob_label})
        return loss
if __name__=='__main__':
    tf.reset_default_graph()
    Eparams=default_params.double_cartpole_params()
    Pparams=default_params.PRL_params(Eparams)
    sess=tf.Session()
    PINN=PINN(Pparams,sess)
    