# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
class Distribution(object):
    '''
        Base class for distribution. Useful for estimating and sampling
        initial state distributions
    '''
    def fit(data):
        raise NotImplementedError

    def sample(self, n_samples=1):
        raise NotImplementedError

    @property
    def dim(self):
        return self.__dim

    @dim.setter
    def dim(self, dim):
        self.__dim = dim

class Gaussian(Distribution):
    def __init__(self, mean, cov):
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        self.dim = self.mean.size

    @property
    def cov(self):
        return self.__cov

    @cov.setter
    def cov(self, cov):
        self.__cov = cov
        if cov is not None:
            assert cov.shape[0] == cov.shape[1]
            self.cov_chol = np.linalg.cholesky(cov)

    def sample(self, n_samples=1):
        return self.mean + np.random.randn(
            n_samples, self.mean.size).dot(self.cov_chol)

    def __call__(self, mean=None, cov=None, n_samples=1):
        if mean is not None:
            self.mean = mean
        if cov is not None:
            self.cov = cov
        return self.sample(n_samples)

def complex_represent(data,angi):
    
    if type(data)==np.ndarray:
        if data.ndim<2:
            data=data.reshape([1,-1])
        angi_cs=np.zeros([data.shape[0],2*angi])
        for i in range(angi):
            angi_cs[:,2*i]=np.cos(data[:,i])
            angi_cs[:,2*i+1]=np.sin(data[:,i])
        data=np.concatenate([angi_cs,data[:,angi:]],axis=1)
    else:

        if len(data.shape)<2:
            data=tf.reshape(data,[1,-1])
        angi_cs=tf.zeros([data.shape.as_list()[0],0])

        for i in range(angi):

            angi_cs=tf.concat([angi_cs,tf.cos(data[:,i:i+1]),tf.sin(data[:,i:i+1])],axis=1)
  
        data=tf.concat([angi_cs,data[:,angi:]],axis=1)
    return data