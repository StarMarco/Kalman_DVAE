import tensorflow as tf 
import tensorflow_probability as tfp 
import numpy as np 
from kalman_ruls.networks.transition_models.base import TransitionBase

class ConstantTransition(TransitionBase):
    def __init__(self, xdim):
        super().__init__()
        self.xdim = xdim

        F = np.random.randn(xdim, xdim)
        self.F = tf.Variable(F, name="F", dtype=tf.float64)

        self.Q_L = tfp.bijectors.FillScaleTriL(diag_bijector=tfp.bijectors.Softplus(low=np.array([0], dtype=np.float64))) 
        self.Q_rt_flat = tf.Variable(tf.random.uniform([int(xdim/2. * (xdim + 1))], dtype=tf.float64), name="Q_rt_flat", dtype=tf.float64)
       
    def create_Fs(self, xs):
        """
        Reshapes the F matrix (xdim, xdim) to (bs, seq, xdim, xdim) so it can work as an 
        input to the filtering and smoothing methods 

        Inputs:
            xs (tensor): any tensor that has the required (bs, seq) as it's first 2 dimension sizes 

        Outputs:
            Fs (tensor): the constant F matrix copied to have size (bs, seq, xdim, xdim) 
        """

        bs, seq, _ = xs.shape
        Fs = tf.expand_dims(tf.expand_dims(self.F, 0), 0)   # (1,1,xdim,xdim)
        Fs = tf.tile(Fs, [bs, seq, 1, 1])   # (bs, seq, xdim, xdim)
        return Fs 

    def create_Qs(self, xs):
        """
        Create constant through time Q matrices 
        """
        bs, seq, _ = xs.shape
        Q_lower = self.Q_L(self.Q_rt_flat)
        Q = tf.linalg.matmul(Q_lower, Q_lower, transpose_b=True) 
        Qs = tf.expand_dims(tf.expand_dims(Q, 0), 0)   # (1,1,xdim,xdim)
        Qs = tf.tile(Qs, [bs, seq, 1, 1])
        return Qs 

