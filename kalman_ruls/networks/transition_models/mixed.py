import tensorflow as tf 
import tensorflow_probability as tfp 
import numpy as np 
from kalman_ruls.networks.transition_models.base import TransitionBase
from kalman_ruls.networks.utils import MLP
naxis = tf.newaxis

class MixedTransition(TransitionBase):
    def __init__(self, xdim, **kwargs):
        super().__init__()
        self.xdim = xdim 
        K = kwargs["K"]
        hdim = kwargs["hdim"]


        F = np.random.randn(K, xdim, xdim)        
        self.F = tf.Variable(F, name="F", dtype=tf.float64)

        self.Q_L = tfp.bijectors.FillScaleTriL(diag_bijector=tfp.bijectors.Softplus(low=np.array([0], dtype=np.float64))) 
        self.Q_rt_flat = tf.Variable(tf.random.uniform([int(xdim/2. * (xdim + 1))], dtype=tf.float64), name="Q_rt_flat", dtype=tf.float64)

        self.log_weight_net = MLP(xdim, hdim, K)

    def normalize_weights(self, logw):
        """
        Normalizing log weights (dividing in log space is subtraction), hence, 
        we sum the weights by converting to uniform space first then summing and 
        converting back to log space

            i.e. Z = log(sum(exp(log weights)))
        
        then normalizing means we divide the weights by this value (or subtract in
        log space)

            i.e. normalized = log weights - Z 
        
        Inputs:
            logw (tensor): the log weights outputted from the network, size (bs, seq, K)

        Outputs:
            norm_w (tensor): the normalized weights in standard uniform space, size (bs, seq, K)
        """
        norm = tf.reduce_logsumexp(logw, axis=-1, keepdims=True)
        norm_logw = logw - norm 
        norm_w = tf.exp(norm_logw)
        return norm_w

    def get_weights(self, xs):
        logw = self.log_weight_net(xs)
        ws = self.normalize_weights(logw)
        return ws 
    
    def create_Fs(self, xs):
        """
        Given data (xs) the network will calculate weights to find an average 
        of the K transition dynamics matrices (F). 
        Hence, we are essentially interpolating between multiple profiles that 
        describe the dynamics of the system. 

        Inputs:
            xs (tensor): Data used as the input to the network to generate the weights, size (bs, seq, *) 

        Outputs:
            Fs (tensor): the transition matrices througout time, size (bs, seq, xdim, xdim)
        """
        ws = self.get_weights(xs)   # (bs, seq, K)
        F = self.F                  # (K, xdim, xdim)
        Fs = ws[..., naxis, naxis] * F[naxis, naxis, ...]   # (bs, seq, K, xdim, xdim)
        Fs = tf.reduce_sum(Fs, axis=2)  # (bs, seq, xdim, xdim)
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


        