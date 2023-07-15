import tensorflow as tf 
import tensorflow_probability as tfp 
import numpy as np 
from kalman_ruls.networks.measurement_models.base import MeasurementBase
from kalman_ruls.networks.utils import MLP
naxis = tf.newaxis

class MixedMeasurement(MeasurementBase):
    def __init__(self, xdim, ydim, **kwargs):
        super().__init__()
        self.xdim = xdim 
        self.ydim = ydim 
        K = kwargs["K"]
        hdim = kwargs["hdim"]

        H = np.random.randn(K, ydim, xdim)        
        self.H = tf.Variable(H, name="H", dtype=tf.float64)

        self.R_L = tfp.bijectors.FillScaleTriL(diag_bijector=tfp.bijectors.Softplus(low=np.array([0], dtype=np.float64))) 
        self.R_rt_flat = tf.Variable(tf.random.uniform([int(ydim/2. * (ydim + 1))], dtype=tf.float64), name="R_rt_flat", dtype=tf.float64)

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
    
    def create_Hs(self, xs):
        """
        Given data (xs) the network will calculate weights to find an average 
        of the K measurement matrices (H). 
        Hence, we are essentially interpolating between multiple profiles that 
        describe the dynamics of the system. 

        Inputs:
            xs (tensor): Data used as the input to the network to generate the weights, size (bs, seq, *) 

        Outputs:
            Hs (tensor): the transition matrices througout time, size (bs, seq, xdim, xdim)
        """
        ws = self.get_weights(xs)   # (bs, seq, K)
        H = self.H                  # (K, xdim, xdim)
        Hs = ws[..., naxis, naxis] * H[naxis, naxis, ...]   # (bs, seq, K, xdim, xdim)
        Hs = tf.reduce_sum(Hs, axis=2)  # (bs, seq, xdim, xdim)
        return Hs 

    def create_Rs(self, xs):
        """
        Create constant through time R matrices 
        """
        bs, seq, _ = xs.shape
        R_lower = self.R_L(self.R_rt_flat)
        R = tf.linalg.matmul(R_lower, R_lower, transpose_b=True) 
        Rs = tf.expand_dims(tf.expand_dims(R, 0), 0)   # (1,1,xdim,xdim)
        Rs = tf.tile(Rs, [bs, seq, 1, 1])
        return Rs 


        