import tensorflow as tf 
import tensorflow_probability as tfp 
import numpy as np 
from kalman_ruls.networks.measurement_models.base import MeasurementBase

class ConstantMeasurement(MeasurementBase):
    def __init__(self, xdim, ydim):
        super().__init__()
        self.xdim = xdim 
        self.ydim = ydim

        H = np.random.randn(ydim, xdim)
        self.H = tf.Variable(H, name="H", dtype=tf.float64)

        self.R_L = tfp.bijectors.FillScaleTriL(diag_bijector=tfp.bijectors.Softplus(low=np.array([0], dtype=np.float64))) 
        self.R_rt_flat = tf.Variable(tf.random.uniform([int(ydim/2. * (ydim + 1))], dtype=tf.float64), name="R_rt_flat", dtype=tf.float64)
       
    def create_Hs(self, xs):
        """
        Reshapes the H matrix (xdim, xdim) to (bs, seq, ydim, xdim) so it can work as an 
        input to the filtering and smoothing methods 

        Inputs:
            xs (tensor): any tensor that has the required (bs, seq) as it's first 2 dimension sizes 

        Outputs:
            Hs (tensor): the constant F matrix copied to have size (bs, seq, ydim, xdim) 
        """

        bs, seq, _ = xs.shape
        Hs = tf.expand_dims(tf.expand_dims(self.H, 0), 0)   # (1,1,ydim,xdim)
        Hs = tf.tile(Hs, [bs, seq, 1, 1])   # (bs, seq, ydim, xdim)
        return Hs 

    def create_Rs(self, xs):
        bs, seq, _ = xs.shape
        R_lower = self.R_L(self.R_rt_flat)
        R = tf.linalg.matmul(R_lower, R_lower, transpose_b=True) 
        Rs = tf.expand_dims(tf.expand_dims(R, 0), 0)   # (1,1,ydim,ydim)
        Rs = tf.tile(Rs, [bs, seq, 1, 1])
        return Rs 