import tensorflow as tf 
import tensorflow_probability as tfp 
import numpy as np 
tfd = tfp.distributions 

from kalman_ruls.filtering_smoothing.filtering import kf, pkf, LGSSM
from kalman_ruls.filtering_smoothing.smoothing import ks, pks 
from kalman_ruls.networks.transition_models.base import TransitionBase
from kalman_ruls.networks.measurement_models.base import MeasurementBase

def transpose_first_dims(*args):
    """
    Takes in multiple matrices and vectors and swaps the first 2 dimensions/axes 
    then returns them back to the user as a tuple 
    """
    elements = [] 
    for ele in args:
        ele_transpose = tf.einsum("ij...->ji...", ele)
        elements.append(ele_transpose)
    return tuple(elements) 

class BayesStateEstimator(tf.keras.Model):
    def __init__(self, transition_model: TransitionBase, measurement_model: MeasurementBase):
        super().__init__()
        self.x_dim = measurement_model.xdim 
        self.y_dim = measurement_model.ydim 

        self.transition = transition_model
        self.measurement = measurement_model

    def create_Fs(self, ys):
        """
        Create a state space matrix for discrete dynamics used in a Kalman filter, 

        Inputs:
            ys (tensor): Any tensor of size (bs, seq, ydim)

        Outputs:
            F (tensor): size (bs, seq, zdim, zdim)
        """
        Fs = self.transition.create_Fs(ys)
        return Fs 

    def create_Hs(self, ys):
        Hs = self.measurement.create_Hs(ys)
        return Hs 

    def get_noise_covar(self, ys):
        Qs = self.transition.create_Qs(ys)
        Rs = self.measurement.create_Rs(ys)
        return Qs, Rs 

    def get_lgssm(self, F, Q, H, R, P0, m0):
        return LGSSM(F, Q, H, R, P0, m0)
        
    def batch_eye(self, ys, dim):
        bs = ys.shape[0]
        I = np.eye(dim) * np.ones([bs, dim, dim])
        return I

    def create_system(self, us):
        Fs = self.create_Fs(us)
        Hs = self.create_Hs(us)
        Qs, Rs = self.get_noise_covar(us)
        return Fs, Hs, Qs, Rs 

    def filter(self, ys, m0, P0, us, ds, parallel=True):
        """
        Inputs: 
            ys (Tensor): measurements, size (bs, seq, ydim)
            m0 (Tensor): intial state, size (bs, xdim)
            P0 (Tensor): intial state, size (bs, xdim, xdim)
            us (Tensor): control inputs impacting the state dynamics, size (bs, seq, xdim)
            ds (Tensor): control inputs impacting the measurement model, size (bs, seq, ydim)
            parallel (bool): if True performs the parallel-in-time Kalman filter algorithm  

        Outputs:
            fxs (Tensor): sequence of filtered means, size (bs, seq, xdim)
            fPs (Tensor): sequence of filtered covariances, size (bs, seq, xdim, xdim)
        """
        Fs = self.create_Fs(us)
        Hs = self.create_Hs(us)

        #m0, P0 = self.init_variables(ys)
        Qs, Rs = self.get_noise_covar(us)

        # Seq first for scans (bs, seq, ...) -> (seq, bs, ...)
        ys, us, ds, Fs, Qs, Hs, Rs = transpose_first_dims(ys, us, ds, Fs, Qs, Hs, Rs)

        # create Linear Gaussian State Space Model (stores dynamics data)
        self.lgssm = self.get_lgssm(Fs, Qs, Hs, Rs, P0, m0)

        if parallel:
            fxs, fPs = pkf(self.lgssm, ys, us, ds)  # type: ignore
        else:
            fxs, fPs = kf(self.lgssm, ys, us, ds)  # type: ignore

        fxs, fPs = transpose_first_dims(fxs, fPs)   # (seq,bs,...) -> (bs,seq,...)
        return fxs, fPs

    def smooth(self, fxs, fPs, us, parallel=True):
        """
        Need to run filter before hand to get filtered means and convariances, the 
        Linear Gaussian State Space Model (lgssm) is also defined in the filter method and needed 
        for smoothing. 
        """
        fxs, fPs = transpose_first_dims(fxs, fPs) # seq first 
        us = tf.transpose(us, perm=[1,0,2])

        if parallel:
            sxs, sPs = pks(self.lgssm, fxs, fPs, us)  # type: ignore
        else:
            sxs, sPs = ks(self.lgssm, fxs, fPs, us)  # type: ignore

        sxs, sPs = transpose_first_dims(sxs, sPs)
        return sxs, sPs

    def get_marginal_dist(self, fxs, fPs, m0, P0, us, ds):
        Fs, Qs, Hs, Rs, P0, m0 = self.lgssm
        fxs, fPs, us, ds = transpose_first_dims(fxs, fPs, us, ds)

        filtered_means = tf.concat([tf.expand_dims(m0, 0), fxs[:-1]], axis=0)
        filtered_covs = tf.concat([tf.expand_dims(P0, 0), fPs[:-1]], axis=0)
        predicted_means = tf.linalg.matvec(Fs, filtered_means) + us 
        predicted_covs = tf.linalg.matmul(Fs, tf.linalg.matmul(filtered_covs, Fs, transpose_b=True)) + Qs 
        obs_means = tf.linalg.matvec(Hs, predicted_means) + ds
        obs_covs = tf.linalg.matmul(Hs, tf.linalg.matmul(predicted_covs, Hs, transpose_b=True)) + Rs

        obs_means, obs_covs = transpose_first_dims(obs_means, obs_covs)
        dists = tfd.MultivariateNormalTriL(obs_means, tf.linalg.cholesky(obs_covs))
        return dists

    def predict(self, x0, P0, us, ds):
        Fs = self.create_Fs(us)
        Hs = self.create_Hs(us)
        Qs, Rs = self.get_noise_covar(us)

        # seq as first dim, (bs,seq,...) -> (seq,bs,...)
        us, ds, Fs, Qs, Hs, Rs = transpose_first_dims(us, ds, Fs, Qs, Hs, Rs)

        y0 = tf.linalg.matvec(Hs[0], x0)
        S0 = tf.linalg.matmul(Hs[0], tf.linalg.matmul(P0, Hs[0], transpose_b=True))

        def predict_body(carry, inp):
            x, P, y, S = carry
            F, Q, H, R, u, d = inp

            x = tf.linalg.matvec(F, x) + u
            P = tf.linalg.matmul(F, tf.linalg.matmul(P, F, transpose_b=True)) + Q 
            y = tf.linalg.matvec(H, x) + d
            S = tf.linalg.matmul(H, tf.linalg.matmul(P, H, transpose_b=True)) + R

            return x, P, y, S

        xs, Ps, ys, Ss = tf.scan(predict_body, (Fs, Qs, Hs, Rs, us, ds), (x0, P0, y0, S0))
        xs, Ps, ys, Ss = transpose_first_dims(xs, Ps, ys, Ss)
        return xs, Ps, ys, Ss