import tensorflow as tf 
import tensorflow_probability as tfp 
import numpy as np 
tfd = tfp.distributions 

from kalman_ruls.networks.bayes_state_estimator import BayesStateEstimator
from kalman_ruls.networks.utils import MLP
from kalman_ruls.networks.transition_models.base import TransitionBase
from kalman_ruls.networks.measurement_models.base import MeasurementBase 
from kalman_ruls.networks.encoders.base import EncoderBase

class Kalman_DVAE(tf.keras.Model):
    def __init__(self, encoder: EncoderBase, transition_model: TransitionBase, measurement_model: MeasurementBase):
        super().__init__()
        rdim = 1 
        self.xdim = encoder.xdim 
        self.zdim = encoder.zdim 
        self.hdim = encoder.hdim
        self.rdim = rdim

        # Filter + Generative model 
        self.model = BayesStateEstimator(transition_model, measurement_model)  # measurements and process all in latent space 

        # initial conditions net 
        self.init_net = MLP(self.xdim, self.hdim, self.zdim + int(self.zdim/2. * (self.zdim+1)))
        self.P0_L = tfp.bijectors.FillScaleTriL(diag_bijector=tfp.bijectors.Softplus(low=np.array([0], dtype=np.float64)))

        # deterministic 
        self.encoder = encoder 

    def get_init_states(self, xs):
        x0 = xs[:,0,:]
        init_states = self.init_net(x0)

        m0 = init_states[...,:self.zdim]
        P0_flat = init_states[...,self.zdim:]
        P0_L = self.P0_L(P0_flat)
        P0 = tf.matmul(P0_L, P0_L, transpose_b=True)
        return m0, P0 

    def call(self, xs, us=None, ds=None): 
        bs = xs.shape[0]

        # initial values 
        z0, P0 = self.get_init_states(xs)

        # controls 
        if us is None:
            us = self.encoder.encode_u(xs)
        if ds is None:
            ds = self.encoder.encode_d(xs)

        # use generative model p(z,r|x)
        zs, Ps, rs, Ss = self.model.predict(z0, P0, us, ds)
        return zs, Ps, rs, Ss

    def inference(self, xs, rs, z0, P0, parallel=True):
        us = self.encoder.encode_u(xs)
        ds = self.encoder.encode_d(xs)

        fzs, fzPs = self.model.filter(rs, z0, P0, us, ds, parallel=parallel)

        return fzs, fzPs, us, ds
    
    def replay_overshoot(self, xs, rs, fzs, fPs, us, ds, parallel=True):
        """
        See paper,
        
        A. H. Li, P. Wu and M. Kennedy, "Replay Overshooting: Learning Stochastic Latent Dynamics with the Extended Kalman Filter," 
        2021 IEEE International Conference on Robotics and Automation (ICRA), Xi'an, China, 2021, pp. 852-858, doi: 10.1109/ICRA48506.2021.9560811.

        for more details on Replay Overshooting. 
        """
        # smoother to get initial conditions
        szs, sPs = self.model.smooth(fzs, fPs, us, parallel)

        # using smoothing initial conditions (z0, P0) and extrapolate to find p(rul_{1:T}|X)
        sz0, sP0 = szs[:,0,:], sPs[:,0,:]
        _, _, r_ests, Ss = self(xs, us, ds)

        # calculate the -log likelihood 
        rul_dists = tfd.MultivariateNormalTriL(r_ests, tf.linalg.cholesky(Ss))
        nll = -tf.reduce_sum(rul_dists.log_prob(rs), 1) 
        nll = tf.reduce_mean(nll)
        return nll 

    def get_loss(self, xs, rs, replay=True, alpha=0.4, Beta=1e-4, parallel=True):
        z0, P0 = self.get_init_states(xs)

        fzs, fPs, us, ds = self.inference(xs, rs, z0, P0, parallel)
        rul_dists = self.model.get_marginal_dist(fzs, fPs, z0, P0, us, ds)

        # regularize eigenvalues for stability (|eigvals| < 1 = stable in a discrete system)
        F = self.model.transition.F
        F_eigs, _ = tf.linalg.eig(F)
        reg = tf.reduce_sum(tf.math.real(F_eigs) ** 2)  
        
        loss = -tf.reduce_sum(rul_dists.log_prob(rs), 1)   # take the mean along the batch dim.
        loss = tf.reduce_mean(loss)  # sum log probs across time (equivalent to taking products of all the probs)
        loss = loss + Beta * reg 

        if replay:
            nll = self.replay_overshoot(xs, rs, fzs, fPs, us, ds)
            loss_with_replay = alpha * loss + (1. - alpha) * nll 
            return loss_with_replay
        return loss 
    
    def store_inference_models(self, inference_encoder: EncoderBase, inference_transition: TransitionBase,):
        self.inference_encoder = inference_encoder
        self.inference_transition = inference_transition
        self.inf_model = BayesStateEstimator(self.inference_transition, self.model.measurement) 

    def get_ELBO(self, xs, rs, Beta=1e-4):
        z0, P0 = self.get_init_states(xs)

        us = self.encoder.encode_u(xs)              # regular encoding of x_{1:T}
        ds = self.encoder.encode_d(xs)
        
        xrs = tf.concat([xs, rs], axis=-1)
        us_inf = self.inference_encoder.encode_u(xrs)    # combine obs. and conditional sequences to get y_{1:T} and x_{1:T} encoded 
        
        # use inference + measurement model 
        zs_inf, Ps_inf, rs_inf, Ss_inf = self.inf_model.predict(z0, P0, us_inf, ds)
        
        # use transition model to find prior stats 
        Fs, _, Qs, _ = self.model.create_system(us)
        zs = tf.linalg.matvec(Fs, zs_inf) + us
        Ps = tf.linalg.matmul(Fs, tf.linalg.matmul(Ps_inf, Fs, transpose_b=True)) + Qs

        p_pri = tfd.MultivariateNormalTriL(zs, tf.linalg.cholesky(Ps))
        p_inf = tfd.MultivariateNormalTriL(zs_inf, tf.linalg.cholesky(Ps_inf))
        p_obs = tfd.MultivariateNormalTriL(rs_inf, tf.linalg.cholesky(Ss_inf))

        nll = -tf.reduce_sum(p_obs.log_prob(rs), 1) 
        nll = tf.reduce_mean(nll)

        kl = tf.reduce_sum(tfd.kl_divergence(p_inf, p_pri), 1)
        kl = tf.reduce_mean(kl)

        loss = nll + kl 

        # regularize eigenvalues for stability (|eigvals| < 1 = stable in a discrete system)
        F = self.model.transition.F
        F_eigs, _ = tf.linalg.eig(F)
        reg = tf.reduce_sum(tf.math.real(F_eigs) ** 2)  
        
        loss = loss + Beta * reg 

        return loss 