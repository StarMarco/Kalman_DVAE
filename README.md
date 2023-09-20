# Kalman_DVAE

*Note this code is research code used to test a models performance on a benchmark dataset. This is not optimized or designed for production/industry applications. I simply post it here as it may help 
others understand my research or the topics presented here.* 

A Dynamical Variational Autoencoder (DVAEs) that is trained using a Kalman filter (as opposed to an ELBO-based loss). This is applied to the CMAPSS turbofan engine dataset from NASA for Remaining Useful Life (RUL) estimation using sensor signals as inputs
(data can be found in this repository or [here](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) under header 6. Turbofan Engine Degradation Simulation). 
The code here was used in my PhD thesis ([Degradation Vector Fields with Uncertainty Considerations](http://hdl.handle.net/20.500.11937/93343)). The code here is based on Chapter 7 of the thesis. There is a notebook that goes through
a simple use case of the model for RUL estimation. This may be easier to understand than the training and testing scripts whose arguments try to account for many different cases the user may want to test.

The training and testing scripts allow one to train and test models from the command line using arguments to perform different tests on a variety of sub-models used to construct a Kalman-DVAE (K-DVAE). 
The main idea behind the K-DVAE is to use DVAE models to estimate the RUL given sensor signals; as is done in my other [repository](https://github.com/StarMarco/DVAE_torch). 
However, with the K-DVAE we realize that the RUL target values are linear (it decays linearly to zero from the current time point), and so we could use a Kalman filter to find the marginal likelihood
$p(y_{1:T}|x_{1:T})$ directly instead of estimating it via. the Evidence Lower Bound (ELBO) loss. There are some works in the literature that do this, but not in the field of machinery prognostics and 
hence, their targets are often nonlinear so they have to resort to a particle filtering framework (which introduces difficulties are resampling is typically not differentiable) or a nonlinear version 
of the Kalman filter (such as the Extended Kalman Filter). Because in the machinery prognostics application our targets are the RUL trajectories which are linear we can just use the linear Kalman filter. 
One may then ask how do we solve a problem that is typically considered a highly nonlinear problem (converting sensor signals to a RUL) using a linear model. The answer is that all our nonlinearity is 
hidden within the control variables (they aren't used as control variables here but in state space modelling they are often referred to as such). 

Hence, in the DVAE we introduced transition and measurement models. In the K-DVAE these are linear Gaussian state space models so that they are compatible with the Kalman filter. 

Transition: 

$$
z_t = F_t z_{t-1} + u_t + w_t
$$

Measurement:

$$
y_t = H_t z_t + d_t + v_t
$$

where $y_t$ is the RUL, $z_t$ is the latent variable, $F_t$ are the transition matrix, $H_t$ is the measurement matrix, $u_t$ and $d_t$ are control variables, and $w_t$ and $v_t$ are noise terms. 
Hence, $u_t$ and $d_t$ are outputs from neural networks which account for the nonlinearity of the problem. From the DVAE we know that the conditional variables should be noncausal and so $u_t$ and $d_t$
must represent the sequence $x_{1:T}$. This makes the K-DVAE a sequence-to-sequence model which outputs a sequence of RUL values $y_{1:T}$ given a sequence of multivariate sensor signals $x_{1:T}$. 
In this case, a bidirectional recurrent neural network is used (you can choose to use a GRU or LSTM both are implemented) and the hidden variable outputs from these represent $x_{1:T}$.  
