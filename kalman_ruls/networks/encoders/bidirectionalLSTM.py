import tensorflow as tf 
from kalman_ruls.networks.encoders.base import EncoderBase

class BidirectionalLSTM(EncoderBase):
    def __init__(self, xdim, hdim, zdim, encode_d=True):
        super().__init__()
        rdim = 1 
        self.rdim = rdim 
        self.xdim = xdim 
        self.hdim = hdim 
        self.zdim = zdim 

        self.u_net = tf.keras.Sequential()
        self.u_net.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hdim, return_sequences=True)))
        self.u_net.add(tf.keras.layers.Dense(zdim))

        if encode_d:    # might not need this for inference models to test ELBO vs Kalman loss functions 
            self.d_net = tf.keras.Sequential()
            self.d_net.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hdim, return_sequences=True)))
            self.d_net.add(tf.keras.layers.Dense(rdim))

    def encode_u(self, xs):
        return self.u_net(xs)

    def encode_d(self, xs):
        return self.d_net(xs)