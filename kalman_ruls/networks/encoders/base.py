import abc 
import tensorflow as tf 

class EncoderBase(tf.keras.Model, metaclass=abc.ABCMeta):
    xdim: int 
    zdim: int 
    hdim: int 

    @abc.abstractmethod
    def encode_u(self, xs: tf.Tensor):
        """
        Encodes the input "control" variables to have the same dimension as the 
        latent space dynamics 

        Inputs: 
            xs (tensor): input sensor signal, size (bs, seq, xdim)

        Outputs:
            us (tensor): control variables used in z = Fz + u, size (bs, seq, zdim)
        """

    @abc.abstractmethod
    def encode_d(self, xs: tf.Tensor):
        """
        Encodes the input "control" variables to have the same dimension as the 
        inputs/measurements 

        Inputs:
            xs (tensor): input sensor signal, size (bs, seq, xdim)

        Outputs:
            ds (tensor): control variables used in r = Hz + d, size (bs, seq, rdim)        
        """
