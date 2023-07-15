import abc 
import tensorflow as tf 

class TransitionBase(tf.keras.Model, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create_Fs(self, xs: tf.Tensor):
        """
        Computes the transition matrices over a time period, and is compatable with 
        batched varaibles 
        """

    @abc.abstractmethod
    def create_Qs(self, xs: tf.Tensor):
        """
        Computes the process covariance over a time periods, and is compatable with 
        batched variables  
        """
