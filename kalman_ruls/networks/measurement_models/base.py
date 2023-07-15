import abc 
import tensorflow as tf 

class MeasurementBase(tf.keras.Model, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create_Hs(self, xs: tf.Tensor):
        """
        Computes the measurement matrices over a time period, and is compatable with 
        batched varaibles 
        """

    @abc.abstractmethod
    def create_Rs(self, xs: tf.Tensor):
        """
        Computes the measurement covariance over a time periods, and is compatable with 
        batched variables  
        """
