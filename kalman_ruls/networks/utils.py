import tensorflow as tf 
import numpy as np 

def alpha_coverage(y, y_mean, y_std, number_stds=1.96):
    """
    Counting the amount of points within the confidence interval (CI). The closer the number is to the CI 
    the better e.g. if we take the mean of these alpha_coverages and get 0.95 for a 95% CI then this is ideal.

    Found in the paper from Mitici, de Pater, Barros, Zeng (2023),
    "Dynamic predictive maintenance for multiple components using data-driven probabilist RUL prognostics:
    The case of turbofan engines"

    Inputs: 
        y (tensor): actual target, size (seq, dim)
        y_mean (tensor): estimated mean of target, size (seq, dim)
        y_std (tensor): estimated standard deviation of target, size (seq, dim)
        number_stds (float): the number of standard deviations away that represent 
        the confidence interval

    Outputs:
        alpha_coverage: a metric which counts if the target falls within the 
            confidence bounds, size (seq, dim)
    """
    lower_bo = y_mean - y_std * number_stds
    upper_bo = y_mean + y_std * number_stds
    l = (lower_bo <= y).astype(int)
    u = (y <= upper_bo).astype(int)
    bound = l + u

    coverage = (bound == 2).astype(float)
    return coverage 

def alpha_mean(y_mean, y_std, number_stds=1.96):
    """
    The width of the confidence intervals 
    also found in the paper from Mitici, de Pater, Barros, Zeng (2023),
    "Dynamic predictive maintenance for multiple components using data-driven probabilist RUL prognostics:
    The case of turbofan engines"

    Inputs: 
        y (tensor): actual target, size (seq, dim)
        y_mean (tensor): estimated mean of target, size (seq, dim)
        y_std (tensor): estimated standard deviation of target, size (seq, dim)
        number_stds (int): the number of standard deviations away that represent 
        the confidence interval
    Outputs:
        alpha_mean: a metric for the width of the confidence intervals, size (seq, dim)
    """
    lower_bo = y_mean - y_std * number_stds
    upper_bo = y_mean + y_std * number_stds

    return upper_bo - lower_bo
    
def score_func(x, t):
    '''
    Score function from NASA C-MAPSS data pdf (comes with C-MAPSS dataset download)
    *Note bs = 1 in testing (ONLY USE THIS FUNCTION IN TESTING)

    INPUTS: 
        x (tensor): RUL estimates 
        t (tensor): True RUL values 
    
    OUTPUTS: 
        score (tensor):
            when,
                e = x-t < 0, score = exp(-e/13) - 1
                e = x-t > 0, score = exp(e/10) - 1

        The negetive values are penalized less then the postive values. 
        Hence, when e < 0 true RUL is higher than the estimate so our estimate 
        says the component will fail before it actually will (estimate is conservative)
        so it is less penalized. 
    '''

    error = (x - t)       # error = estimated RUL - True RUL
    error_less = (error < 0) * error
    error_more = (error >= 0) * error

    score_less = np.exp(-error_less / 13) - 1
    score_more = np.exp(error_more / 10) - 1 

    score = score_less + score_more
    return score 

class MLP(tf.keras.Model):
    def __init__(self, inputs, hidden, output, activation='relu'):
        # even though I don't need the input dims I find it useful for code readability 
        super().__init__()
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden, activation=activation),
                tf.keras.layers.Dense(hidden, activation=activation),
                tf.keras.layers.Dense(output)
            ]
        )
    def call(self, x):
        return self.net(x)