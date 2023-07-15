import numpy as np 

def sliding_window(x, T):
    """
    Use a sliding window of length T, over the sequential data 

    Inputs:
        x (array): sequence to break into time windows, size (seq, dim)
        T (int): window size 

    Outputs:
        x_T (array): batch of time-windowed sequences, size (bs, seq, dim) 
    """
    if x.shape[0] < T:
        return np.expand_dims(x, 0)
    seq = x.shape[0] - T

    sub_windows = (
        np.expand_dims(np.arange(T), 0) +
        np.expand_dims(np.arange(seq + 1), 0).T
    )
    return x[sub_windows]

def diag_idx(array, idx):
    """
    returns the diagonal elements of a matrix 
    e.g. 
    
        [1, 2, 3
    A =  4, 5, 6
         7, 8, 9
         10,11,12]

    diag_idx(A, 1) = np.array([4, 2])
    diag_idx(A, 0) = np.array([1])
    diag_idx(A, 3) = np.array([10, 8, 6])
    """
    def relu(x):
        if x <= 0:
            return 0 
        else:
            return x 

    n = array.shape[0]
    m = array.shape[1]
    idx = int(idx)

    #if n == 1:
        #return x[]
    if n < m:
        a = n 
        n = m
        m = a  
        array = array.swapaxes(0,1) # shape (higher dim, lower dim)

    i_start = min(idx, n-1)
    j_start = relu(idx - (n-1))
    if idx >= m:
        j_end = m 
    else: 
        j_end = idx+1

    j_idxs = np.arange(j_start, j_end, dtype=int)
    size = j_idxs.shape[0]
    
    i_idxs = np.linspace(i_start, i_start-size+1, size, dtype=int)

    return array[i_idxs, j_idxs]

def win_to_seq(x_T):
    """
    Given a bunch of time windowed batches in order
    i.e. 
    x_T =  [[0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5]
            etc...]
    returns the array that is a single sequence 
    i.e. 
    x = [0,1,2,3,4,5,...]

    * Note the overlapping terms are averaged

    e.g.
    diag_idx(x_T, 1).mean() = np.array([1, 1]).mean() = 1. 
    In this case we have the same values at each overlapping time point, but in 
    some cases (such as non-causal network estimates, this may not be the case)
    """
    bs = x_T.shape[0]
    T = x_T.shape[1]
    seq = bs + T - 1 
    
    xts = [diag_idx(x_T, i).mean(0) for i in range(seq)]
    x = np.stack(xts)
    return x 