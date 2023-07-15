import tensorflow as tf 
import tensorflow_probability as tfp 

from functools import partial
from collections import namedtuple
import numpy as np
import math

# Largly taken from https://github.com/EEA-sensors/sequential-parallelization-examples/tree/main/python/temporal-parallelization-bayes-smoothers 
# which implements the parallel Kalman filter and smoother described in the paper "Temporal Parallelization of Bayesian Smoothers" by S. Särkkä and Á. F. García-Fernández
# in IEEE Transactions on Automatic Control, vol. 66, no. 1, pp. 299-306, Jan. 2021, doi: 10.1109/TAC.2020.2976316.

mm = tf.linalg.matmul
mv = tf.linalg.matvec
LGSSM = namedtuple("LGSSM", ["Fs", "Q", "H", "R", "P0", "m0"])

# --------------------------------------------------------------
# Sequential 
# --------------------------------------------------------------
@partial(tf.function, experimental_relax_shapes=True)
def ks(lgssm, ms, Ps, us):
    Fs, Qs, _, _, _, _ = lgssm

    def body(carry, inp):
        m, P, u, F, Q = inp
        sm, sP = carry

        pm = mv(F, m) + u
        pP = F @ mm(P, F, transpose_b=True) + Q

        chol = tf.linalg.cholesky(pP)
        Ct = tf.linalg.cholesky_solve(chol, F @ P)

        sm = m + mv(Ct, (sm - pm), transpose_a=True)
        sP = P + mm(Ct, sP - pP, transpose_a=True) @ Ct
        return sm, sP

    (sms, sPs) = tf.scan(body, (ms[:-1], Ps[:-1], us[:-1], Fs[1:], Qs[1:]), (ms[-1], Ps[-1]), reverse=True)
    sms = tf.concat([sms, tf.expand_dims(ms[-1], 0)], 0)
    sPs = tf.concat([sPs, tf.expand_dims(Ps[-1], 0)], 0)
    return sms, sPs

# --------------------------------------------------------------
# Parallel
# --------------------------------------------------------------
@partial(tf.function, experimental_relax_shapes=True)
def last_smoothing_element(m, P):
    return tf.zeros_like(P), m, P

@partial(tf.function, experimental_relax_shapes=True)
def generic_smoothing_element(F, Q, m, P, u):
 
    Pp = F @ mm(P, tf.linalg.matrix_transpose(F)) + Q

    chol = tf.linalg.cholesky(Pp)
    E  = tf.linalg.matrix_transpose(tf.linalg.cholesky_solve(chol, F @ P))
    g  = m - mv(E @ F, m) - mv(E, u)
    L  = P - E @ mm(Pp, tf.linalg.matrix_transpose(E))
    return E, g, L

@partial(tf.function, experimental_relax_shapes=True)
def make_associative_smoothing_elements(Fs, Q, filtering_means, filtering_covariances, us):
    last_elems = last_smoothing_element(filtering_means[-1], filtering_covariances[-1])
    generic_elems = tf.vectorized_map(lambda inp: generic_smoothing_element(inp[0], inp[1], inp[2], inp[3], inp[4]), 
                                      (Fs[1:], Q[1:], filtering_means[:-1], filtering_covariances[:-1], us[:-1]),
                                      fallback_to_while_loop=False)
    return tuple(tf.concat([gen_es, tf.expand_dims(last_e, 0)], axis=0) 
                 for gen_es, last_e in zip(generic_elems, last_elems)) 

@partial(tf.function, experimental_relax_shapes=True)
def smoothing_operator(elems):
    elem1, elem2 = elems
    E1, g1, L1 = elem1
    E2, g2, L2 = elem2

    E = E2 @ E1
    g = mv(E2, g1) + g2
    L = E2 @ mm(L1, E2, transpose_b=True) + L2

    return E, g, L

@partial(tf.function, experimental_relax_shapes=True)
def pks(lgssm, filtered_means, filtered_covariances, us, max_parallel=10000):
    Fs, Q, _, _, _, _ = lgssm
    initial_elements = make_associative_smoothing_elements(Fs, Q, filtered_means, filtered_covariances, us)
    reversed_elements = tuple(tf.reverse(elem, axis=[0]) for elem in initial_elements)   
    def vectorized_operator(a, b):
        return tf.vectorized_map(smoothing_operator, (a, b), fallback_to_while_loop=False)
    final_elements = tfp.math.scan_associative(vectorized_operator, 
                                               reversed_elements, 
                                               max_num_levels=math.ceil(math.log2(max_parallel)))
    return tf.reverse(final_elements[1], axis=[0]), tf.reverse(final_elements[2], axis=[0])