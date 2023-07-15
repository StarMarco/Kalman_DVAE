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
def kf(lgssm, observations, us, ds):
    Fs, Qs, Hs, Rs, P0, m0 = lgssm

    def body(carry, inp):
        m, P = carry
        y, u, d, F, Q, H, R = inp

        m = mv(F, m) + u
        P = F @ mm(P, F, transpose_b=True) + Q
        P = 0.5 * (P + tf.linalg.matrix_transpose(P))

        S = H @ mm(P, H, transpose_b=True) + R

        chol = tf.linalg.cholesky(S)
        Kt = tf.linalg.cholesky_solve(chol, H @ P)

        m = m + mv(Kt, y - mv(H, m) - d, transpose_a=True)
        P = P - mm(Kt, S, transpose_a=True) @ Kt
        return m, P

    fms, fPs = tf.scan(body, (observations, us, ds, Fs, Qs, Hs, Rs), (m0, P0))
    return fms, fPs

# --------------------------------------------------------------
# Parallel
# --------------------------------------------------------------
@partial(tf.function, experimental_relax_shapes=True)
def first_filtering_element(F, H, Q, R, m0, P0, y, u, d):

    m1 = mv(F, m0) + u 
    P1 = F @ mm(P0, F, transpose_b=True) + Q
    S1 = H @ mm(P1, H, transpose_b=True) + R
    S1_chol = tf.linalg.cholesky(S1)
    K1t = tf.linalg.cholesky_solve(S1_chol, H @ P1)
    
    A = tf.zeros_like(F) * tf.ones_like(P0)
    b = m1 + mv(tf.linalg.matrix_transpose(K1t), y - mv(H, m1) - d)
    C = P1 - mm(tf.linalg.matrix_transpose(K1t), S1) @ K1t

    S = H @ mm(Q, H, transpose_b=True) + R
    chol = tf.linalg.cholesky(S)
    HF = H @ F
    eta = mv(HF, 
             tf.squeeze(tf.linalg.cholesky_solve(chol, tf.expand_dims(y - mv(H,u) - d, -1)), -1), 
             transpose_a=True)
    J = mm(HF, tf.linalg.cholesky_solve(chol, H @ F), transpose_a=True) * tf.ones_like(P0)
    return A, b, C, J, eta

@partial(tf.function, experimental_relax_shapes=True)
def generic_filtering_element(F, H, Q, R, m0, P0, y, u, d):

    S = H @ mm(Q, H, transpose_b=True) + R
    chol = tf.linalg.cholesky(S)

    Kt = tf.linalg.cholesky_solve(chol, H @ Q)
    A = (F - mm(Kt, H, transpose_a=True) @ F) * tf.ones_like(P0)
    b = u + mv(Kt, y - mv(H,u) - d, transpose_a=True)
    C = (Q - mm(Kt, H, transpose_a=True) @ Q) * tf.ones_like(P0)

    HF = H @ F
    eta = mv(HF, 
             tf.squeeze(tf.linalg.cholesky_solve(chol, tf.expand_dims(y - mv(H,u) - d, -1)), -1),   
             transpose_a=True)
    
    J = mm(HF, tf.linalg.cholesky_solve(chol, HF), transpose_a=True) * tf.ones_like(P0)
    return A, b, C, J, eta

@partial(tf.function, experimental_relax_shapes=True)
def make_associative_filtering_elements(Fs, H, Q, R, m0, P0, observations, us, ds):
    first_elems = first_filtering_element(Fs[0], H[0], Q[0], R[0], m0, P0, observations[0], us[0], ds[0])
    generic_elems = tf.vectorized_map(lambda inp: generic_filtering_element(inp[0], inp[1], inp[2], inp[3], m0, P0, inp[4], inp[5], inp[6]), 
                                      (Fs[1:], H[1:], Q[1:], R[1:], observations[1:], us[1:], ds[1:]), fallback_to_while_loop=False)
    return tuple(tf.concat([tf.expand_dims(first_e, 0), gen_es], 0) 
                 for first_e, gen_es in zip(first_elems, generic_elems)) #type:ignore

@partial(tf.function, experimental_relax_shapes=True)
def filtering_operator(elems):
    elem1, elem2 = elems

    A1, b1, C1, J1, eta1 = elem1
    A2, b2, C2, J2, eta2 = elem2
    dim = A1.shape[-1]
    I = tf.eye(dim, dtype=A1.dtype, )

    temp = tf.linalg.solve(I + C1 @ J2, tf.linalg.matrix_transpose(A2), adjoint=True)
    A = mm(tf.linalg.matrix_transpose(temp), A1)
    b = mv(tf.linalg.matrix_transpose(temp), b1 + mv(C1, eta2)) + b2
    C = mm(tf.linalg.matrix_transpose(temp), mm(C1, tf.linalg.matrix_transpose(A2))) + C2

    temp = tf.linalg.solve(I + J2 @ C1, A1, adjoint=True)
    eta = mv(tf.linalg.matrix_transpose(temp), eta2 - mv(J2, b1)) + eta1
    J = mm(tf.linalg.matrix_transpose(temp), J2 @ A1) + J1

    return A, b, C, J, eta

@partial(tf.function, experimental_relax_shapes=True)
def pkf(lgssm, observations, us, ds, max_parallel=10000):
    Fs, Qs, Hs, Rs, P0, m0 = lgssm
    initial_elements = make_associative_filtering_elements(Fs, Hs, Qs, Rs, m0, P0, observations, us, ds)
    def vectorized_operator(a, b):
        return tf.vectorized_map(filtering_operator, (a, b), fallback_to_while_loop=False)
    final_elements = tfp.math.scan_associative(vectorized_operator, 
                                               initial_elements, 
                                               max_num_levels=math.ceil(math.log2(max_parallel)))
    return final_elements[1], final_elements[2]
