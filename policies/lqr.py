from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax.numpy.linalg import inv, norm

def lqr_infinite_horizon(A, B, Q, R):
    """Iteratively compute an lqr controller"""
    dx, du = A.shape[0], B.shape[1]
    P, K_current = jnp.eye(dx), jnp.zeros((du, dx))

    while True:
        K_new = -inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        P = Q + (K_new.T @ R @ K_new) + (A + B @ K_new).T @ P @ (A + B @ K_new)
        if norm(K_current - K_new, 2.0) <= 1e-4:
            break
        K_current = K_new

    return K_new, P


def lqr_finite_horizon(A_seq, B_seq, Q_seq, R_seq):
    """Iteratively compute an lqr controller"""
    assert A_seq.shape[0] == B_seq.shape[0] == Q_seq.shape[0] == R_seq.shape[0]

    P = np.eye(A_seq.shape[2])
    K_seq = np.zeros_like(B_seq.transpose((0, 2, 1)))
    P_seq = np.zeros_like(Q_seq)

    _, P = lqr_infinite_horizon(A_seq[-1], B_seq[-1], Q_seq[-1], R_seq[-1])

    n_steps = A_seq.shape[0]
    for i in range(n_steps - 1, -1, -1):
        A, B, Q, R = A_seq[i], B_seq[i], Q_seq[i], R_seq[i]

        K = -inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        P = Q + (K.T @ R @ K) + (A + B @ K).T @ P @ (A + B @ K)

        K_seq[i] = K
        P_seq[i] = P

    return K_seq, P_seq


class LQR(object):
    @dataclass
    class Config:
        x_ref: jnp.ndarray
        u_ref: jnp.ndarray
        Q: jnp.ndarray
        R: jnp.ndarray

    def __init__(self, config, dynamics):
        self.config = config
        self.K, _ = lqr_infinite_horizon(
            dynamics.get_A_matrix(config.x_ref, config.u_ref),
            dynamics.get_B_matrix(config.x_ref, config.u_ref),
            config.Q,
            config.R,
        )

    def __call__(self, x):
        return self.K @ (x - self.config.x_ref) + self.config.u_ref
