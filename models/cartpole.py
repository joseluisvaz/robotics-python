from functools import partial

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, jacfwd


class Cartpole(object):
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_scale = 30.0
        self.dt = 0.02  # seconds between state updates

        self.xdim = 4
        self.udim = 1

        self.get_A_matrix = jit(jacfwd(self.__call__, argnums=0))
        self.get_B_matrix = jit(jacfwd(self.__call__, argnums=1))

        self.x_names = [
            "x",
            "x_dot",
            "theta",
            "theta_dot",
        ]
        self.u_names = [
            "F",
        ]

    def get_physical_inputs(self, u):
        return self.force_scale * jnp.tanh(u)

    def plot(self, ax, state, **kwargs):
        x1 = state[0]
        theta = state[2]

        x2 = x1 + self.length * np.sin(theta)
        y2 = self.length * np.cos(theta)

        ax.plot([x1, x2], [0, y2], **kwargs)
        ax.add_patch(
            plt.Circle((x2, y2), 0.01, color=kwargs["c"], alpha=kwargs["alpha"])
        )

    @partial(jit, static_argnums=(0,))
    def __call__(self, state, u):

        force = self.get_physical_inputs(u)[0]

        x, x_dot, theta, theta_dot = state

        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot * theta_dot * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * xacc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * thetaacc
        return jnp.array([x, x_dot, theta, theta_dot])
