import numpy as np
from functools import partial

import jax.numpy as jnp
from jax import jit, jacfwd


class BicycleModel(object):
    def __init__(self):
        self.dt = 0.1
        self.lf = 2.5
        self.lr = 1.5
        self.wheelbase = 1.5

        self.a_scale_mpss = 2.0
        self.steering_scale_rad = np.deg2rad(20.0)

        self.x_names = [
            "x",
            "y",
            "heading",
            "speed",
        ]
        self.u_names = [
            "accel",
            "steering",
        ]

        self.xdim = len(self.x_names)
        self.udim = len(self.u_names)
        self.get_A_matrix = jit(jacfwd(self.__call__, argnums=0))
        self.get_B_matrix = jit(jacfwd(self.__call__, argnums=1))

    def get_physical_inputs(self, u):
        scale = np.array([self.a_scale_mpss, self.steering_scale_rad])
        return scale * jnp.tanh(u)

    def plot(self, ax, state, **kwargs):
        x = state[0]
        y = state[1]
        heading = state[2]

        w = self.wheelbase
        lr = self.lr
        lf = self.lf
        br = 0.5  # distance from rear axle to rear bumper
        bf = 0.5  # distance from front axle to front bumper
        dr = br  # distance to rear
        df = lf + lr + bf  # distance to front

        # builds corners for bounding box, plus a triangle to specify heading
        x_ = np.array([-dr, df, df, -dr, -dr, df, dr])
        y_ = np.array([w, w, -w, -w, w, 0.0, -w])

        xs = x + x_ * np.cos(heading) - y_ * np.sin(heading)
        ys = y + x_ * np.sin(heading) + y_ * np.cos(heading)
        ax.plot(xs, ys, **kwargs)

    @partial(jit, static_argnums=(0,))
    def __call__(self, state, input):
        input = self.get_physical_inputs(input)
        a, steering = input[0], input[1]

        x, y, theta, v = state

        x = x + self.dt * (v * jnp.cos(theta))
        y = y + self.dt * (v * jnp.sin(theta))
        theta = theta + self.dt * (v / (self.lf + self.lr) * jnp.tan(steering))
        v = v + self.dt * a

        return jnp.array([x, y, theta, v])


class BicycleModelWithInputIntegrators(BicycleModel):
    def __init__(self):
        super().__init__()

        self.jerk_scale = 1.0
        self.dsteering_scale = 0.1

        self.x_names = [
            "x",
            "y",
            "heading",
            "speed",
            "accel",
            "steer",
        ]
        self.u_names = [
            "jerk",
            "dsteer",
        ]

        self.xdim = len(self.x_names)
        self.udim = len(self.u_names)
        self.get_A_matrix = jit(jacfwd(self.__call__, argnums=0))
        self.get_B_matrix = jit(jacfwd(self.__call__, argnums=1))

    def get_physical_inputs(self, u):
        scale = np.array([self.jerk_scale, self.dsteering_scale])
        return scale * jnp.tanh(u)

    @partial(jit, static_argnums=(0,))
    def __call__(self, state, input):
        input = self.get_physical_inputs(input)
        jerk, dsteering = input[0], input[1]

        x, y, theta, v, a, steering = state

        x = x + self.dt * (v * jnp.cos(theta))
        y = y + self.dt * (v * jnp.sin(theta))
        theta = theta + self.dt * (v / (self.lf + self.lr) * jnp.tan(steering))
        v = v + self.dt * a
        a = a + self.dt * jerk
        steering = steering + self.dt * dsteering

        return jnp.array([x, y, theta, v, a, steering])
