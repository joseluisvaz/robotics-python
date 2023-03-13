import jax.numpy as jnp
import numpy as np
from jax import hessian, jit, jacfwd, vmap
from matplotlib import pyplot as plt
from pathlib import Path
import subprocess

from models.bicycle_model import BicycleModelWithInputIntegrators

from dataclasses import dataclass



class iLQRMatrices(object):
    def __init__(self, dynamics, cost_function):
        self.dynamics = dynamics
        self.cost_function = cost_function.get_stage_cost
        self.cost_function_terminal = cost_function.get_terminal_cost

        self.get_fx_batched = jit(vmap(jacfwd(dynamics.__call__, argnums=0)))
        self.get_fu_batched = jit(vmap(jacfwd(dynamics.__call__, argnums=1)))

        self.get_l_batched = jit(vmap(self.cost_function))
        self.get_lx_batched = jit(vmap(jacfwd(self.cost_function, argnums=0)))
        self.get_lxx_batched = jit(vmap(hessian(self.cost_function, argnums=0)))
        self.get_lu_batched = jit(vmap(jacfwd(self.cost_function, argnums=1)))
        self.get_luu_batched = jit(vmap(hessian(self.cost_function, argnums=1)))
        self.get_lux_batched = jit(
            vmap(jacfwd(jacfwd(self.cost_function, argnums=1), argnums=0))
        )

        self.get_l_terminal = jit(vmap(self.cost_function_terminal))
        self.get_lx_terminal = jit(vmap(jacfwd(self.cost_function_terminal)))
        self.get_lxx_terminal = jit(vmap(hessian(self.cost_function_terminal)))

    def get_cost(self, states, actions, params):
        l = self.get_l_batched(states[:-1], actions, params[:-1]).sum()
        ln = self.get_l_terminal(states[-1, None], params[-1, None]).sum()
        return l + ln

    def get_matrices(self, states, actions, params):

        fx = self.get_fx_batched(states[:-1], actions, params[:-1])
        fu = self.get_fu_batched(states[:-1], actions, params[:-1])
        lx = self.get_lx_batched(states[:-1], actions, params[:-1])[..., None]
        lxx = self.get_lxx_batched(states[:-1], actions, params[:-1])
        lu = self.get_lu_batched(states[:-1], actions, params[:-1])[..., None]
        luu = self.get_luu_batched(states[:-1], actions, params[:-1])
        lux = self.get_lux_batched(states[:-1], actions, params[:-1])

        lx_N = self.get_lx_terminal(states[-1, None], params[-1, None])[..., None]
        lxx_N = self.get_lxx_terminal(states[-1, None], params[-1, None])

        return (
            np.array(fx),
            np.array(fu),
            np.concatenate([np.array(lx), np.array(lx_N)]),
            np.concatenate([np.array(lxx), np.array(lxx_N)]),
            np.array(lu),
            np.array(luu),
            np.array(lux),
        )


@dataclass
class iLQRConfig:
    horizon: int
    iters: int
    mu: float = 0.5
    tol: float = 1e-6


class iLQR(object):
    def __init__(self, config, dynamics, cost_function):
        self.config = config
        self.dynamics = dynamics
        self.cost_function = cost_function
        self.ilqr_matrices = iLQRMatrices(dynamics, cost_function)
        self.params_function = None

    def backward_pass(self, x, u, params, debug=False):
        fx, fu, lx, lxx, lu, luu, lux = self.ilqr_matrices.get_matrices(x, u, params)
        
        if debug:
            print("fx: \n", fx[0].T)
            print("fu: \n", fu[0].T)
            print("lx: \n", lx[0].T)
            print("lxx: \n", lxx[0].T)
            print("lu: \n", lu[0].T)
            print("luu: \n", luu[0].T)
            print("lux: \n", lux[0].T)
            print("lux: \n", lux[0].shape)

        # Create control vector and matrices
        K = np.zeros((self.config.horizon, self.dynamics.udim, self.dynamics.xdim))
        k = np.zeros((self.config.horizon, self.dynamics.udim, 1))

        I = np.eye(self.dynamics.xdim)

        # Get terminal values.
        Vx = lx[-1]
        Vxx = lxx[-1]

        # Iterate backwards and via dynamic programming compute the feedback policy tuple (K, k)
        for t in range(self.config.horizon - 1, -1, -1):
            Qx = lx[t] + fx[t].T @ Vx
            Qu = lu[t] + fu[t].T @ Vx
            Qxx = lxx[t] + fx[t].T @ Vxx @ fx[t]
            Quu = luu[t] + fu[t].T @ (Vxx + self.config.mu * I) @ fu[t]
            Qux = lux[t] + fu[t].T @ (Vxx + self.config.mu * I) @ fx[t]

            #K[t] = -np.linalg.solve(Quu, Qux)
            #k[t] = -np.linalg.solve(Quu, Qu)
            Quu_inv = np.linalg.inv(Quu)
            K[t] = -Quu_inv @ Qux
            k[t] = -Quu_inv @ Qu

            Vx = Qx + K[t].T @ Quu @ k[t] + K[t].T @ Qu + Qux.T @ k[t]
            Vxx = Qxx + K[t].T @ Quu @ K[t] + K[t].T @ Qux + Qux.T @ K[t]
            Vxx = 0.5 * (Vxx + Vxx.T)  # force symmetry

            if debug:
                print("Qu: \n", Qu.T)
                print("lu: \n", lu[t].T)
                print("fu: \n", fu[t].T)


                print("Qx: \n", Qx.T)
                print("Qxx: \n", Qxx)
                print("Qux: \n", Qux)
                print("lux: \n", lux[t])
                print("fu: \n", fu[t])
                print("fx: \n", fx[t])


                print("Quu: \n", Quu)
                print("Vx: \n", Vx.T)
                print("Vxx: \n", Vxx)
                print("K: \n", K[t])
                print("k: \n", k[t])

        return K, k

    def forward_pass(self, x_0, x_bar, u_bar, K, k, alpha, params):
        x = np.zeros_like(x_bar)
        u = np.zeros_like(u_bar)

        x[0] = x_0
        for t in range(self.config.horizon):
            u[t] = u_bar[t] + alpha * k[t].squeeze() + K[t] @ (x[t] - x_bar[t])
            x[t + 1] = self.dynamics(x[t], u[t], params[t])
        return x, u

    def rollout(self, x_0, u, params):
        x = np.zeros((u.shape[0] + 1, self.dynamics.xdim))
        x[0] = x_0
        for k in range(x.shape[0] - 1):
            x[k + 1] = self.dynamics(x[k], u[k], params[k])
        return x

    def solve(
        self,
        x_0,
        warmstart=None,
        callback=None,
        debug=False,
        warmstart_x=None,
    ):
        # Initialize algorithm with a zero policy or the warmstart policy.
        u_ = np.zeros((self.config.horizon, self.dynamics.udim))
        u = warmstart if warmstart is not None else u_

        # TODO: setup params size
        if self.params_function is None:
            self.params_function = lambda _x, _u : np.zeros((self.config.horizon + 1, self.dynamics.udim))

        # TODO: create an initial params function that takes as input the warmstart state sequence and not the input sequence.
        # This will help for curviilnear optimization. For now initialize with curvature = 0
        params = np.zeros((self.config.horizon + 1, self.dynamics.udim))
        if warmstart_x is not None and warmstart is not None:
            params = self.params_function(warmstart_x, warmstart)


        # Perform a first rollout to get the first state sequence
        x = self.rollout(x_0, u, params)
        for i in range(self.config.iters):
            
            # Get parameters
            params = self.params_function(x, u)

            # Perform a backward pass using dynamic programming
            K, k = self.backward_pass(x, u, params)
            J = self.ilqr_matrices.get_cost(x, u, params)

            if debug:
                print("iter: ", i, " cost: ", J)

            # Backtracking linesearch for a decreasing cost
            alphas = 0.5 ** np.arange(10)
            for alpha in alphas:
                # Perform the forward pass and compute the cost.
                x_new, u_new = self.forward_pass(x_0, x, u, K, k, alpha, params)
                J_new = self.ilqr_matrices.get_cost(x_new, u_new, params)
                if J_new < J:
                    x, u = x_new, u_new
                    # if we stopped improving terminate early.
                    if np.abs((J - J_new) / J) < self.config.tol:
                        return x, u
                    # Update the cost
                    J = J_new
                    break
            if callback is not None:
                callback(i, x, u)
        return x, u

    def __call__(self, obs, warmstart=None):
        return self.solve(obs, warmstart)[1][0]


class ParkingCostFunction(object):
    def __init__(self, goal):
        self.goal = goal
        self.Q = 0.00001 * np.diag(np.array([1.0, 1.0, 10.0, 1.0, 100.0, 100.0]))
        self.Q_N = 100 * np.diag(np.array([1.0, 1.0, 100.0, 1.0, 100.0, 100.0]))
        self.R = 0.1 * np.eye(2)

    def get_stage_cost(self, x, u, params):
        return (x - self.goal).T @ self.Q @ (x - self.goal) + u.T @ self.R @ u

    def get_terminal_cost(self, x, params):
        return (x - self.goal).T @ self.Q_N @ (x - self.goal)


def plot_trajectory(ax, model, trajectory, **kwargs):
    alphas = np.linspace(0, 1.0, len(trajectory))

    base_alpha = kwargs["alpha"]
    for alpha, state in zip(alphas[::3], trajectory[::3]):
        kwargs["alpha"] = alpha * base_alpha
        model.plot(ax, state, **kwargs)


def visualize_optimization(starting_state, policy, dynamics, make_gif=False):
    xdim = policy.dynamics.xdim
    udim = policy.dynamics.udim

    fig1, x_axis = plt.subplots(xdim, 1, figsize=(12, 16))
    fig2, u_axis = plt.subplots(udim, 1, figsize=(12, 8))
    fig1.tight_layout()
    fig2.tight_layout()
    if not make_gif:
        fig3, xy_axis = plt.subplots(1, 1, figsize=(8, 8))
        fig3.tight_layout()

    def callback_fn(iter, x, u):
        alpha = 0.2

        if make_gif:
            _, xy_axis = plt.subplots(1, 1, figsize=(8, 8))
            policy.dynamics.plot(xy_axis, policy.cost_function.goal, c="r", alpha=0.5)
            plot_trajectory(xy_axis, policy.dynamics, x, c="b", alpha=0.5)
            xy_axis.set_xlim(-50, 50)
            xy_axis.set_ylim(-50, 50)
            plt.savefig(str(Path.cwd() / f"iter{iter}.jpg"), dpi=300)

        u_solution_physical = dynamics.get_physical_inputs(u)
        for ax, seq, y_label in zip(x_axis, x.T, dynamics.x_names):
            ax.plot(seq, c="r", linestyle="dashed", alpha=alpha)
            ax.set_xlabel("Timesteps")
            ax.set_ylabel(y_label)
        for ax, seq, y_label in zip(u_axis, u_solution_physical.T, dynamics.u_names):
            ax.plot(seq, c="r", linestyle="dashed", alpha=alpha)
            ax.set_xlabel("Timesteps")
            ax.set_ylabel(y_label)

    # Warmstart 0.1 mps^2 and 0.01 steering angle rate.
    warmstart = np.array([0.1, 0.01]) * np.ones(
        (policy.config.horizon, policy.dynamics.udim)
    )
    x_solution, u_solution = policy.solve(
        starting_state, warmstart=warmstart, callback=callback_fn, debug=True
    )

    u_solution_physical = dynamics.get_physical_inputs(u_solution)
    for ax, seq in zip(x_axis, x_solution.T):
        ax.plot(seq, c="b", linewidth=3)
    for ax, seq in zip(u_axis, u_solution_physical.T):
        ax.plot(seq, c="b", linewidth=3)

    if make_gif:
        subprocess.call(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                "10",
                "-i",
                "iter%01d.jpg",
                "-r",
                "30",
                "vehicle.gif",
            ]
        )

        for path in Path.cwd().glob("*.jpg"):
            if path.is_file():
                path.unlink()
    else:
        plot_trajectory(xy_axis, policy.dynamics, x_solution, c="b", alpha=1.0)
        plt.show()


def visualize_benchmark(policy):
    _, xy_axis = plt.subplots(1, 1, figsize=(8, 8))

    initial_states = [
        np.array([-20.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([-0.0, -20.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([20.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([-0.0, 20.0, 0.0, 0.0, 0.0, 0.0]),
    ]

    colors = ["C0", "C1", "C2", "C3"]

    policy.dynamics.plot(xy_axis, policy.cost_function.goal, c="r", alpha=1.0)
    for x_0, c in zip(initial_states, colors):
        x_solution, _ = policy.solve(x_0, debug=True)
        plot_trajectory(xy_axis, policy.dynamics, x_solution, c=c, alpha=1.0)
    xy_axis.axis("equal")
    plt.show()


if __name__ == "__main__":
    dynamics = BicycleModelWithInputIntegrators()
    goal = jnp.array([0.5, 0.0, np.pi / 2.0, 0.0, 0.0, 0.0])
    cost_function = ParkingCostFunction(goal)

    config = iLQRConfig(horizon=200, iters=20)
    policy = iLQR(config, dynamics, cost_function)

    # x0 = np.array([-0.0, -20.0, 0.0, 0.0, 0.0, 0.0])
    # visualize_optimization(x0, policy, dynamics)
    visualize_benchmark(policy)
