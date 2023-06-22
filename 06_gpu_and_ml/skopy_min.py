from skopt.space import Real
from skopt import gp_minimize
from skopt.utils import use_named_args
import modal


image = modal.Image.debian_slim().pip_install("numpy", "scikit-optimize")
stub = modal.Stub(image=image)


# In this example we are minimizing a simple quadratic function
# But this can easily be used to tune hyperparameters of a machine learning model
@stub.function()
def objective(x):
    return (x - 5) ** 2


# We'll create a space with our hyperparameter x
space = [Real(-10, 10, name="x")]


@use_named_args(space)
def decorated_function(**hyperparams):
    return objective.call(hyperparams["x"])


def plot_callback(res):
    print("Iteration {}".format(len(res.func_vals)))
    print("Current minimum {}".format(res.fun))


@stub.local_entrypoint()
async def main():
    res = gp_minimize(decorated_function, space, n_calls=15, callback=[plot_callback], random_state=42)
