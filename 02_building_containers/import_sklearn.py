# # Install scikit-learn in a custom image
#
# This builds a custom image which installs the sklearn (scikit-learn) Python package in it.
# It's an example of how you can use packages, even if you don't have them installed locally.
#
# First, the imports

import time

import modal

# Next, define an stub, with a custom image that installs `sklearn`.

stub = modal.Stub(
    image=modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(["sklearn"])
)

# The `stub.is_inside()` lets us conditionally run code in the global scope.
# This is needed because we might not have sklearn and numpy installed locally,
# but we know they are installed inside the custom image. `stub.is_inside()`
# will return `False` when we run this locally, but `True` when it runs in the cloud.

if stub.is_inside():
    import numpy as np
    from sklearn import datasets, linear_model

# Now, let's define a function that uses one of scikit-learn's built-in datasets
# and fits a very simple model (linear regression) to it


@stub.function
def run():
    print("Inside run!")
    t0 = time.time()
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    diabetes_X = diabetes_X[:, np.newaxis, 2]
    regr = linear_model.LinearRegression()
    regr.fit(diabetes_X, diabetes_y)
    return time.time() - t0


# Finally, let's trigger the run locally. We also time this. Note that the first time we run this,
# it will build the image. This might take 1-2 min. When we run this subsequent times, the image
# is already build, and it will run much much faster.


if __name__ == "__main__":
    t0 = time.time()
    with stub.run():
        t = run()
        print("Function time spent:", t)
    print("Full time spent:", time.time() - t0)
