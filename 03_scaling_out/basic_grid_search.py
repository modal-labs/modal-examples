# # Hyperparameter tuning with grid search
#
# This example showcases a very simplistic grid search in 1-D where we try different parameters for a model
# and pick the one with the best results on a holdout set.

import modal

# ## Defining the image
#
# First, let's build a custom image and install scikit-learn in it.

stub = modal.Stub(image=modal.Image.debian_slim().pip_install(["sklearn"]))

# ## Conditionally importing in the global scope
#
# The `image.inside()` function returns `False` when it runs locally, and `True` when it runs inside the
# image in the cloud. This is needed so that we can run our script even if we don't have scikit-learn
# installed locally.

if stub.is_inside():
    from sklearn.datasets import make_moons
    from sklearn.neighbors import KNeighborsClassifier

    X_train, y_train = make_moons(10000, noise=0.7, random_state=0)
    X_test, y_test = make_moons(1000, noise=0.7, random_state=1)
else:
    print("Not importing scikit-learn")


# ## The Modal function
#
# Next, define the function. Note that we use the custom image with scikit-learn in it.
# We also take the hyperparameter `k` which is how many nearest neighbors we use.


@stub.function
def fit_knn(k):
    clf = KNeighborsClassifier(k)
    clf.fit(X_train, y_train)
    score = float(clf.score(X_test, y_test))
    print("k = %3d, score = %.4f" % (k, score))
    return score, k


# ## Hyperparameter search
#
# To do a hyperparameter search, let's map over this function with lots of different values
# for `k`, and then pick whichever `k` has the best score on the holdout set:

if __name__ == "__main__":
    with stub.run():
        # Do a basic hyperparameter search
        best_score, best_k = max(fit_knn.map(range(1, 100)))
        print("Best k = %3d, score = %.4f" % (best_k, best_score))
