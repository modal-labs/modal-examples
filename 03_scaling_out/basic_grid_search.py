# # Hyperparameter search
#
# This example showcases a simple grid search in one dimension, where we try different
# parameters for a model and pick the one with the best results on a holdout set.
#
# ## Defining the image
#
# First, let's build a custom image and install scikit-learn in it.

import modal

app = modal.App(
    "example-basic-grid-search",
    image=modal.Image.debian_slim().pip_install("scikit-learn~=1.5.0"),
)

# ## The Modal function
#
# Next, define the function. Note that we use the custom image with scikit-learn in it.
# We also take the hyperparameter `k`, which is how many nearest neighbors we use.


@app.function()
def fit_knn(k):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    clf = KNeighborsClassifier(k)
    clf.fit(X_train, y_train)
    score = float(clf.score(X_test, y_test))
    print("k = %3d, score = %.4f" % (k, score))
    return score, k


# ## Parallel search
#
# To do a hyperparameter search, let's map over this function with different values
# for `k`, and then select for the best score on the holdout set:


@app.local_entrypoint()
def main():
    # Do a basic hyperparameter search
    best_score, best_k = max(fit_knn.map(range(1, 100)))
    print("Best k = %3d, score = %.4f" % (best_k, best_score))
