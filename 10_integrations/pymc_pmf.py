# ---
# cmd: ["modal", "run", "10_integrations/pymc_pmf.py::pymc_stub"]
# ---
# # Probabilistic programming (PyMC)
#
# This example shows how you can use Modal to scale up PyMC code to run with massive parallelism.
# It's based on the [official example]( https://docs.pymc.io/en/stable/pymc-examples/examples/case_studies/probabilistic_matrix_factorization.html)
# but uses a Modal-specific sampler class that aids with the parallelization.

# ## Setup
#
# Let's start with imports. `modal.extensions.pymc` provides helpers for working with PyMC.

import copy

from modal.extensions.pymc import ModalSampler, pymc_stub

# These libraries are only imported inside the pymc image.
# This allows you to run the code locally even if you don't have `pymc` or `numpy` installed.

if pymc_stub.is_inside():
    import numpy as np
    import pandas as pd
    import pymc3 as pm
    import pymc3.parallel_sampling as ps

# ## Statistical model
#
# The rest of the code to set up the model and the training data is the same as in the
# [official example](https://docs.pymc.io/en/stable/pymc-examples/examples/case_studies/probabilistic_matrix_factorization.html).


def split_train_test(data, percent_test=0.1):
    """Split dense matrix into train/test sets. The dimension of both test and train outputs is
    the same as the input.
    """
    n, m = data.shape  # # users, # movies
    N = n * m  # # cells in matrix

    # Prepare train/test ndarrays.
    train = data.copy()
    test = np.ones(data.shape) * np.nan

    # Draw random sample of training data to use for testing.
    tosample = np.where(~np.isnan(train))  # ignore nan values in data
    idx_pairs = list(
        zip(tosample[0], tosample[1])
    )  # tuples of row/col index pairs

    test_size = int(
        len(idx_pairs) * percent_test
    )  # use 10% of data as test set
    train_size = len(idx_pairs) - test_size  # and remainder for training

    indices = np.arange(len(idx_pairs))  # indices of index pairs
    sample = np.random.choice(indices, replace=False, size=test_size)

    # Transfer random sample from train set to test set.
    for idx in sample:
        idx_pair = idx_pairs[idx]
        test[idx_pair] = train[idx_pair]  # transfer to test set
        train[idx_pair] = np.nan  # remove from train set

    # Verify everything worked properly
    assert train_size == N - np.isnan(train).sum()
    assert test_size == N - np.isnan(test).sum()

    # Return train set and test set
    return train, test


class PMF:
    """Probabilistic Matrix Factorization model using pymc3.
    Adapted from https://docs.pymc.io/en/stable/pymc-examples/examples/case_studies/probabilistic_matrix_factorization.html.
    """

    def __init__(self, train, dim, alpha=2, std=0.01):
        self.dim = dim
        self.alpha = alpha
        self.std = np.sqrt(1.0 / alpha)
        self.data = train.copy()
        self.bounds = (1, 5)
        n, m = self.data.shape

        # Perform mean value imputation
        nan_mask = np.isnan(self.data)
        self.data[nan_mask] = self.data[~nan_mask].mean()

        # Low precision reflects uncertainty; prevents overfitting.
        # Set to the mean variance across users and items.
        self.alpha_u = 1 / self.data.var(axis=1).mean()
        self.alpha_v = 1 / self.data.var(axis=0).mean()

        print("building the PMF model")
        with pm.Model() as pmf:
            U = pm.MvNormal(
                "U",
                mu=0,
                tau=self.alpha_u * np.eye(dim),
                shape=(n, dim),
                testval=np.random.randn(n, dim) * std,
            )
            V = pm.MvNormal(
                "V",
                mu=0,
                tau=self.alpha_v * np.eye(dim),
                shape=(m, dim),
                testval=np.random.randn(m, dim) * std,
            )
            pm.Normal(
                "R",
                mu=(U @ V.T)[~nan_mask],
                tau=self.alpha,
                observed=self.data[~nan_mask],
            )

        print("done building the PMF model")
        self.model = pmf

    def draw_samples(self, **kwargs):
        with self.model:
            self.trace = pm.sample(**kwargs)

    def predict(self, U, V):
        """Estimate R from the given values of U and V."""
        R = np.dot(U, V.T)
        n, m = R.shape
        sample_R = np.random.normal(R, self.std)
        # bound ratings
        low, high = self.bounds
        sample_R[sample_R < low] = low
        sample_R[sample_R > high] = high
        return sample_R

    def rmse(self, test_data, predicted=None):
        if predicted is None:
            sample = self.trace[-1]
            predicted = self.predict(sample["U"], sample["V"])

        I = ~np.isnan(test_data)  # indicator for missing values
        N = I.sum()  # number of non-missing values
        sqerror = abs(test_data - predicted) ** 2  # squared error array
        mse = sqerror[I].sum() / N  # mean squared error
        return np.sqrt(mse)  # RMSE


def random_predictions(data):
    result = copy.deepcopy(data)
    nan_mask = np.isnan(result)
    masked_train = np.ma.masked_array(result, nan_mask)
    pmin, pmax = masked_train.min(), masked_train.max()
    N = nan_mask.sum()
    result[nan_mask] = np.random.uniform(pmin, pmax, N)
    return result


# ## Modal function
#
# The pmf function runs inside Modal, but it's not parallelized in itself.
# Note that we patch in `ModalSampler` to replace pymc's `ParallelSampler`.


@pymc_stub.function()
def pmf():
    data = pd.read_csv(
        pm.get_data("ml_100k_u.data"),
        sep="\t",
        names=["userid", "itemid", "rating", "timestamp"],
    )

    num_users = data.userid.unique().shape[0]
    num_items = data.itemid.unique().shape[0]
    sparsity = 1 - len(data) / (num_users * num_items)
    print(f"Users: {num_users}\nMovies: {num_items}\nSparsity: {sparsity}")

    dense_data = data.pivot(
        index="userid", columns="itemid", values="rating"
    ).values
    dense_data = dense_data[:100, :200]
    train, test = split_train_test(dense_data)

    ALPHA = 2
    DIM = 10

    pmf = PMF(train, DIM, ALPHA, std=0.05)

    r = random_predictions(train)
    print(f"Random baseline RMSE: {pmf.rmse(test, r)}")

    ps.ParallelSampler = ModalSampler

    pmf.draw_samples(
        draws=100,
        tune=100,
        cores=4,
        chains=4,
        return_inferencedata=False,
    )
    print(f"Train RMSE: {pmf.rmse(train)}")
    print(f"Test RMSE: {pmf.rmse(test)}")


# Now we can run the probabilistic program.


@pymc_stub.local_entrypoint
def run():
    pmf.call()
