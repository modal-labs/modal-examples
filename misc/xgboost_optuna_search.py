# # Distributed Hyperparameter Optimization with Optuna & XGBoost

# [Optuna](https://optuna.org) is an open-source Python framework designed to automate the process
# of finding the optimal hyperparameters for machine learning models.

# This example demonstrates how to parallelize your hyperparameter search with Optuna on Modal with XGBoost,
# [Hyperband pruning](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html),
# and access to the Optuna Dashboard. Pruning automatically stops unpromising trials at the early stages of training,
# saving on compute time.

# ## Defining the App

# We start by defining the image and Modal app.

from pathlib import Path

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "optuna==4.3.0",
    "scikit-learn==1.6.1",
    "numpy==2.2.5",
    "xgboost==3.0.0",
    "optuna-dashboard==0.18.0",
)
app = modal.App("xgboost-optuna-prune", image=image)

# We create a Modal Volume to hold Optuna's
# [JournalStorage](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.JournalStorage.html)
# to track the trials and hyperparameters.

volume = modal.Volume.from_name("xgboost-optuna-logs-prune", create_if_missing=True)
DATA_DIR = Path("/data")
JOURNAL_STORAGE_LOG = DATA_DIR / ".log"

# ## Optuna Worker

# For this example, we restrict the number of containers to 20 and run 500 trials.

CONCURRENCY = 20
N_TRIALS = 500

# The Optuna worker is responsible for:
# 1. Evaluating a specific hyperparameter configuration.
# 2. Loading the dataset into memory during startup, so that the data stays warm for future trials.
# 3. For each XGBoost iteration, call back into the Optuna head to define if the current evaluation should be pruned


@app.cls(cpu=4, memory=1024, max_containers=CONCURRENCY)
class OptunaWorker:
    @modal.enter()
    def load_data(self):
        """Loads the data into memory during startup. Here we use a simple digits dataset. For large production
        datasets, we recommend saving your data into a modal Volume and loading the data from the Volume."""
        import xgboost as xgb
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split

        X, y = load_digits(return_X_y=True)

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42)
        self.dtrain = xgb.DMatrix(X_train, label=y_train)
        self.dvalid = xgb.DMatrix(X_valid, label=y_valid)

    @modal.method()
    def evaluate(self, params: dict, trial_number: int) -> float:
        """Evaluates the XGBoost model for `params`."""
        import numpy as np
        import xgboost as xgb
        from sklearn.metrics import accuracy_score

        # An XGBoost callback that checks whether the run should be pruned
        class XGBoostPruningCallback(xgb.callback.TrainingCallback):
            def __init__(self, head, observation_key: str, trial_number: int):
                self.observation_key = observation_key
                self.head = head
                self.trial_number = trial_number

            def after_iteration(
                self, model: xgb.Booster, epoch: int, evals_log: dict
            ) -> bool:
                evaluation_results = {}
                for dataset, metrics in evals_log.items():
                    for metric, scores in metrics.items():
                        key = dataset + "-" + metric
                        assert isinstance(scores, list), scores
                        evaluation_results[key] = scores[-1]

                current_score = evaluation_results[self.observation_key]

                # The Optuna head defines a `should_prune` method that signals to the worker, if it should prune
                should_prune = self.head.should_prune.remote(
                    current_score, epoch, self.trial_number
                )
                return should_prune

        optuna_head = OptunaHead()
        pruning_callback = XGBoostPruningCallback(
            optuna_head,
            observation_key="validation-merror",
            trial_number=trial_number,
        )

        bst = xgb.train(
            params,
            self.dtrain,
            evals=[(self.dvalid, "validation")],
            callbacks=[pruning_callback],
        )
        preds = bst.predict(self.dvalid)
        pred_labels = np.rint(preds)
        return float(accuracy_score(self.dvalid.get_label(), pred_labels))


# ## Optuna Head

# The Optuna Head object is responsible for:
# 1. Keeping track of the results of the trials.
# 2. Decide when a trial should be pruned.
# 3. Run an optuna dashboard to visualize the progress and trials.
# 4. Concurrently, spawn Optuna workers with a concurrency limit of `CONCURRENCY`.

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna


# The Optuna head requires a consistent state to manage the running trials, so we restrict the number of
# containers to one and allow it to take multiple inputs. With multiple inputs, this one container can handle
# all the requests from the optuna dashboard and the worker's can all `should_prune` at the same time.


@app.cls(cpu=2, memory=2048, volumes={"/data": volume}, max_containers=1)
@modal.concurrent(max_inputs=1000)
class OptunaHead:
    @modal.enter()
    def create_study(self):
        """Define the optuna study."""
        DATA_DIR.mkdir(exist_ok=True)
        JOURNAL_STORAGE_LOG.touch(exist_ok=True)
        import os

        import optuna
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend

        # Keeps track of the running trials
        self.trials: dict[int, optuna.Trial] = {}

        storage = JournalStorage(JournalFileBackend(os.fspath(JOURNAL_STORAGE_LOG)))
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(
            direction="maximize",
            study_name="xgboost-optuna",
            storage=storage,
            pruner=pruner,
            load_if_exists=True,
        )

    @modal.method()
    def should_prune(
        self, intermediate_value: float, step: int, trial_number: int
    ) -> bool:
        """Return True if a trial should be pruned."""
        try:
            trial = self.trials[trial_number]
            trial.report(intermediate_value, step)
            return trial.should_prune()
        except KeyError:
            return False

    async def spawn_worker(self, semaphore: asyncio.Semaphore):
        """Ask the study for a trial and spawn a Optuna worker to evaluate it. The semaphore is used to
        restrict the number of workers at the Python level. This enables the current workers to report back their
        results and inform the hyperparameters for future trials."""
        import optuna

        async with semaphore:
            trial = await asyncio.to_thread(self.study.ask)
            self.trials[trial.number] = trial
            params = await asyncio.to_thread(self.get_param_from_trial, trial)

            try:
                result = await OptunaWorker().evaluate.remote.aio(
                    params, trial_number=trial.number
                )
                await asyncio.to_thread(
                    self.study.tell,
                    trial,
                    result,
                    state=optuna.trial.TrialState.COMPLETE,
                )
                print(f"Completed with: {result}, {trial.params}")
            except Exception:
                await asyncio.to_thread(
                    self.study.tell, trial, state=optuna.trial.TrialState.FAIL
                )
                print(f"Failed with: {trial.params}")

            del self.trials[trial.number]

    @modal.method()
    async def run_trials(self):
        """Entry point for running `N_TRIALS` trials."""
        semaphore = asyncio.Semaphore(CONCURRENCY)
        trials = [self.spawn_worker(semaphore) for _ in range(N_TRIALS)]

        await asyncio.gather(*trials)

    @modal.web_server(port=8000, startup_timeout=30)
    def optuna_dashboard(self):
        """Entry point for the optuna dashboard."""
        volume.reload()
        import os
        import subprocess
        from shutil import which

        optuna_dashboard = which("optuna-dashboard")
        assert optuna_dashboard is not None

        cmd = [
            optuna_dashboard,
            os.fspath(JOURNAL_STORAGE_LOG),
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]

        subprocess.Popen(" ".join(cmd), shell=True)

    def get_param_from_trial(self, trial: "optuna.Trial") -> dict:
        """Helper method to get a hyperparameter configuration from a optuna Trial."""
        param = {
            "objective": "multi:softmax",
            "num_class": 10,
            "eval_metric": "merror",
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "tree_method": "hist",
        }

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical(
                "sample_type", ["uniform", "weighted"]
            )
            param["normalize_type"] = trial.suggest_categorical(
                "normalize_type", ["tree", "forest"]
            )
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        return param


# ## Deploying and Running the Hyperparameter search

# We deploy this Optuna App by running: `modal deploy xgboost_optuna_search.py`. This will give you access to
# the optuna dashboard. Finally, we trigger the hyperparameter search by running `python xgboost_optuna_search.py`.
# This runs the following code, which calls `run_trials` on the Optuna head node.

if __name__ == "__main__":
    head = modal.Cls.from_name("xgboost-optuna-prune", "OptunaHead")()
    fc = head.run_trials.spawn()
    print("Called with function id", fc.object_id)
