"""Grid-sweep helpers used by the example's `run_sweep` Modal function.

Two pure functions plus their selection knobs:

- `expand_sweep` turns a dict of axes into a list of `(target, binder, seed, ...)`
  config dicts -- one per Modal job in the sweep.
- `select_designs` joins the per-job result frames returned by `design_binder`,
  annotates each design with isoelectric point, drops non-acidic minibinders,
  and keeps the top `top_n` per `(target, binder)` group ranked by an equal-
  weighted blend of hero-critic ipTM and scaling-critic distogram-iPTM-proxy.

Both functions are deliberately Modal-agnostic: the example's `run_sweep`
function calls them inside the orchestrating container, but they are equally
usable from a notebook or a one-off script.
"""

# Selection keeps the top designs per (target, binder) by selection_score.
TOP_N = 84
# Minibinders are filtered to acidic designs; antibodies are kept regardless.
ISOELECTRIC_POINT_MAX = 6.0
# Critic names containing this substring are the smaller "scaling" critics that
# only expose distogram proxies; everything else is treated as a hero critic
# whose calibrated ipTM we use directly.
SCALING_CHECKPOINT_SUBSTRING = "ESMFold2-Experimental-Fast-base"


def expand_sweep(line_sweeps: dict[str, list]) -> list[dict]:
    """Expand a dict of sweep axes into one config dict per grid point."""
    from itertools import product

    keys = list(line_sweeps)
    return [dict(zip(keys, vals)) for vals in product(*line_sweeps.values())]


def select_designs(
    configs: list[dict],
    raw_results: list,
    top_n: int = TOP_N,
    isoelectric_point_max: float = ISOELECTRIC_POINT_MAX,
):
    """Join per-job result frames, filter to plausible designs, and keep the top per group.

    Each design's score is the average of two terms:

    - `iptm_score` -- mean ipTM across hero critics (calibrated by Biohub).
    - `iptm_proxy_score` -- mean distogram-iPTM-proxy across scaling critics
      (uncalibrated but cheap, so we run a larger ensemble of them).

    Antibodies use the CDR-restricted distogram proxy; minibinders use the full
    one. With no scaling critics in the sweep, only `iptm_score` is non-zero.

    Returns a `pandas.DataFrame` of selected designs with `target_name` and
    `binder_name` as columns (suitable for parquet round-trips).
    """
    import pandas as pd
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    from tqdm.auto import tqdm

    # Each `design` call returns (best_sequences, trajectory, critic_results);
    # we only need critic_results for selection, broadcast with config metadata.
    df_result = pd.concat(
        [pd.DataFrame(r[2]).assign(**cfg) for cfg, r in zip(configs, raw_results)],
        ignore_index=True,
    )
    df_result["binder_sequence"] = df_result.designed_sequence.str.split(r"\|").str[1]
    df_result["isoelectric_point"] = [
        ProteinAnalysis(seq).isoelectric_point()
        for seq in tqdm(df_result.binder_sequence.values)
    ]

    # Antibodies aren't acid-filtered; minibinders are.
    df_filter = df_result[
        df_result.is_antibody | df_result.isoelectric_point.lt(isoelectric_point_max)
    ]

    def _rank_group(group):
        is_scaling = group.critic_name.str.contains(
            SCALING_CHECKPOINT_SUBSTRING, regex=False, na=False
        )
        iptm_proxy = group.distogram_iptm_proxy.where(
            ~group.is_antibody, group.cdr_distogram_iptm_proxy
        )
        # Hero critics contribute calibrated ipTM; scaling critics contribute
        # the distogram proxy. Mask each into its own column so the per-sequence
        # mean below pulls only from the right subset of critics.
        group = group.assign(
            iptm_score=group.iptm.where(~is_scaling),
            iptm_proxy_score=iptm_proxy.where(is_scaling),
        )
        scores = group.groupby("designed_sequence", as_index=False).agg(
            iptm_score=("iptm_score", "mean"),
            iptm_proxy_score=("iptm_proxy_score", "mean"),
        )
        # Designs missing one side (e.g., no scaling critics in the sweep) get
        # 0 for that term rather than dropping out of the ranking entirely.
        scores["selection_score"] = 0.5 * scores.iptm_score.fillna(
            0
        ) + 0.5 * scores.iptm_proxy_score.fillna(0)
        return scores.nlargest(min(len(scores), top_n), "selection_score")

    # `reset_index` so `target_name` / `binder_name` survive the parquet
    # round-trip as columns rather than living on a MultiIndex.
    return (
        df_filter.groupby(["target_name", "binder_name"])
        .apply(_rank_group, include_groups=False)
        .reset_index(level=["target_name", "binder_name"])
        .reset_index(drop=True)
    )
