# EVOLVE-BLOCK-START
import random


def propose_candidate(
    seed: int = 42, n: int = 64, iterations: int = 200
) -> list[float]:
    rng = random.Random(seed)
    best = [1.0] * n
    best_score = evaluate_sequence(best)

    for _ in range(iterations):
        candidate = best.copy()
        index = rng.randrange(n)
        candidate[index] = max(0.0, candidate[index] * rng.uniform(0.25, 1.25))
        score = evaluate_sequence(candidate)
        if score < best_score:
            best = candidate
            best_score = score

    return best


# EVOLVE-BLOCK-END


def evaluate_sequence(sequence: list[float]) -> float:
    if not sequence:
        return float("inf")

    values = [min(1000.0, max(0.0, float(value))) for value in sequence]
    total = sum(values)
    if total < 0.01:
        return float("inf")

    convolution = [0.0] * (2 * len(values) - 1)
    for i, left in enumerate(values):
        for j, right in enumerate(values):
            convolution[i + j] += left * right

    return 2 * len(values) * max(convolution) / (total**2)


def run_code() -> tuple[list[float], float]:
    heights = propose_candidate()
    return heights, evaluate_sequence(heights)
