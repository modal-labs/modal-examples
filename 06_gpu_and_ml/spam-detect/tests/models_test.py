import math
from typing import NamedTuple

from spam_detect import models


class Example(NamedTuple):
    email: str
    spam: bool


def test_prob_calculation():
    dataset = [
        Example(email="spam rules", spam=True),
        Example(email="ham rules", spam=False),
        Example(email="hello ham", spam=False),
    ]

    classify_func, _ = models.NaiveBayes(decision_boundary=0.5).train(dataset)
    email = "hello spam"
    probs_if_ham = [
        (1 + 0.5) / (2 + 2 * 0.5),  # "hello" (present)
        (0 + 0.5) / (2 + 2 * 0.5),  # "spam" (present)
        1 - (2 + 0.5) / (2 + 2 * 0.5),  # "ham" (not present)
        1 - (1 + 0.5) / (2 + 2 * 0.5),  # "rules" (not present)
    ]
    probs_if_spam = [
        (0 + 0.5) / (1 + 2 * 0.5),  # "hello" (present)
        (1 + 0.5) / (1 + 2 * 0.5),  # "spam" (present)
        1 - (0 + 0.5) / (1 + 2 * 0.5),  # "ham" (not present)
        1 - (1 + 0.5) / (1 + 2 * 0.5),  # "rules" (not present)
    ]
    p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
    p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))

    # Should be about 0.83
    assert classify_func(email).score == p_if_spam / (p_if_spam + p_if_ham)
