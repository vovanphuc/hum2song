from typing import List


def mean_reciprocal_rank(preds: List[str], gt: str, k: int = 10):
    preds = preds[: min(k, len(preds))]
    score = 0
    for rank, pred in enumerate(preds):
        if pred == gt:
            score = 1 / (rank + 1)
            break
    return score
