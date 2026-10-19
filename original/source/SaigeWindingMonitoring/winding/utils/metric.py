def iou_range(a: dict, b: dict) -> float:
    if a["direction"] != b["direction"]:
        return 0.0
    start_max = max(a["start"], b["start"])
    end_min = min(a["end"], b["end"])
    inter = max(0, end_min - start_max + 1)
    union = max(a["end"], b["end"]) - min(a["start"], b["start"]) + 1
    return inter / union


def match_segments(
    gt: list[dict], pred: list[dict], iou_threshold: float = 0.5
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    matches = []
    used_gt = set()
    used_pred = set()

    # Compute all IoU pairs
    iou_pairs = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            iou = iou_range(g, p)
            if iou >= iou_threshold:
                iou_pairs.append((i, j, iou))

    # Sort by IoU descending
    iou_pairs.sort(key=lambda x: x[2], reverse=True)

    for i, j, iou in iou_pairs:
        if i not in used_gt and j not in used_pred:
            matches.append((i, j, iou))
            used_gt.add(i)
            used_pred.add(j)

    unmatched_gt = [i for i in range(len(gt)) if i not in used_gt]
    unmatched_pred = [j for j in range(len(pred)) if j not in used_pred]

    return matches, unmatched_gt, unmatched_pred


def compute_metrics(gt: list[dict], pred: list[dict], iou_threshold: float = 0.5) -> dict:
    matches, unmatched_gt, unmatched_pred = match_segments(gt, pred, iou_threshold)

    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = tp / len(gt) if len(gt) > 0 else 0.0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "num_gt": len(gt),
        "num_pred": len(pred),
        "num_matches": tp,
        "unmatched_gt": unmatched_gt,
        "unmatched_pred": unmatched_pred,
        "matches": matches,
    }

    return metrics
