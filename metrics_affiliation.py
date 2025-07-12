# metrics_affiliation.py
import numpy as np

def get_events(y):
    events = []
    in_event = False
    for i, val in enumerate(y):
        if val == 1 and not in_event:
            start = i
            in_event = True
        elif val == 0 and in_event:
            end = i - 1
            events.append((start, end))
            in_event = False
    if in_event:
        events.append((start, len(y) - 1))
    return events

def affiliation_precision_recall_f1(y_true, y_pred):
    """
    Berechne Affiliation-basierte Precision, Recall, F1
    """

    if y_pred.dtype != np.bool_:
        y_pred = (y_pred >= 0.5).astype(int)

    true_events = get_events(y_true)
    pred_events = get_events(y_pred)

    tp = 0
    matched_pred = set()

    for t_start, t_end in true_events:
        for i, (p_start, p_end) in enumerate(pred_events):
            if i in matched_pred:
                continue
            if p_end >= t_start and p_start <= t_end:
                tp += 1
                matched_pred.add(i)
                break

    fn = len(true_events) - tp
    fp = len(pred_events) - tp

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1
