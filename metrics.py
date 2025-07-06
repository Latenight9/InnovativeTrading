# metrics.py

import numpy as np

def event_based_scores(y_true, y_pred, threshold=0.5):
    """
    Event-wise Precision, Recall, F1-Score.

    Annahme:
        y_true: binärer Vektor (0 = normal, 1 = Anomalie)
        y_pred: Score-Vektor oder binär (0/1)
    """

    if y_pred.dtype != np.bool_:
        y_pred = (y_pred >= threshold).astype(int)

    # Events identifizieren
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

    true_events = get_events(y_true)
    pred_events = get_events(y_pred)

    tp = 0
    matched = set()

    for t_start, t_end in true_events:
        for i, (p_start, p_end) in enumerate(pred_events):
            if i in matched:
                continue
            if p_end >= t_start and p_start <= t_end:
                tp += 1
                matched.add(i)
                break

    fn = len(true_events) - tp
    fp = len(pred_events) - tp

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1


def accuracy(y_true, y_pred, threshold=0.5):
    if y_pred.dtype != np.bool_:
        y_pred = (y_pred >= threshold).astype(int)
    return np.mean(y_true == y_pred)
