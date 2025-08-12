# metrics_affiliation.py
import numpy as np

def _events_from_binary(y):
   
    y = np.asarray(y).astype(int)
    events = []
    in_evt = False
    start = 0
    for i, v in enumerate(y):
        if v == 1 and not in_evt:
            start = i
            in_evt = True
        elif v == 0 and in_evt:
            events.append((start, i - 1))
            in_evt = False
    if in_evt:
        events.append((start, len(y) - 1))
    return events

def _coverage_fraction(seg, cover_mask):
    
    s, e = seg
    if e < s:
        return 0.0
    length = e - s + 1
    if length <= 0:
        return 0.0
    return float(cover_mask[s:e+1].sum()) / float(length)

def affiliation_metrics(y_true, y_pred, threshold=None):
   
    y_true = np.asarray(y_true).astype(int)

    yp = np.asarray(y_pred)
    if yp.dtype == np.bool_:
        y_pred_bin = yp
    elif np.issubdtype(yp.dtype, np.integer):
        y_pred_bin = (yp >= 1)
    else:
        th = 0.5 if threshold is None else float(threshold)
        y_pred_bin = (yp >= th)

    # Events
    true_events = _events_from_binary(y_true)
    pred_events = _events_from_binary(y_pred_bin.astype(int))

    # Edge cases
    if len(true_events) == 0 and len(pred_events) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_events) == 0:
        # Keine Vorhersagen → keine Präzision, Recall=0 (wenn es True-Events gibt)
        return 0.0, 0.0, 0.0
    if len(true_events) == 0:
        # Nur Vorhersagen, keine True-Events → Präzision/Recall 0
        return 0.0, 0.0, 0.0

    # Vereinigungen als Masken
    T_union = (y_true == 1)
    P_union = (y_pred_bin == 1)

    # Affiliation-Precision: mittlere Abdeckung JEDER Vorhersage durch True-Union
    prec_contrib = [_coverage_fraction(pe, T_union) for pe in pred_events]
    precision = float(np.mean(prec_contrib)) if prec_contrib else 0.0

    # Affiliation-Recall: mittlere Abdeckung JEDES True-Events durch Pred-Union
    rec_contrib = [_coverage_fraction(te, P_union) for te in true_events]
    recall = float(np.mean(rec_contrib)) if rec_contrib else 0.0

    # F1 (aus obigen affiliation-basierten Größen)
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
    return precision, recall, f1



def affiliation_metrics_from_scores(y_true, scores, threshold):
    return affiliation_metrics(y_true, scores, threshold=threshold)
