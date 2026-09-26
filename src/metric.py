"""Exact supplied entity macro F0.5, including empty truth sets."""
def entity_f05(truth, prediction):
    truth, prediction = set(truth), set(prediction)
    if not truth:
        return float(not prediction)
    return 1.25 * len(truth & prediction) / (0.25 * len(truth) + len(prediction))


def evaluate(truth, predictions, ids=None):
    ids = list(truth if ids is None else ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Evaluation requires nonempty unique entity IDs')
    if any(q not in truth for q in ids):
        raise ValueError('Missing ground truth is not a singleton')
    tp = nt = npred = single = single_ok = 0
    scores = []
    for q in ids:
        t, p = set(truth[q]), set(predictions.get(q, []))
        scores.append(entity_f05(t, p))
        tp += len(t & p); nt += len(t); npred += len(p)
        single += not t; single_ok += not t and not p
    return {'macro_f0.5': sum(scores)/len(ids), 'precision': tp/npred if npred else 0.,
            'recall': tp/nt if nt else 0., 'singleton_accuracy': single_ok/single if single else None,
            'entities': len(ids), 'true_pairs': nt, 'predicted_pairs': npred}
