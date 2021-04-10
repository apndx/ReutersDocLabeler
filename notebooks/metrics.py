import torch

def count_hits(output, target, thresh):
    '''
    Parameters:
    - output needs to be a list (of tensors) that has been sigmoided
    - target needs to be a list (of tensors) of 0's and 1's
    - thresh is a float [0, 1) indicating a threshold over which is considered as 1
    '''
    out = output > thresh
    tgt = target > thresh
    tp = (out & tgt)
    tn = ((~out) & (~tgt))
    fp = (out & (~tgt))
    fn = ((~out) & tgt)
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

def evaluate(tp_l, tn_l, fp_l, fn_l):
    '''
    Parameters:
    - tp_l, tn_l, fp_l and fn_l are lists (of tensors) of 0's and 1's
    '''
    tp = int(tp_l.sum())
    tn = int(tn_l.sum())
    fp = int(fp_l.sum())
    fn = int(fn_l.sum())
    precision = (tp / (tp + fp + 1e-12))
    recall = (tp / (tp + fn + 1e-12))
    f_score = (2 * precision * recall)/(precision + recall + 1e-12)

    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f_score': f_score}

