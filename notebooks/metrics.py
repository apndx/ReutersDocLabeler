def evaluate(output, target, thresh):
    '''
    Parameters:
    - output needs to be a list (of tensors) that has been sigmoided
    - target needs to be a list (of tensors) of 0's and 1's
    - thresh is a float [0, 1) indicating a threshold over which is considered as 1
    '''
    out = output > thresh
    tgt = target > thresh
    tp = int(sum(out & tgt))
    tn = int(sum((~out) & (~tgt)))
    fp = int(sum(out & (~tgt)))
    fn = int(sum((~out) & tgt))
    precision = (tp / (tp + fp + 1e-12))
    recall = (tp / (tp + fn + 1e-12))
    f_score = (2 * precision * recall)/(precision + recall)

    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f_score': f_score}
