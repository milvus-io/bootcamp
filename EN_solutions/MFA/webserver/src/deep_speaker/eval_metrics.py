import numpy as np


def evaluate(sims, labels):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 1.0, 0.001)
    fm, tpr, acc = calculate_roc(thresholds, sims, labels)
    eer = calculate_eer(thresholds, sims, labels)
    return fm, tpr, acc, eer


def calculate_roc(thresholds, sims, labels):
    nrof_pairs = min(len(labels), len(sims))
    nrof_thresholds = len(thresholds)

    tprs = np.zeros((nrof_thresholds))
    fprs = np.zeros((nrof_thresholds))
    acc_train = np.zeros((nrof_thresholds))
    precisions = np.zeros((nrof_thresholds))
    fms = np.zeros((nrof_thresholds))
    accuracy = 0.0

    indices = np.arange(nrof_pairs)

    # Find the best threshold for the fold

    for threshold_idx, threshold in enumerate(thresholds):
        tprs[threshold_idx], fprs[threshold_idx], precisions[threshold_idx], \
        fms[threshold_idx], acc_train[threshold_idx] = calculate_accuracy(threshold, sims, labels)

    bestindex = np.argmax(fms)
    bestfm = fms[bestindex]
    besttpr = tprs[bestindex]
    bestacc = acc_train[bestindex]

    return bestfm, besttpr, bestacc


def calculate_accuracy(threshold, sims, actual_issame):
    predict_issame = np.greater(sims, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)  # recall
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    precision = 0 if (tp + fp == 0) else float(tp) / float(tp + fp)
    fm = 2 * precision * tpr / (precision + tpr + 1e-12)
    acc = float(tp + tn) / (sims.size + 1e-12)
    return tpr, fpr, precision, fm, acc


def calculate_eer(thresholds, sims, labels):
    nrof_pairs = min(len(labels), len(sims))
    nrof_thresholds = len(thresholds)

    indices = np.arange(nrof_pairs)

    # Find the threshold that gives FAR = far_target
    far_train = np.zeros(nrof_thresholds)
    frr_train = np.zeros(nrof_thresholds)
    eer_index = 0
    eer_diff = 100000000
    for threshold_idx, threshold in enumerate(thresholds):
        frr_train[threshold_idx], far_train[threshold_idx] = calculate_val_far(threshold, sims, labels)
        if abs(frr_train[threshold_idx] - far_train[threshold_idx]) < eer_diff:
            eer_diff = abs(frr_train[threshold_idx] - far_train[threshold_idx])
            eer_index = threshold_idx

    frr, far = frr_train[eer_index], far_train[eer_index]

    eer = (frr + far) / 2

    return eer


def calculate_val_far(threshold, sims, actual_issame):
    predict_issame = np.greater(sims, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    if n_diff == 0:
        n_diff = 1
    if n_same == 0:
        return 0, 0
    val = float(true_accept) / float(n_same)
    frr = 1 - val
    far = float(false_accept) / float(n_diff)
    return frr, far
