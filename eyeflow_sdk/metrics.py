import numpy as np
import copy
# ---------------------------------------------------------------------------------------------------------------------------------

def intersection_over_union(p_boxA, p_boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(p_boxA["x_min"], p_boxB["x_min"])
    yA = max(p_boxA["y_min"], p_boxB["y_min"])
    xB = min(p_boxA["x_max"], p_boxB["x_max"])
    yB = min(p_boxA["y_max"], p_boxB["y_max"])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (p_boxA["x_max"] - p_boxA["x_min"] + 1) * (p_boxA["y_max"] - p_boxA["y_min"] + 1)
    boxBArea = (p_boxB["x_max"] - p_boxB["x_min"] + 1) * (p_boxB["y_max"] - p_boxB["y_min"] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    if (boxAArea + boxBArea - interArea) > 0:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    # return the intersection over union value
    return iou
# ---------------------------------------------------------------------------------------------------------------------------------


class mAP():
    def __init__(self, min_iou=0.75):
        self._map_list = []
        self._total_gt = 0
        self._min_iou = min_iou

    def add_detection(self, y_pred, y_true):
        gtp = copy.deepcopy(y_true)
        preds = copy.deepcopy(y_pred)
        self._total_gt += len(gtp)
        tpd = {}
        fpd = {}
        fnd = {}
        for idx_gtp, tr in enumerate(gtp):
            found_tr = False
            for idx_pr, pr in enumerate(preds):
                iou = intersection_over_union(pr["bbox"], tr["bbox"])
                if iou > self._min_iou:
                    found_tr = True
                    if pr["class"] == tr["class"]:
                        if idx_gtp not in tpd.keys():
                            tpd[idx_gtp] = (tr, pr)
                            del(preds[idx_pr])
                        elif idx_gtp not in fpd.keys():
                            fpd[idx_gtp] = [tpd[idx_gtp], (tr, pr)]
                            tpd[idx_gtp] = None
                            del(preds[idx_pr])
                        else:
                            fpd[idx_gtp].append((tr, pr))
                            del(preds[idx_pr])
                    else:
                        if idx_gtp not in fnd.keys():
                            fnd[idx_gtp] = [(tr, pr)]
                            del(preds[idx_pr])
                        else:
                            fnd[idx_gtp].append((tr, pr))
                            del(preds[idx_pr])

            if found_tr:
                del(gtp[idx_gtp])

        fpl = [pr for pr in preds]

        TP = len(tpd)
        FP = len(fpd) + len(fpl)
        FN = len(fnd) + len(gtp)

        self._map_list.append((TP, FP, FN))

    def add_fn(self, y_true):
        self._map_list.append((0, 0, len(y_true)))

    def get_map(self):
        tp_acc = 0
        fp_acc = 0
        fn_acc = 0
        precision = []
        recall = []
        for tp, fp, fn in self._map_list:
            tp_acc += tp
            fp_acc += fp
            fn_acc += fn
            if (tp_acc + fp_acc) > 0:
                precision.append(tp_acc / (tp_acc + fp_acc))
            else:
                precision.append(0)

            if self._total_gt > 0:
                recall.append(tp_acc / self._total_gt)
            else:
                recall.append(0)

        precision = np.array(precision)
        recall = np.array(recall)
        p_interp = []
        for r in range(11):
            indices = np.nonzero((recall >= r/10) & (recall < (r + 1)/10))
            if indices[0].shape[0] > 0:
                last_indices = indices
            p_interp.append(np.max(precision[last_indices]))

        return np.mean(np.array(p_interp))
# ---------------------------------------------------------------------------------------------------------------------------------
