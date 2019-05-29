import gluoncv as gcv
import mxnet as mx
import numpy as np

def parse_batch_detection_outputs(batch_pred_scores_cls_boxes, batch_labels):
    batch_pred_bboxes = batch_pred_scores_cls_boxes[:, :, 2:]
    batch_pred_labels = batch_pred_scores_cls_boxes[:, :, 1]
    batch_pred_scores = batch_pred_scores_cls_boxes[:, :, 0]
    batch_gt_boxes = batch_labels[:, :, 1:]
    batch_gt_labels = batch_labels[:, :, 0]
    return batch_pred_bboxes, batch_pred_labels, batch_pred_scores, batch_gt_boxes, batch_gt_labels

class PascalVocMetric():
    def __init__(self, class_names, anchors, get_scr_cls_box_fn):
        """
        anchors: (N, 4)
        """
        self._metric = gcv.utils.metrics.voc_detection.VOCMApMetric(iou_thresh=0.5, class_names=class_names)
        self._anchors = mx.nd.array(anchors).expand_dims(axis=0)  # (1, N, 4)
        self._get_scr_cls_box_fn = get_scr_cls_box_fn

    def reset(self):
        self._metric.reset()

    def update(self, tensor_preds, imgs, labels):
        """
        tensor_preds: mx.nd.NDArray, (b, N, 25)
        imgs: list
        labels: list
        """
        batch_scores_cls_boxes = self._get_scr_cls_box_fn(tensor_preds, self._anchors.as_in_context(tensor_preds.context))
        parsed_detection_output = parse_batch_detection_outputs(batch_scores_cls_boxes, labels)
        self._metric.update(*parsed_detection_output)

    def calc_ap(self, net, dataloader_val):
        self.reset()
        for mx_imgs, mx_labels in dataloader_val:
            tensor_preds = net(mx_imgs.as_in_context(net.ctx))
            self.update(tensor_preds, mx_imgs, mx_labels)
        return self.get()[1][-1]

    def get(self):
        return self._metric.get()
