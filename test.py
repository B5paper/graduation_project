import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx
import myutils
import yolo_utils
import importlib
import gluoncv as gcv

root_path = r'D:\Documents\Data_Files\Datasets\Pascal\VOC2012'
dataset = myutils.Dataset(root_path, mode='train')
img, label = dataset[0]

ctx = mx.cpu()
yolo_v1 = yolo_utils.YOLO_v1(ctx)
yolo_v1.load_parameters(r'D:\Documents\Data_Files\Parameters\params_loss_less_than_7300', ctx=ctx)

mx_img, mx_label = myutils.prepare_datum_for_net(img, label, (300, 300))
tensor_preds = yolo_v1(mx_img)
tensor_pred = tensor_preds[0]
print('tensor_pred shape:', tensor_pred.shape)

scores_class_boxes = yolo_utils.get_pred_scores_class_and_boxes(img.shape[:2], tensor_pred)
print('scores_class_boxes shape:', scores_class_boxes.shape)

scores_class_boxes_nms = mx.nd.contrib.box_nms(mx.nd.array(scores_class_boxes),
                                               overlap_thresh=0.5, valid_thresh=0.005, score_index=0, id_index=1)
scores_class_boxes_nms = scores_class_boxes_nms.asnumpy()
scores_class_boxes_nms = scores_class_boxes_nms[scores_class_boxes_nms[:, 0] != -1]
print('scores_class_boxes_nms shape:', scores_class_boxes_nms.shape)

metric = gcv.utils.metrics.voc_detection.VOCMApMetric()
metric.reset()
mx_pred_bboxes = np.expand_dims(scores_class_boxes_nms[:, 2:], axis=0).astype('int32')
mx_pred_labels = np.expand_dims(scores_class_boxes_nms[:, 1], axis=0).astype('int32')
mx_pred_scores = np.expand_dims(scores_class_boxes_nms[:, 0], axis=0)
mx_gt_boxes = np.expand_dims(label[:, 1:], axis=0)
mx_gt_labels = np.expand_dims(label[:, 0], axis=0)
metric.update(mx_pred_bboxes, mx_pred_labels, mx_pred_scores, mx_gt_boxes, mx_gt_labels)
print(metric.get())