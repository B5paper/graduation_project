import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np
import myutils
import gluoncv as gcv


# transform_fn 将来可以设计成动态生成的，放到 myutils 里。
def transform_fn(*data):
    """
    This function is used as the parameter of dataset.transform(). The coords in mx_label is absolute coords.
    The processing procedure of image contains resize, augmentation, color_normalize, and to_tensor.
    The processing procedure of label just contains resize.

    :param data: (img, label), img: np.array, int, (h, w, c), label: absolute, np.array, int, (N, 5)
    :return: (mx_img, mx_label), mx_img: mx.nd.array, float, (c, h, w), mx_label: absolute, mx.nd.array, (N, 5)
    """
    img, label = data
    img = img.astype('float32')  # deepcopy
    label = label.astype('float32')

    aug_img, aug_label = myutils.data_augment(img, label, size=(300, 300))
    norm_img = mx.img.color_normalize(mx.nd.array(aug_img),
                                      mean=mx.nd.array(myutils.mean),
                                      std=mx.nd.array(myutils.std))
    mx_img = myutils.to_tensor(norm_img)
    mx_label = mx.nd.array(aug_label)

    return mx_img, mx_label


# 这个函数将考虑应该认到 myutils 里，因为每个图片的 label 数都不相等，各个模型 batchify 的时候，都会遇到这个问题
batchify_fn = gcv.data.batchify.Tuple((gcv.data.batchify.Stack(), gcv.data.batchify.Pad(pad_val=-1)))


class YOLO_v1(mx.gluon.HybridBlock):
    """
    Generate a yolo net.
    """
    def __init__(self, ctx=mx.cpu(), **kwargs):
        super(YOLO_v1, self).__init__(**kwargs)
        self.ctx = ctx

        resnet34_v2 = mx.gluon.model_zoo.vision.resnet34_v2(pretrained=True, ctx=ctx,
                                                       root=r'D:\Documents\Data_Files\Parameters')
        with self.name_scope():
            # self.expanded_feature_extractor = resnet50_v2.features[:11]  # (b, 2048, h, w)
            self.expanded_feature_extractor = resnet34_v2.features[:11]
            self.expanded_feature_extractor.add(mx.gluon.nn.Conv2D(1024, kernel_size=(1, 1), activation='relu'))
            self.expanded_feature_extractor.add(mx.gluon.nn.Conv2D(1024, kernel_size=(3, 3), activation='relu'))
            self.expanded_feature_extractor.add(mx.gluon.nn.Conv2D(1024, kernel_size=(3, 3), activation='relu'))
            self.expanded_feature_extractor.add(mx.gluon.nn.Conv2D(1024, kernel_size=(3, 3), activation='relu'))
            self.expanded_feature_extractor.add(mx.gluon.nn.Dense(4096, activation='relu'))
            self.expanded_feature_extractor.add(mx.gluon.nn.Dropout(0.5))
            self.expanded_feature_extractor.add(mx.gluon.nn.Dense(7 * 7 * 30))

        self.expanded_feature_extractor[-7:].collect_params().initialize(init=mx.init.Xavier(), ctx=ctx)
        return

    def forward(self, x):
        out = self.expanded_feature_extractor(x)
        out = out.reshape((-1, 7, 7, 30))
        return out


# visualization
def visualize_grids(img, label, S=7):
    """
    Plot grids and label bounding boxes on the given image.

    :param img: np.array, uint8, (h, w, c)
    :param label: np.array, int32, (N, 5), N represents the id of bbox, the 5 represents (cls_id, x1, y1, x2, y2)
    :param S: the image is divided by S * S grids
    :return: fig, the figure on which plot.
    """

    label = label.reshape((-1, 5))

    fig = plt.imshow(img)
    axes = fig.axes

    height, width = img.shape[:2]
    x_interval = width / S
    y_interval = height / S

    grid_line_start_point = []
    grid_line_end_point = []
    for i in range(S+1):
        grid_line_start_point.append([x_interval * i, 0])
        grid_line_end_point.append([x_interval * i, height])
        grid_line_start_point.append([0, y_interval * i])
        grid_line_end_point.append([width, y_interval * i])

    for i in range(len(grid_line_start_point)):
        x_coords, y_coords = zip(*(grid_line_start_point[i], grid_line_end_point[i]))
        plt.plot(x_coords, y_coords, 'b-', linewidth=1)

    axes.set_xmargin(0)
    axes.set_ymargin(0)

    for obj_label in label:
        rltv_bbox = myutils.bbox_abs_to_rel(bbox=obj_label[1:], pic_size=img.shape[:2])
        myutils._add_rectangle(axes, rltv_bbox)

        x_center, y_center = get_center_coord_of_bboxes(obj_label[1:])[0]
        plt.plot(x_center, y_center, 'r.', markersize=15)

    return fig


def get_center_coord_of_bboxes(bboxes):
    """
    Calculate coordinates of the center point for given bounding boxes.

    :param bboxes: np.array, int, absolute, (N, 5) or (4, )
    :return: np.array, float, absolute, (N, 2)
    """

    bboxes = bboxes.reshape((-1, 4))

    x_center = (bboxes[:, 0] + bboxes[:, 2]) / 2
    y_center = (bboxes[:, 1] + bboxes[:, 3]) / 2
    center_coord = np.array(list(zip(x_center, y_center)))

    return center_coord


def get_center_grid_of_bboxes(img_size, boxes, S):
    """
    find the index of the grid that the center of bounding box locates in.

    :param img_size: np.array or tuple, (height, width)
    :param box: (N, 4), (4, )
    :param S:
    :return: (N, 2), 2: (row_idx, col_idx), the idx starts from 0.
    """
    boxes = boxes.copy().reshape((-1, 4))

    height, width = img_size
    x_interval = width / S
    y_interval = height / S

    center_coords = get_center_coord_of_bboxes(boxes)
    idx_row_col = np.array([])
    for center_coord in center_coords:
        center_x, center_y = center_coord
        for i in range(S):
            if x_interval * i <= center_x <= x_interval * (i+1):
                col_idx = i
            if y_interval * i <= center_y <= y_interval * (i+1):
                row_idx = i

        temp_idx_row_col = np.array([row_idx, col_idx]).reshape((1, 2))
        if idx_row_col.size == 0:
            idx_row_col = temp_idx_row_col
        else:
            idx_row_col = np.concatenate((idx_row_col, temp_idx_row_col))
    return idx_row_col


def translate_box_yolo_to_abs(img_size, boxes_yolo, center_grids, S):
    """
    translate the format of coordinates for given boxes from yolo to absolute.

    :param img_size: np.array or tuple, (h, w)
    :param boxes_yolo: np.array, (N, 4) or (4, )
    :param center_grids: np.array, (N, 2)
    :param S: int, default value is 7
    :return: the translated coordinates, np.array, (N, 4)
    """
    boxes_yolo = boxes_yolo.copy().reshape((-1, 4))
    center_grids = center_grids.copy().reshape((-1, 2))

    height, width = img_size
    row_interval = height / S
    col_interval = width / S

    grid_row, grid_col = np.array(list(zip(*center_grids)))

    coor_center = boxes_yolo[:, :2] * (col_interval, row_interval) +\
        np.array((col_interval * grid_col, row_interval * grid_row)).transpose((1, 0))
    width_height = boxes_yolo[:, 2:4] * (width, height)

    top_left_corner = coor_center - width_height / 2
    bottom_right_corner = coor_center + width_height / 2
    box_abs = np.concatenate((top_left_corner, bottom_right_corner), axis=1)

    return box_abs


def translate_box_abs_to_yolo(img_size, boxes_abs, S):
    """
    translate the format of coords for given boxes from absolute to yolo.

    :param img_size: np.array or tuple, (2, )
    :param boxes_abs: np.array, (N, 4)
    :param S: the number of grids on one side
    :return: the translated coordinates, np.array, (N, 4)
    """
    boxes_abs = boxes_abs.copy()
    boxes_abs = boxes_abs.reshape((-1, 4))

    center_grid_indices = get_center_grid_of_bboxes(img_size, boxes_abs, S)

    height, width = img_size
    row_interval = height / S
    col_interval = width / S

    grid_origin = center_grid_indices * [row_interval, col_interval]
    grid_origin = grid_origin[:, ::-1]
    center_coord = get_center_coord_of_bboxes(boxes_abs)
    yolo_center = (center_coord - grid_origin) / [col_interval, row_interval]

    half_w_h = boxes_abs[:, 2:4] - center_coord
    full_w_h = half_w_h * 2
    rel_w_h = full_w_h / [width, height]

    boxes_yolo = np.concatenate((yolo_center, rel_w_h), axis=1)
    return boxes_yolo


def _generate_random_pred_tensor():
    def generate_random_box():
        box = np.random.uniform(0, 1, size=(4, ))
        return box.reshape((1, 4))

    boxes = np.empty(shape=(2, 4))
    for i in range(2):
        boxes[i] = generate_random_box()
    print(boxes)
    pred_tensor = np.random.uniform(size=(7, 7, 30))
    pred_tensor[4, 2, 0:4] = boxes[0]
    pred_tensor[4, 2, 5:9] = boxes[1]
    return pred_tensor


def generate_target(img_size, label, tensor_pred):
    """
    generate the target of one image for the loss calculation.

    :param img: np.array, (h, w)
    :param label: np.array, int32, absolute, shape: (N, 5), [class_id, x1, y1, x2, y2]
    :param tensor_pred: np.array, float32, (S, S, (B*5 + C))
    :return: target tensor, np.array, float32, (S, S, (B*5 + C))
    """
    tensor_pred = tensor_pred.copy()
    tensor_targ = np.zeros(tensor_pred.shape)

    height, width = img_size
    S = int(tensor_pred.shape[0])
    B = int((tensor_pred.shape[-1] - 20) // 5)
    center_grids = get_center_grid_of_bboxes(img_size, label[:, 1:], S)  # (N, 2), (row, column)

    for label_idx, center_grid in enumerate(center_grids):
        grid_row, grid_col = center_grid

        # box confidence
        boxes_pred = np.array([])
        for i in range(B):
            temp_boxes_pred = tensor_pred[grid_row, grid_col, i*5:i*5+4]
            temp_boxes_pred = temp_boxes_pred.reshape((1, 4))
            if boxes_pred.size == 0:
                boxes_pred = temp_boxes_pred
            else:
                boxes_pred = np.concatenate((boxes_pred, temp_boxes_pred))

        boxes_rel = np.empty(boxes_pred.shape)
        for i, box in enumerate(boxes_pred):
            box_abs = translate_box_yolo_to_abs(img_size, box, center_grid, S)
            box_rel = myutils.bbox_abs_to_rel(box_abs, (height, width))
            boxes_rel[i] = box_rel

        rltv_label_box = myutils.bbox_abs_to_rel(label[label_idx, 1:], img_size)

        iou = mx.nd.contrib.box_iou(mx.nd.array(boxes_rel), mx.nd.array(rltv_label_box.reshape((1, 4))))
        idx_max_iou = np.argmax(iou.asnumpy())

        tensor_targ[grid_row, grid_col, idx_max_iou*5+4] = 1

        # box coordinates
        yolo_label = label[label_idx].copy().reshape((-1, 5))
        yolo_label[:, 1:] = translate_box_abs_to_yolo(img_size, yolo_label[:, 1:], S)
        tensor_targ[grid_row, grid_col, idx_max_iou*5:idx_max_iou*5+4] = yolo_label.flatten()[1:]

        # class probability
        tensor_targ[grid_row, grid_col, int(B*5 + yolo_label.flatten()[0])] = 1

    return tensor_targ


def visualize_pred(img, label, tensor_pred):
    """
    Visualize the comparable boxes predicted for each ground-truth box.

    :param img: np.array, (h, w, c)
    :param label: np.array, (N, 5)
    :param tensor_pred: np.array, (S, S, B*5+20)
    :return: the figure that plots on
    """
    img_size = img.shape[:2]
    S = int(tensor_pred.shape[0])
    B = int((tensor_pred.shape[-1] - 20) // 5)
    center_grids = get_center_grid_of_bboxes(img_size, label[:, 1:], S)  # (N, 2), (row, column)

    boxes_pstv = np.array([])
    for center_grid in center_grids:
        grid_row, grid_col = center_grid
        boxes_pred = np.array([])
        for i in range(B):
            temp_boxes_pred = tensor_pred[grid_row, grid_col, i*5:i*5+4].reshape((1, 4))
            if boxes_pred.size == 0:
                boxes_pred = temp_boxes_pred
            else:
                boxes_pred = np.concatenate((boxes_pred, temp_boxes_pred))

        for box_yolo in boxes_pred:
            box_abs = translate_box_yolo_to_abs(img_size, box_yolo, center_grid, S)
            if boxes_pstv.size == 0:
                boxes_pstv = box_abs.reshape((1, 4))
            else:
                boxes_pstv = np.concatenate((boxes_pstv, box_abs.reshape((1, 4))))

    fig = myutils.data_visualize(img, label[:, 1:])
    axes = fig.axes[0]
    for box in boxes_pstv:
        box_rel = myutils.bbox_abs_to_rel(box, img_size)
        myutils._add_rectangle(axes, box_rel, 'blue')

    return fig


def calc_yolo_loss(tensor_pred, tensor_targ, lambda_coord=5, lambda_noobj=0.5):
    """
    calculate the loss function of yolo algorithm.

    :param tensor_pred: mx.nd.array, (S, S, (B*5 + C))
    :param tensor_targ: mx.nd.array, (S, S, (B*5 + C))
    :return:
    """
    # tensor_pred = mx.nd.array(tensor_pred)
    # tensor_targ = mx.nd.array(tensor_targ)
    if type(tensor_targ) == type(mx.nd.array([])):
        ctx = tensor_targ.context
    else:
        ctx = mx.cpu()
        tensor_targ = mx.nd.array(tensor_targ)

    S = int(tensor_pred.shape[0])
    B = int((tensor_pred.shape[-1] - 20) // 5)
    C = int(tensor_pred.shape[2] - B * 5)

    lambda_coord = 5
    lambda_noobj = 0.5

    boxes_pred = tensor_pred[:, :, :B*5]
    boxes_targ = tensor_targ[:, :, :B*5]
    cls_pred = tensor_pred[:, :, B*5:]
    cls_targ = tensor_targ[:, :, B*5:]

    mask_obj = np.zeros((B, S, S))
    for i in range(B):
        idx_temp = tensor_targ[:, :, 5*i+4].asnumpy() == 1
        mask_obj[i][idx_temp] = 1
    mask_obj = mx.nd.array(mask_obj, ctx=ctx)

    mask_noobj = np.zeros(mask_obj.shape)
    mask_noobj[mask_obj.asnumpy() == 0] = 1
    mask_noobj = mx.nd.array(mask_noobj, ctx=ctx)

    loss = mx.nd.array([0.], ctx=ctx)
    for i in range(B):
        box_pred = boxes_pred[:, :, 5*i:5*(i+1)]
        box_targ = boxes_targ[:, :, 5*i:5*(i+1)]

        x_pred, y_pred = box_pred[:, :, 0], box_pred[:, :, 1]
        x_targ, y_targ = box_targ[:, :, 0], box_targ[:, :, 1]
        center_loss = lambda_coord * mx.nd.sum(((x_pred - x_targ)**2 + (y_pred - y_targ)**2) * mask_obj[i], axis=(0, 1))

        w_pred, h_pred = box_pred[:, :, 2], box_pred[:, :, 3]
        w_targ, h_targ = box_targ[:, :, 2], box_targ[:, :, 3]
        # loss = loss + lambda_coord * mx.nd.sum(((mx.nd.sqrt(w_pred) - mx.nd.sqrt(w_targ))**2 +
        #               (mx.nd.sqrt(h_pred) - mx.nd.sqrt(h_targ))**2) * mask_obj, axis=(0, 1))
        size_loss = lambda_coord * mx.nd.sum(((w_pred - w_targ) ** 2 +
                                    (h_pred - h_targ) ** 2) * mask_obj[i], axis=(0, 1))

        c_pred = box_pred[:, :, 4]
        c_targ = box_targ[:, :, 4]
        confidence_loss = mx.nd.sum((c_pred - c_targ)**2 * mask_obj[i], axis=(0, 1))

        negative_loss = lambda_noobj * mx.nd.sum((c_pred - c_targ)**2 * mask_noobj[i], axis=(0, 1))

        loss = loss + center_loss + size_loss + confidence_loss + negative_loss

    mask_cls_obj = mask_obj.asnumpy().copy()
    mask_cls_obj = mask_cls_obj[np.where(mask_cls_obj == 1)[0], :, :][0].reshape((S, S))
    mask_cls_obj = mx.nd.array(mask_cls_obj, ctx=ctx)

    temp_value = mx.nd.sum((cls_pred - cls_targ)**2, axis=-1) * mask_cls_obj
    class_loss = mx.nd.sum(temp_value, axis=(0, 1))

    loss = loss + class_loss

    return loss


def generate_target_batch(img_size, labels, tensor_preds):
    """
    Generate a batch of target.
    Note the type of input tensors must be mx.nd.array

    :param img_size: np.array or tuple, (2, ), (height, width)
    :param labels: mx.nd.array, (b, N, 5)
    :param tensor_preds: mx.nd.array, (b, S, S, B*5+C)
    :return:
    """
    ctx = labels.context
    labels = labels.asnumpy()
    batch_num = labels.shape[0]

    targets = np.array([])
    for i in range(batch_num):
        label = labels[i]
        pad_idx = []
        for j, cls_box in enumerate(label):
            if (cls_box[:2] == [-1, -1]).all():
                pad_idx.append(j)
        pad_idx = np.array(pad_idx)
        pad_bool_idx = np.zeros(label.shape[0]).astype('bool')
        pad_bool_idx[pad_idx.tolist()] = True
        label = label[~pad_bool_idx].reshape((-1, 5))

        target = generate_target(img_size, label, tensor_preds[i].asnumpy())
        target = np.expand_dims(target, axis=0)
        if targets.size == 0:
            targets = target
        else:
            targets = np.concatenate((targets, target), axis=0)
    return mx.nd.array(targets, ctx=ctx)


def calc_batch_loss(tensor_preds, tensor_targs, lambda_coord=5, lambda_noobj=0.5):
    """
    The calc_yolo_loss function's batch version.

    :param tensor_preds: mx.nd.array, (b, S, S, B*5+C)
    :param tensor_targs: mx.nd.array, (b, S, S, B*5+C)
    :param lambda_coord:
    :param lambda_noobj:
    :return:
    """
    batch_num = tensor_preds.shape[0]
    batch_loss = mx.nd.array([0.], ctx=tensor_targs.context)
    for i in range(batch_num):
        loss = calc_yolo_loss(tensor_preds[i], tensor_targs[i], lambda_coord, lambda_noobj)
        batch_loss = batch_loss + loss

    return batch_loss


def get_pred_boxes_with_class_id(img_size, tensor_pred, center_grids):
    if type(tensor_pred) == type(mx.nd.array([])):
        tensor_pred = tensor_pred.asnumpy()

    center_num = center_grids.shape[0]
    S = int(tensor_pred.shape[0])
    B = int((tensor_pred.shape[-1] - 20) // 5)
    C = int(tensor_pred.shape[2] - B * 5)

    pred_boxes_with_class_id = np.zeros((center_num, 5))
    for i, center_grid in enumerate(center_grids):
        center_grid = center_grid.reshape((-1, 2))  # (1, 2)
        pred_in_one_grid = tensor_pred[tuple(center_grid.flatten())]  # (30, )
        pred_boxes_with_confidence = pred_in_one_grid[:B*5]  # (B*5, )

        pred_boxes = pred_boxes_with_confidence.reshape((-1, 5))[:, :4]  # (B, 4)
        pred_confs = pred_boxes_with_confidence.reshape((-1, 5))[:, 4:5]  # (B, 1)
        pred_class = pred_in_one_grid[B*5:]  # (C, )

        pred_boxes_abs = translate_box_yolo_to_abs(img_size, pred_boxes, center_grid.repeat(B, axis=0), S)  # (B, 4)
        pred_boxes_abs = pred_boxes_abs[np.argmax(pred_confs)]  # (4, )
        pred_class_id = np.argmax(pred_class).reshape((1, )) # (1, )
        pred_boxes_with_class_id[i] = np.concatenate((pred_class_id, pred_boxes_abs))  # (5, )

    return pred_boxes_with_class_id # (N, 5), 5: (cls_id, x_1, y_1, x_2, y_2), absolute


def get_pred_scores_class_and_boxes(img_size, tensor_pred):
    tensor_pred = tensor_pred.asnumpy()

    S = int(tensor_pred.shape[0])
    B = int((tensor_pred.shape[-1] - 20) // 5)
    C = int(tensor_pred.shape[2] - B * 5)

    # class id
    cls_id = np.argmax(tensor_pred[:, :, B*5:], axis=2).reshape(S, S, 1, 1)  # (S, S, 1)
    cls_id = cls_id.repeat(B, axis=2)  # (S, S, B, 1)

    # boxes
    boxes_idx = np.array([range(i*5, i*5+4) for i in range(B)])
    boxes = tensor_pred[:, :, boxes_idx]  # (S, S, B, 4)
    boxes = boxes.reshape((-1, 4))
    center_grids = np.array(np.meshgrid(np.arange(S), np.arange(S)))
    center_grids = center_grids.transpose((2, 1, 0))  # (S, S, 2)
    center_grids = np.expand_dims(center_grids, axis=2).repeat(2, axis=2).reshape((-2, 2))  # (N, 2)
    boxes = translate_box_yolo_to_abs(img_size, boxes, center_grids, S)  # (N, 4)
    boxes = boxes.reshape((S, S, B, 4))  # (S, S, B, 4)

    # confidence
    confs = tensor_pred[:, :, 4:B*5:5].reshape((S, S, B, 1))  # (S, S, B, 1)

    # class probability
    cls_prob = tensor_pred[:, :, B*5:]  # (S, S, C)
    cls_prob = mx.nd.array(cls_prob)
    cls_prob = cls_prob.softmax().asnumpy()
    cls_prob_idx = np.argmax(cls_prob, axis=-1)  # (S, S)
    cls_prob = cls_prob.take(cls_prob_idx)
    cls_prob = cls_prob.reshape((S, S, 1, 1)).repeat(B, axis=2)  # (S, S, B, 1)

    # scores
    scores = confs * cls_prob  # (S, S, B, 1)

    # output
    scores_class_and_boxes = np.concatenate((scores, cls_id, boxes), axis=-1)
    scores_class_and_boxes = scores_class_and_boxes.reshape((-1, 6))

    # sort
    order_idx = scores_class_and_boxes[:, 0].argsort()  # (S * S * B, )
    order_idx = order_idx[::-1]
    scores_class_and_boxes = scores_class_and_boxes[order_idx]

    return scores_class_and_boxes  # (-1, 6), 6: (score, cls_id, x_1, y_1, x_2, y_2), absolute


def get_pred_batch_scores_class_and_boxes(img_size, tensor_preds):
    """

    :param img_size: (h, w)
    :param tensor_preds: mx.nd.array, (B, )
    :return:
    """


    pass


def visualize_prediction(img, label, net, size=0, cls_names=None, fig=None):
    """

    :param img: np.array, shape: (h, w, c)
    :param label: np.array, shape: (N, 5)
    :param net: the net that used (net object must has 'ctx' attribute)
    :param size: the size of the image that adapts the input shape of net, np.array or tuple, shape: (2, ), 2: (h, w)
    :param cls_names: the class names list
    :param fig: the figure that will be plotted on
    :return: the figure where plotted
    """
    mx_img, mx_label = myutils.prepare_datum_for_net(img, label, size=size)
    mx_img = mx_img.as_in_context(net.ctx)
    tensor_preds = net(mx_img)
    tensor_pred = tensor_preds[0]

    scores_class_boxes = get_pred_scores_class_and_boxes(img.shape[:2], tensor_pred)
    scores_class_boxes_nms = mx.nd.contrib.box_nms(mx.nd.array(scores_class_boxes),
                                                   overlap_thresh=0.5, valid_thresh=0.005, score_index=0, id_index=1)
    scores_class_boxes_nms = scores_class_boxes_nms.asnumpy()
    scores_class_boxes_nms = scores_class_boxes_nms[scores_class_boxes_nms[:, 0] != -1]

    if not fig:
        fig = plt.figure()

    myutils.visualize_pred(img, label, scores_class_boxes_nms, cls_names, fig=fig, show_label=True)
    return fig




def train_model(dataset, net, epoch, lr, batch_size=1, ctx=mx.cpu()):
    trainer = mx.gluon.Trainer(net.collect_params(), 'sgd')
    trainer.set_learning_rate(lr)

    batchify_fn = gcv.data.batchify.Tuple((gcv.data.batchify.Stack(), gcv.data.batchify.Pad(pad_val=-1)))
    dataloader = mx.gluon.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, batchify_fn=batchify_fn)

    for epoch in range(epoch):
        epoch_loss = 0
        for imgs, labels in dataloader:
            batch_num = imgs.shape[0]
            with mx.autograd.record():
                tensor_preds = net(imgs.as_in_context(ctx))
                tensor_targs = generate_target_batch(imgs.shape[2:], labels.as_in_context(ctx), tensor_preds)
                batch_loss = calc_batch_loss(tensor_preds, tensor_targs)

            batch_loss.backward()
            trainer.step(batch_num)

        epoch_loss += batch_loss
        print('epoch', epoch, 'loss:', epoch_loss.abs().mean().asscalar())



if __name__ == '__main__':
    ctx = mx.gpu()
    root_path = r'D:\Documents\Data_Files\Datasets\Pascal\VOC2012'

    dataset = myutils.Dataset(root_path)
    mx_dataset = dataset.transform(transform_fn)
    img, label = dataset[6]

    yolo_v1 = YOLO_v1(ctx=ctx)
    yolo_v1.hybridize()
    output = yolo_v1(mx.nd.random.normal(shape=(2, 3, 300, 300), ctx=ctx))
    print(output.shape)

    img, label = dataset[6]
    mx_img, mx_label = myutils.prepare_datum_for_net(img, label, size=(300, 300))

    trainer = mx.gluon.Trainer(yolo_v1.collect_params(), 'sgd')
    trainer.set_learning_rate(0.1)

    with mx.autograd.record():
        tensor_preds = yolo_v1(mx_img.as_in_context(ctx))
        tensor_targs = generate_target_batch(mx_img.shape[2:], mx_label.as_in_context(ctx), tensor_preds)
        batch_loss = calc_batch_loss(tensor_preds, tensor_targs)
    batch_loss.backward()
    trainer.step(1)