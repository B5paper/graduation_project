import numpy as np
import myutils
import mxnet as mx
import gluoncv as gcv
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class SSD(mx.gluon.HybridBlock):
    def __init__(self, backbone_root_path=None, ctx=mx.cpu(), **kwargs):
        super(SSD, self).__init__(**kwargs)

        if not backbone_root_path:
            raise Exception('backbone parameter path is None.')

        super(SSD, self).__init__(**kwargs)
        self.ctx = ctx
        self.model_img_size = (300, 300)
        self.anchors = np.array([])
        self.feature_map_shapes_list = []
        self.anchor_num_per_position = 6
        self.start_layer_idx = 6  # idx: 6~10 (included)
        self._num_feature_maps = 5  # build-in feature maps: 3, expanded feature maps: 2

        backbone = mx.gluon.model_zoo.vision.resnet18_v2(pretrained=True, ctx=self.ctx, root=backbone_root_path)

        with self.name_scope():
            self.features = backbone.features[:9]
            # expand feature maps
            for _ in range(2):
                layer = self._make_downsample_layer(channels=256, kernel_size=(3, 3), strides=(2, 2))
                self.features.add(layer)

            # define predictors
            self.cls_predictors = []
            self.box_predictors = []
            for i in range(self.start_layer_idx, len(self.features)):
                setattr(self, 'cls_predictor_' + str(i),
                        mx.gluon.nn.Conv2D(self.anchor_num_per_position * 21, kernel_size=(3, 3), strides=1, padding=1))
                setattr(self, 'box_predictor_' + str(i),
                        mx.gluon.nn.Conv2D(self.anchor_num_per_position * 4, kernel_size=(3, 3), strides=1, padding=1))
                eval('self.cls_predictor_' + str(i) + '.initialize(mx.init.Xavier(), ctx=self.ctx)')
                eval('self.box_predictor_' + str(i) + '.initialize(mx.init.Xavier(), ctx=self.ctx)')
                self.cls_predictors.append(eval('self.cls_predictor_' + str(i)))
                self.box_predictors.append(eval('self.box_predictor_' + str(i)))
        return

    def _make_downsample_layer(self, channels=None, kernel_size=None, strides=None):
        layer = mx.gluon.nn.HybridSequential()
        with layer.name_scope():
            layer.add(mx.gluon.nn.BatchNorm())
            layer.add(mx.gluon.nn.Conv2D(channels, kernel_size, strides, use_bias=False))
            layer.add(mx.gluon.nn.BatchNorm())
            layer.add(mx.gluon.nn.Activation('relu'))
        layer.initialize(mx.init.Xavier(), ctx=self.ctx)
        return layer

    def _generate_anchors_for_one_feature_map(self, feature_map_shape, sizes, ratios):
        """

        :param feature_map_shape: tuple, (b, c, h, w)
        :param sizes: tuple, sizes of anchors
        :param ratios: tuple, ratios of anchors
        :return: anchors, np.array, shape: (N, 4), relative
        """
        anchors = mx.nd.contrib.MultiBoxPrior(mx.nd.empty((feature_map_shape)), sizes, ratios)
        return anchors[0].asnumpy()

    def _generate_anchors(self, feature_map_shapes):
        """
        :param feature_map_shapes: tuple, element: tuple, shape: (b, c, h, w)
        :return: anchors, np.array, shape: (N, 4), relative
        """
        s_min = 0.2
        s_max = 0.9
        m = len(feature_map_shapes)
        ratios = (1, 2, 3, 1 / 2, 1 / 3)
        anchors = np.array([])
        for i, feature_map_shape in enumerate(feature_map_shapes):
            s_k = s_min + (s_max - s_min) / (m - 1) * (i - 1)
            s_k1 = s_min + (s_max - s_min) / (m - 1) * i
            s_kk = np.sqrt(s_k * s_k1)
            sizes = (s_k, s_kk)
            anchor = self._generate_anchors_for_one_feature_map(feature_map_shape, sizes, ratios)
            if anchors.size == 0:
                anchors = anchor
            else:
                anchors = np.concatenate((anchors, anchor), axis=0)  # (N+M, 4)
        return anchors

    mx.gluon.data.vision.transforms.Resize

    def get_transform_fn_val(self):
        def _transform_fn(*data):
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

            # validation 数据集没有 color jitter, crop 等操作，只有 resize
            aug_img, aug_label = myutils.data_augment(img, label, size=self.model_img_size)
            norm_img = mx.img.color_normalize(mx.nd.array(aug_img),
                                              mean=mx.nd.array(myutils.mean),
                                              std=mx.nd.array(myutils.std))
            mx_img = myutils.to_tensor(norm_img)
            aug_label[:, 1:] = myutils.bbox_abs_to_rel(aug_label[:, 1:], mx_img.shape[-2:])
            mx_label = mx.nd.array(aug_label)

            return mx_img, mx_label
        return _transform_fn


    def get_feature_map_shapes_list(self):
        if len(self.feature_map_shapes_list) != 0:
            return self.feature_map_shapes_list

        x = mx.nd.empty((1, 3, 300, 300), ctx=self.ctx)
        for i in range(self.start_layer_idx, len(self.features)):
            # downsample
            if i == self.start_layer_idx:
                x = self.features[:self.start_layer_idx + 1](x)
            else:
                x = self.features[i:i + 1](x)

            self.feature_map_shapes_list.append(x.shape)
        return self.feature_map_shapes_list

    def get_anchors(self):
        if self.anchors.size != 0:
            return self.anchors

        feature_maps_list = self.get_feature_map_shapes_list()
        self.anchors = self._generate_anchors(feature_maps_list)
        return self.anchors

    def get_transform_fn(self):
        def _transform_fn(*data):
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

            aug_img, aug_label = myutils.data_augment(img, label, size=self.model_img_size, rb=0.0, rc=0.0, rh=0.0, rs=0.0, 
                rflr=False, re=True, rcp=False)
            norm_img = mx.img.color_normalize(mx.nd.array(aug_img),
                                              mean=mx.nd.array(myutils.mean),
                                              std=mx.nd.array(myutils.std))
            mx_img = myutils.to_tensor(norm_img)
            aug_label[:, 1:] = myutils.bbox_abs_to_rel(aug_label[:, 1:], mx_img.shape[-2:])
            mx_label = mx.nd.array(aug_label)

            return mx_img, mx_label
        return _transform_fn

    def forward(self, x):  # 可以改成 hybrid_foward
        tensor_pred_list = []  # element shape: (b, N, 5, ((C+1)+4)), N: the number of anchors on each feature map
        cls_pred = []
        box_pred = []

        for i in range(self.start_layer_idx, len(self.features)):
            # downsample
            if i == self.start_layer_idx:
                x = self.features[:self.start_layer_idx + 1](x)
            else:
                x = self.features[i:i + 1](x)

            # prediction
            cls_pred = self.cls_predictors[i - self.start_layer_idx](x)  # shape: (b, 5*(C+1), h, w)
            box_pred = self.box_predictors[i - self.start_layer_idx](x)  # (b, 5*4, h, w)
            tensor_pred_list.append(mx.nd.concat(
                cls_pred.transpose((0, 2, 3, 1)).reshape((0, -1, self.anchor_num_per_position, 21)),
                box_pred.transpose((0, 2, 3, 1)).reshape((0, -1, self.anchor_num_per_position, 4)),
                dim=3))

        tensor_pred = tensor_pred_list[0]
        for tensor in tensor_pred_list[1:]:
            tensor_pred = mx.nd.concat(tensor_pred, tensor, dim=1)

        return tensor_pred.reshape((0, -1, 25))  # (b, N, 25)


def visualize_anchor(img, anchors, i):
    """
    :function: visualize the i-th group of anchors. if i is greater than bounding, this function will give warning.
    :param img: numpy.array, float32 or int8, (h, w, c)
    :param anchors: np.array, (1, N, 4)
    :param i: the i-th group
    :return: None
    """
    if isinstance(anchors, mx.nd.NDArray):
        anchors = anchors.asnumpy()

    total_num = anchors.shape[1]
    if i*5+5 > total_num:
        print('There is no %i-th group of anchors, and the maximum of i is' % i, total_num//5-1)
        return
    bboxes = anchors[0, i*5:i*5+5, :]
    bboxes = myutils.bbox_rel_to_abs(bbox=bboxes, pic_size=img.shape[:2])
    fig = myutils.data_visualize(img, bboxes)
    return fig


def _generate_target(mx_img, mx_label, anchors, do_hard_mining=False, tensor_pred=None, neg_thresh=0.2):
    """
    这个函数需要修改，但是不是现在  
    mx_img: mx.nd.array, (b, 3, h, w)  
    mx_label: mx.nd.array, (b, N, 5), relative  
    anchors: np.array, (1, P*A, 4), relative  
    tensor_pred: mx.nd.array, (b, P*A, C+1+4), relative  
    return: cls_targ, box_targ, pos_mask, neg_mask
    """

    height, width = mx_img.shape[-2:]
    # label = mx_labels[0, :, :].asnumpy()
    # height, width = img_size
    gt_h_w = mx.nd.array([mx_label[:, 4] - label[:, 2], label[:, 3] - label[:, 1]])  # the height and width of ground truth boxes
    gt_h_w = gt_h_w.transpose((1, 0))  # (M, 2), 2:(height, width)
    scale = (gt_h_w[:, 0] * gt_h_w[:, 1]) / (height * width)  # (M, )

    pos_mask = np.full((anchors.shape[0], ), False)  # (N, )
    for gt_box in label[:, 1:]:  # gt_box shape: (4, )
        # strategy 1
        rltv_gt_box = myutils.bbox_abs_to_rel(gt_box.reshape((-1, 4)), img_size)  # (1, 4)
        ious = gcv.utils.bbox.bbox_iou(rltv_gt_box.asnumpy(), anchors) # (1, N)
        # ious = mx.nd.contrib.box_iou(mx.nd.array(rltv_gt_box), mx.nd.array(anchor))  
        ious = mx.nd.array(ious)
        max_iou_idx = mx.nd.argmax(ious)
        pos_mask[max_iou_idx] = True

        # strategy 2
        ious = gcv.utils.bbox_iou(rltv_gt_box.asnumpy(), anchors)
        ious = ious[0]  # (N, )
        pos_mask = mx.nd.where(ious > 0.2, mx.nd.full(pos_mask.shape, True), pos_mask)
        # pos_mask[np.where(ious > 0.2)] = True

    mask_not_zero_idx = np.where(pos_mask == True)[0]

    box_target = mx.nd.zeros(anchors.shape)
    cls_target = mx.nd.zeros(anchors.shape[0])
    for anchor_idx in mask_not_zero_idx:
        achr = anchors[anchor_idx]  # (4, )

        rltv_gt_boxes = myutils.bbox_abs_to_rel(label[:, 1:], img_size)
        ious = mx.nd.contrib.box_iou(mx.nd.array(rltv_gt_boxes), mx.nd.array(achr.reshape((-1, 4))))
        ious = ious.asnumpy().flatten()  # (M, )
        max_iou_idx = np.argmax(ious)

        rltv_gt_box = rltv_gt_boxes[max_iou_idx]  # (4, )

        achr_center_x = np.mean(achr[[0, 2]])
        achr_center_y = np.mean(achr[[1, 3]])
        achr_h = achr[3] - achr[1]
        achr_w = achr[2] - achr[0]

        gt_center_x = np.mean(rltv_gt_box[[0, 2]])
        gt_center_y = np.mean(rltv_gt_box[[1, 3]])
        gt_h = rltv_gt_box[3] - rltv_gt_box[1]
        gt_w = rltv_gt_box[2] - rltv_gt_box[0]

        box_target[anchor_idx, 0] = (gt_center_x - achr_center_x) / achr_w / 0.1
        box_target[anchor_idx, 1] = (gt_center_y - achr_center_y) / achr_h / 0.1
        box_target[anchor_idx, 2] = np.log(gt_w / achr_w) / 0.2
        box_target[anchor_idx, 3] = np.log(gt_h / achr_h) / 0.2

        cls_target[anchor_idx] = label[max_iou_idx, 0] + 1  # (N, )
    
    if not do_hard_mining:
        return box_target, pos_mask, cls_target

    # hard negative mining
    neg_mask = _hard_negative_mining(mx_img, mx_label, tensor_pred, anchor, pos_mask, neg_thresh)

    return cls_target, box_target, pos_mask, neg_mask


def generate_batch_target(img_size, labels, anchor):
    box_targs = np.array([])
    box_masks = np.array([])
    cls_targs = np.array([])
    for label in labels:  # (N, 5)
        label_pad_idx = label[:, 0] == -1
        label = label[~label_pad_idx]
        box_targ, box_mask, cls_targ = generate_target(img_size, anchor, label)
        box_targ = np.expand_dims(box_targ, axis=0)  # (1, N, 4), N denotes the total number of anchors
        box_mask = np.expand_dims(box_mask, axis=0)  # (1, N)
        cls_targ = np.expand_dims(cls_targ, axis=0)  # (1, N)
        if box_targs.size == 0:
            box_targs = box_targ
            box_masks = box_mask
            cls_targs = cls_targ
        else:
            box_targs = np.concatenate((box_targs, box_targ))
            box_masks = np.concatenate((box_masks, box_mask))
            cls_targs = np.concatenate((cls_targs, cls_targ))
    return box_targs, box_masks, cls_targs


def _hard_negative_mining(mx_img, mx_label, tensor_pred, anchors, pos_mask, neg_thresh=0.2):
    """

    :param mx_img: (1, 3, h, w)
    :param mx_label: (1, N, 5), N:objects in the image, 5:(cls_id, xmin, xmax, ymin, ymax), absolute
    :param tensor_pred: mx.nd.array, shape:(1, P, A, C+1+4), P:number of positions, A:anchors on each position
    :param anchors: (P*A, 4), 4:(xmin, xmax, ymin, ymax), relative
    :param neg_thresh: threshold of IoU for labeled as negatives
    :return:
    """
    P, A = tensor_pred.shape[1], tensor_pred.shape[2]
    # get negative indices
    label = mx_label.asnumpy()[0]
    label[:, 1:] = myutils.bbox_abs_to_rel(label[:, 1:], mx_img.shape[-2:])
    ious = gcv.utils.bbox_iou(label[:, 1:], anchors)
    neg_masks = []
    for iou in ious:
        neg_masks.append(iou < neg_thresh)
    neg_mask = np.full(anchors.shape[0], True)
    for mask in neg_masks:
        neg_mask *= mask
    
    neg_indices = np.where(neg_mask)[0]  # shape: (P*A, )
    num_negative = neg_indices.size

    # get positive indices
    pos_indices = np.where(pos_mask.flatten())[0]  # shape: (P*A, )
    num_positive = pos_indices.size

    # separate the positive indices from negative indices
    neg_indices = list(set(neg_indices) - set(pos_indices))

    # sort the background confidence at raising order
    temp_tensor_pred = tensor_pred.reshape((1, -1, 25))
    bg_conf_with_idx = np.array([[temp_tensor_pred[0, i, 0].asscalar(), i] for i in neg_indices])  # (N, 2)
    sorted_bg_conf_idx = np.argsort(bg_conf_with_idx[:, 0])

    # pick hard negatives
    num_hard_negative = 3 * num_positive
    if num_hard_negative > num_negative:
        num_hard_negative = num_negative
    hard_neg_idx = np.array(neg_indices)[sorted_bg_conf_idx[:num_hard_negative]]

    # generate mask
    neg_mask = np.full((P*A, ), False)
    neg_mask[hard_neg_idx] = True

    return neg_mask    # box_mask


def batch_hard_negative_mining(mx_imgs, mx_labels, tensor_preds, anchors, neg_thresh=0.2):
    if mx_imgs.shape[0] != mx_labels.shape[0] or mx_imgs.shape[0] != tensor_preds.shape[0]:
        raise Exception('the length of batch mismatch.')
    
    masks = mx.nd.array([])
    for i in range(len(mx_imgs)):
        mx_img = mx_imgs[i].expand_dims(axis=0)
        mx_label = mx_labels[i].expand_dims(axis=0)
        tensor_pred = tensor_preds[i:i+1]  # (b, c, h, w)
        mask = hard_negative_mining(mx_img, mx_label, tensor_pred, anchors, neg_thresh)
        mask = mask.reshape((1, -1))
        if masks.size == 0:
            masks = mask
        else:
            masks = np.concatenate((masks, mask), axis=0)
    return masks


# define loss
calc_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
calc_box_loss = mx.gluon.loss.L1Loss()


def calc_loss(tensor_pred, box_targ, cls_targ, pos_mask, neg_mask):
    """

    :param tensor_pred: mx.nd.array, (b, c, h, w)
    :param box_targ: np.array
    :param box_mask: np.array
    :param cls_targ: np.array
    :return:
    """
    anchor_num = tensor_pred.shape[1] * tensor_pred.shape[2]

    box_pred = tensor_pred[:, :, :, -4:].reshape((0, -1))  # (b, 4*N), N: number of anchors
    cls_pred = tensor_pred[:, :, :, :21].reshape((0, -1, 21))  # (b, N, 21)

    box_targ = mx.nd.array(box_targ.reshape((1, -1)), ctx=box_pred.context)  # (1, 4*N), N: number of anchors
    pos_mask = pos_mask.reshape((1, -1))  # (1, N)
    neg_mask = neg_mask.reshape((1, -1))
    cls_targ = mx.nd.array(cls_targ.reshape((1, -1)), ctx=cls_pred.context)  # (1, N)

    cls_mask = mx.nd.ones(cls_pred.shape, ctx=cls_pred.context)
    box_mask = mx.nd.ones(box_pred.shape, ctx=box_pred.context)

    cls_mask[np.where(pos_mask + neg_mask)] = 0
    box_mask[np.where(pos_mask)] = 0

    cls_loss = calc_cls_loss(cls_pred * cls_mask, cls_targ * cls_mask[:, :, 0])
    box_loss = calc_box_loss(box_pred * box_mask, box_targ * box_mask)
    loss = (cls_loss + box_loss)
    return loss


def get_pred_scores_classes_and_boxes_for_matric(tensor_pred, anchors):
    """
    输出无 rank 的 result，用于 metric  
    输出的坐标是 relative 的  

    tensor_pred: mx.nd.array, (b, N, 25)
    anchors: mx.nd.array, (1, N, 4)
    """
    if not isinstance(tensor_pred, mx.nd.NDArray):
        tensor_pred = mx.nd.array(tensor_pred)
    if not isinstance(anchors, mx.nd.NDArray):
        anchors = mx.nd.array(anchors, ctx=tensor_pred.context)

    box_pred = tensor_pred[:, :, -4:].reshape((0, -1))  # (b, 4*N), N: number of anchors
    cls_preds = tensor_pred[:, :, :21].reshape((0, -1, 21))  # (b, N, 21)

    cls_probs = cls_preds.softmax().transpose(axes=(0, 2, 1))  # (b, 21, N)
    detect = mx.contrib.nd.MultiBoxDetection(cls_probs,
                                             box_pred,
                                             anchors,  # (1, N, 4)
                                             threshold=0.01,
                                             nms_threshold=0.2,
                                             nms_topk=200)
    detection_output = detect
    temp = detection_output[:, :, 0].copy()
    detection_output[:, :, 0] = detection_output[:, :, 1]
    detection_output[:, :, 1] = temp
    return detection_output  # (b, N, 6)


def get_batch_pred_scores_cls_boxes(tensor_preds, img_sizes, anchors):
    if not isinstance(tensor_preds, mx.nd.NDArray):
        tensor_preds = mx.nd.array(tensor_preds)

    batchify_fn = gcv.data.batchify.Pad(pad_val=-1)
    batch_scores_cls_boxes = []
    for i, tensor_pred in enumerate(tensor_preds):
        img_size = img_sizes[i]
        tensor_pred = tensor_pred.expand_dims(axis=0)
        scores_cls_boxes = get_pred_scores_classes_and_boxes(tensor_pred, img_size, anchors)
        batch_scores_cls_boxes.append(scores_cls_boxes)
    batch_scores_cls_boxes = batchify_fn(batch_scores_cls_boxes)
    batch_scores_cls_boxes = batch_scores_cls_boxes.asnumpy()
    return batch_scores_cls_boxes


def visualize_mask(img, label, cls_targ, box_mask, anchors, feature_map_shapes, anchor_num_per_position):
    myutils.data_visualize(img, label[:, 1:])
    figs = []
    figs.append(plt.gcf())

    print('anchors matched:', np.where(box_mask)[0].size)
    box_mask_on_feat_map = []
    cls_on_feat_map = []
    cls_target_copy = cls_targ.copy()
    box_mask_copy = box_mask.copy()
    for featur_map_shape in feature_map_shapes:
        feature_map_size = featur_map_shape[-2:]
        slice_idx = anchor_num_per_position * np.multiply(*feature_map_size)
        box_mask_on_feat_map.append(box_mask_copy[:slice_idx].reshape((feature_map_size[0], feature_map_size[1], anchor_num_per_position)))
        cls_on_feat_map.append(cls_target_copy[:slice_idx].reshape((feature_map_size[0], feature_map_size[1], anchor_num_per_position)))
        cls_target_copy = cls_target_copy[slice_idx:]
        box_mask_copy = box_mask_copy[slice_idx:]
    
    mask_stat_on_feat_map = []
    mask_stat_max_value = 0
    for mask in box_mask_on_feat_map:
        mask_stat = np.sum(mask, axis=2)
        mask_stat_on_feat_map.append(mask_stat)
        if mask_stat_max_value < np.max(mask_stat):
            mask_stat_max_value = np.max(mask_stat)

    cls_stat_on_feat_map = []
    for cls in cls_on_feat_map:
        cls_stat = cls.max(axis=2)
        cls_stat_on_feat_map.append(cls_stat)

    fig = plt.figure()
    axes = fig.subplots(nrows=1, ncols=len(feature_map_shapes))
    for i, mask_stat in enumerate(mask_stat_on_feat_map):
        axes[i].imshow(mask_stat, vmin=0, vmax=6)
    fig.set_figwidth(16)
    fig.set_figheight(fig.get_figwidth() / 5)
    figs.append(fig)

    fig = plt.figure()
    gradient = np.linspace(0, 1, 20)
    gradient = np.vstack((gradient, gradient))
    a = fig.gca().imshow(gradient)
    a.axes.set_axis_off()
    fig.set_figwidth(8)
    fig.set_figheight(fig.get_figwidth() / 5)
    figs.append(fig)

    fig = plt.figure()
    axes = fig.subplots(nrows=1, ncols=len(feature_map_shapes))
    for i, cls_stat in enumerate(cls_stat_on_feat_map):
        axes[i].imshow(cls_stat, cmap=cm.tab20, vmin=0, vmax=20)
    fig.set_figwidth(16)
    fig.set_figheight(fig.get_figwidth() / 5)
    figs.append(fig)

    fig = plt.figure()
    gradient = np.linspace(0, 1, 20)
    gradient = np.vstack((gradient, gradient))
    a = fig.gca().imshow(gradient, cmap=cm.tab20)
    a.axes.set_axis_off()
    fig.set_figwidth(8)
    fig.set_figheight(fig.get_figwidth() / 5)
    figs.append(fig)

    return figs


class DynamicRankMatcher(mx.gluon.Block):
    def __init__(self):
        super(DynamicRankMatcher, self).__init__()
        return
    
    def forward(self, x, gt_boxes):
        """
        x: ious (b, N, M)  # ious 中 padding 的 ious 和 iou = 0 的 ious 都为零
        gt_boxes: (b, M, 4)  # 这里的 gt_boxes 也是经过 padding 的
        """
        ious = x.copy()
        match = -1 * mx.nd.ones(ious.shape[:2], ctx=x.context)  # (b, N)
        for i in range(ious.shape[0]):
            for j in range(ious.shape[2]):
                if gt_boxes[i, j, 0] == -1:
                    break
                iou = ious[i, :, j]  # (N, )   # 为什么这里用的是 slice，却改变不了 x 的值
                anchor_iou_order = mx.nd.argsort(iou, is_ascend=False)  # (N, )
                # 这里需要再增加一些代码，检测是否前 40 个不为 -1。
                match[i, anchor_iou_order[:30]] = j
                ious[i, anchor_iou_order[:30], :] = -1
        return match  # (b, N)


class DynamicMatcherTest(mx.gluon.Block):
    def __init__(self):
        super(DynamicMatcherTest, self).__init__()
        return
    
    def forward(self, x, gt_boxes):
        """
        x: ious (b, N, M)  # 假设被忽略的 sample 为 -1
        gt_boxes: (b, M, 4)
        """
        obj_xmin = mx.nd.slice_axis(gt_boxes, axis=2, begin=0, end=1)
        obj_xmax = mx.nd.slice_axis(gt_boxes, axis=2, begin=2, end=3)
        obj_ymin = mx.nd.slice_axis(gt_boxes, axis=2, begin=1, end=2)
        obj_ymax = mx.nd.slice_axis(gt_boxes, axis=2, begin=3, end=4)

        obj_width = obj_xmax - obj_xmin
        obj_height = obj_ymax - obj_ymin
        # 因为 gt_boxes 的数据已经是 relative 格式的了，所以这里只需要相将宽高相乘就可以了。 
        obj_scales = obj_width * obj_height  # (b, M, 1)
        obj_scales = mx.nd.repeat(obj_scales, repeats=x.shape[1], axis=2)  # (b, M, N)
        obj_scales = mx.nd.transpose(obj_scales, axes=(0, 2, 1))  # (b, N, M)

        # select padding ground-truth boxes
        ignore_mask = mx.nd.where(obj_xmin[:, :, 0] < 0, 
                                mx.nd.ones_like(obj_xmin[:, :, 0]), 
                                mx.nd.zeros_like(obj_xmin[:, :, 0]))  # (b, M)
        ignore_mask = mx.nd.expand_dims(ignore_mask, axis=1)  # (b, 1, M)
        ignore_mask = mx.nd.repeat(ignore_mask, obj_scales.shape[1], axis=1)  # (b, N, M)

        obj_scales = mx.nd.where(ignore_mask > 0, (-1) * mx.nd.ones_like(obj_scales), obj_scales)
        thresh = mx.nd.where((0 < obj_scales) * (obj_scales <= 0.05), 
                             0.2 * mx.nd.ones_like(obj_scales), 
                             (-1) * mx.nd.ones_like(obj_scales))
        thresh = mx.nd.where((0.05 < obj_scales) * (obj_scales <= 0.5),
                             1.5 * obj_scales + 0.125,
                             thresh)
        thresh = mx.nd.where(obj_scales > 0.5, 0.5 * mx.nd.ones_like(obj_scales), thresh)  # (b, N, M)
        argmax = mx.nd.argmax(x, axis=-1)  # (b, N)
        # max_iou = mx.nd.pick(x, argmax, axis=-1)
        # max_thresh = mx.nd.pick(thresh, argmax, axis=-1)
        matches = mx.nd.where(mx.nd.pick(x, argmax, axis=-1) >= mx.nd.pick(thresh, argmax, axis=-1), 
                              argmax, 
                              mx.nd.ones_like(argmax) * -1) # (b, N)
        return matches


scales = []
class DynamicMatcher(mx.gluon.Block):
    def __init__(self, **kwargs):
        super(DynamicMatcher, self).__init__(**kwargs)
        self._thresh = 0.2
        self._matcher = gcv.nn.matcher.MaximumMatcher(self._thresh)
        return

    def forward(self, x, anchors, gt_boxes):
        """
        x: ious (b, N, M)  # 被忽略的 iou 仍会被计算为 0
        anchors: (1, N, 4)
        gt_boxes: (b, M, 4)  # 被忽略的 gt box 的数据为 -1
        """
        padding_mask = mx.nd.where(gt_boxes[:, :, 0] > -0.5, 
            mx.nd.ones(gt_boxes.shape[:2], ctx=gt_boxes.context), 
            mx.nd.zeros(gt_boxes.shape[:2], ctx=gt_boxes.context))

        ious = x
        scale_anr = (anchors[:, :, 2] - anchors[:, :, 0]) * (anchors[:, :, 3] - anchors[:, :, 1])  # (b, N)
        scale_gt = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0]) * (gt_boxes[:, :, 3] - gt_boxes[:, :, 1])  # (b, M)
        scales.append(scale_gt)

        e = mx.nd.sqrt(1 - mx.nd.broadcast_sub(scale_anr.reshape((0, -1, 1)), scale_gt.reshape((0, 1, -1))) ** 2) * ious
        e = mx.nd.where(mx.nd.broadcast_like(padding_mask.expand_dims(axis=1), e), e, -1 * mx.nd.ones_like(e))  # (b, N, M)

        # print(scale_gt.expand_dims(axis=-1).shape)
        obj_scales = mx.nd.broadcast_axis(scale_gt.expand_dims(axis=1), axis=1, size=scale_anr.shape[1])  # (b, N, M)
        s1, s2, t1, t2 = 0.09274982, 0.18148671, 0.11818486, 0.17982725

        thresh = mx.nd.where((0 < obj_scales) * (obj_scales <= s1), 
                             t1 * mx.nd.ones_like(obj_scales), 
                             (-1) * mx.nd.ones_like(obj_scales))
        thresh = mx.nd.where((s1 < obj_scales) * (obj_scales <= s2),
                             (t2 - t1)/(s2 - s1) * obj_scales + (t1 - (t2 - t1)/(s2 - s1) * s1),
                             thresh)
        thresh = mx.nd.where(obj_scales > s2, t2 * mx.nd.ones_like(obj_scales), thresh)  # (b, N, M)

        argmax = mx.nd.argmax(x, axis=-1)  # (b, N)
        matches = mx.nd.where(mx.nd.pick(x, argmax, axis=-1) >= mx.nd.pick(thresh, argmax, axis=-1), 
                              argmax, 
                              mx.nd.ones_like(argmax) * -1) # (b, N)

        # matches = self._matcher(e)
        return matches

class DynamicMatcher_1(mx.gluon.Block):
    # 这个 matcher 按 GT box 的 scale 通过分段函数的方式对匹配阈值进行动态调整
    def __init__(self, param, **kwargs):
        super(DynamicMatcher_1, self).__init__(**kwargs)
        self._thresh = 0.2
        self._matcher = gcv.nn.matcher.MaximumMatcher(self._thresh)
        self._s1, self._s2, self._t1, self._t2 = mx.nd.array(param, ctx=param.context)
        return

    def forward(self, x, anchors, gt_boxes):
        """
        x: ious (b, N, M)  # 被忽略的 iou 仍会被计算为 0
        anchors: (1, N, 4)
        gt_boxes: (b, M, 4)  # 被忽略的 gt box 的数据为 -1
        """
        padding_mask = mx.nd.where(gt_boxes[:, :, 0] < -0.5, 
            mx.nd.ones(gt_boxes.shape[:2], ctx=gt_boxes.context), 
            mx.nd.zeros(gt_boxes.shape[:2], ctx=gt_boxes.context))  # (b, M)

        scale_gt = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0]) * (gt_boxes[:, :, 3] - gt_boxes[:, :, 1])  # (b, M)
        thresh = mx.nd.zeros_like(scale_gt, ctx=scale_gt.context)  # (b, M)
        thresh = mx.nd.where(scale_gt <= self._s1, self._t1.broadcast_like(thresh), thresh)
        thresh = mx.nd.where((self._s1 < scale_gt) * (scale_gt <= self._s2),
                             (self._t2 - self._t1)/(self._s2 - self._s1) * scale_gt + (self._t1 - (self._t2 - self._t1)/(self._s2 - self._s1) * self._s1),
                             thresh)
        thresh = mx.nd.where(scale_gt > self._s2, self._t2.broadcast_like(thresh), thresh)  # (b, M)
        thresh = mx.nd.where(padding_mask, 2 * mx.nd.ones_like(thresh, ctx=thresh.context), thresh)  # 因为 iou 注定无法达到 2，所以被 padding 的 GT boxes 必定匹配失败

        matches = mx.nd.argmax(x, axis=-1)  # (b, N)
        matches = mx.nd.where(mx.nd.pick(x, matches, axis=-1) >= thresh.expand_dims(axis=1).broadcast_like(x).pick(matches, axis=-1), 
                              matches,
                              mx.nd.array([-1], ctx=matches.context).broadcast_like(matches))  # (b, N)

        # matches = self._matcher(e)
        return matches


def generate_random_gt_boxes(b, n):
    batch_random_gt_boxes = mx.nd.array([])
    for _ in range(b):
        random_gt_boxes = mx.nd.array([])
        for _ in range(n):
            while True:
                boxes = mx.nd.random.uniform(0, 1, (1, 4))
                if boxes[0, 2] > boxes[0, 0] and boxes[0, 3] > boxes[0, 1]:
                    break
            if random_gt_boxes.size == 0:
                random_gt_boxes = boxes
            else:
                random_gt_boxes = mx.nd.concatenate([random_gt_boxes, boxes], axis=0)
        random_gt_boxes = random_gt_boxes.expand_dims(axis=0)
        if batch_random_gt_boxes.size == 0:
            batch_random_gt_boxes = random_gt_boxes
        else:
            batch_random_gt_boxes = mx.nd.concatenate([batch_random_gt_boxes, random_gt_boxes], axis=0)
    return batch_random_gt_boxes


def generate_target(anchors, cls_preds, gt_boxes, gt_ids,
                    iou_thresh=0.5, neg_thresh=0.5, negative_mining_ratio=3,
                    stds=(0.1, 0.1, 0.2, 0.2), dynamic_sampling=False):
    """
    gt_ids: (b, N, 1)
    anchors: shape: (1, N, 4), relative  
    cls_preds: shape: (b, N, C+1)
    """
    
    _maximum_matcher = gcv.nn.matcher.MaximumMatcher(iou_thresh)

    if dynamic_sampling:
        _biprtite_matcher = gcv.nn.matcher.BipartiteMatcher()
        # _dynamic_matcher = DynamicMatcher_1(mx.nd.array([0.1, 0.2, 0.1, 0.2], ctx=cls_preds.context))
        _dynamic_matcher = DynamicRankMatcher()

    if negative_mining_ratio > 0:
        _sampler = gcv.nn.sampler.OHEMSampler(negative_mining_ratio, thresh=neg_thresh)
        _use_negative_sampling = True
    else:
        _sampler = gcv.nn.sampler.NaiveSampler()
        _use_negative_sampling = False
    _cls_encoder = gcv.nn.coder.MultiClassEncoder()
    _box_encoder = gcv.nn.coder.NormalizedBoxCenterEncoder(stds=stds)

    # anchors = anchors.reshape((-1, 4))
    ious = mx.nd.transpose(mx.nd.contrib.box_iou(anchors.reshape((-1, 4)), gt_boxes), (1, 0, 2))

    if dynamic_sampling:
        matches_bip, _ = mx.nd.contrib.bipartite_matching(ious, is_ascend=False, threshold=1e-12)
        # matches_dyn = _dynamic_matcher(ious, anchors, gt_boxes)
        matches_dyn = _dynamic_matcher(ious, gt_boxes)
        # return matches_dyn
        matches = matches_bip
        matches = mx.nd.where(matches_bip > -0.5, matches_bip, matches_dyn)
        # print(np.where(matches[0].asnumpy() == 1)[0].size)
        # return matches
    else:
        matches_bip, _ = mx.nd.contrib.bipartite_matching(ious, is_ascend=False, threshold=1e-12)
        matches_max = _maximum_matcher(ious)
        matches = matches_bip
        matches = mx.nd.where(matches_bip > -0.5, matches_bip, matches_max)

    # if True:
    #     naive_sampler = gcv.nn.sampler.NaiveSampler()
    #     pos_samples = naive_sampler(matches)  # pos: > 0, neg: < 0. no ignore
    #     if _use_negative_sampling:
    #         ohem_sampler = gcv.nn.sampler.OHEMSampler(ratio=negative_mining_ratio, thresh=neg_thresh)
    #         pos_neg_samples = ohem_sampler(matches, cls_preds, ious)
    #     else:
    #         pos_neg_samples = None

    ohem_sampler = gcv.nn.sampler.OHEMSampler(ratio=negative_mining_ratio, thresh=neg_thresh)
    pos_neg_samples = ohem_sampler(matches, cls_preds, ious)
    
    # if _use_negative_sampling:
    #     samples = _sampler(matches, cls_preds, ious)  # sample > 0 表示 pos，sample < 0 表示 neg
    # else:
    #     samples = _sampler(matches)

    cls_targets = _cls_encoder(pos_neg_samples, matches, gt_ids)
    box_targets, box_masks = _box_encoder(pos_neg_samples, matches, anchors, gt_boxes)

    return cls_targets, box_targets, pos_neg_samples
    
    # return cls_targets, box_targets, box_masks


def ssd_loss(cls_preds, box_preds, cls_targs, box_targs, pos_neg_samples):
    """
    使用 gluoncv 自带的 OHEM class 来做 negative mining，其余的部分基本都是按照 gluoncv 的
    MultiBoxLoss 来写的。  
    这部分 loss 不包含 confidence loss。
    """
    # cls_masks: (b, anchor_num)
    # box_masks: (b, anchor_num, 4)

    _lambd = 1
    _rho = 1
    _negative_mining_ratio = 3

    cls_preds_copy = cls_preds.copy()
    cls_targs_copy = cls_targs.copy()
    box_targs_copy = box_targs.copy()
    box_preds_copy = box_preds.copy()

    pos_mask = mx.nd.where(pos_neg_samples > 0,
                            mx.nd.ones_like(pos_neg_samples),
                            mx.nd.zeros_like(pos_neg_samples))
    neg_mask = mx.nd.where(pos_neg_samples < 0,
                            mx.nd.ones_like(pos_neg_samples),
                            mx.nd.zeros_like(pos_neg_samples))

    cp = cls_preds_copy
    pred = mx.nd.log_softmax(cp, axis=-1)
    pos = pos_mask
    cls_loss = -1 * mx.nd.pick(pred, cls_targs_copy, axis=-1, keepdims=False)
    
    hard_negative = neg_mask
    # mask out if not positive or negative
    cls_loss = mx.nd.where((pos + hard_negative) > 0, cls_loss, mx.nd.zeros_like(cls_loss))  # (b, N)
    cls_losses = mx.nd.sum(cls_loss, axis=0, exclude=True)  # 对 batch 中的每张图片计算 loss
    num_pos_all = mx.nd.sum(pos, axis=0, exclude=True)
    cls_losses = cls_losses / mx.nd.sum(num_pos_all)  ## !!!!???

    # bp = _reshape_like(box_preds, box_targs)
    bt = box_targs_copy
    bp = box_preds_copy
    box_loss = mx.nd.abs(bp - bt)
    box_loss = mx.nd.where(box_loss > _rho, box_loss - 0.5 * _rho,
                        (0.5 / _rho) * mx.nd.square(box_loss))
    # box loss only apply to positive samples
    box_loss = box_loss * pos.expand_dims(axis=-1)
    box_losses = mx.nd.sum(box_loss, axis=0, exclude=True) / mx.nd.sum(num_pos_all)
    # box_losses.append(nd.sum(box_loss, axis=0, exclude=True))
    sum_losses = cls_losses + _lambd * box_losses

    return sum_losses, cls_losses, box_losses


class SSDLoss(mx.gluon.Block):
    def __init__(self, rho=1.0, lambd=1.0, **kwargs):
        super(SSDLoss, self).__init__(**kwargs)
        self._rho = rho
        self._lambda = lambd
        self._cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        self._box_loss = mx.gluon.loss.L1Loss()
    
    def forward(self, cls_preds, box_preds, cls_targs, box_targs, cls_masks, pos_neg_masks):
        # cls_masks: (b, anchor_num)
        # box_masks: (b, anchor_num, 4)
        cls_masks = mx.nd.where((pos_neg_masks > 0) + (pos_neg_masks < 0), 
                                mx.nd.ones_like(pos_neg_masks), 
                                mx.nd.zeros_like(pos_neg_masks))  # (b, N)
        box_masks = mx.nd.where(pos_neg_masks > 0, 
                                    mx.nd.ones_like(pos_neg_masks), 
                                    mx.nd.zeros_like(pos_neg_masks))  # (b, N)
        
        box_masks = mx.nd.repeat(box_masks.reshape((0, -1, 1)), 4, axis=-1)  # (b, N, 4)
        cls_logit_masks = mx.nd.repeat(cls_masks.reshape((0, -1, 1)), 21, axis=-1)

        cls_loss = self._cls_loss(cls_preds * cls_logit_masks, cls_targs)
        box_loss = self._box_loss(box_preds * box_masks, box_targs)
        loss = cls_loss + self._lambda * box_loss

        return loss

calc_ssd_loss = SSDLoss()

class SSDMetric():
    def __init__(self, class_names, anchors):
        """
        anchors: (N, 4)
        """
        self._metric = gcv.utils.metrics.voc_detection.VOCMApMetric(iou_thresh=0.5, class_names=class_names)
        self._anchors = mx.nd.array(anchors).expand_dims(axis=0)  # (1, N, 4)
    def reset(self):
        self._metric.reset()
    def update(self, tensor_preds, imgs, labels):
        """
        tensor_preds: mx.nd.NDArray, (b, N, 25)
        imgs: list
        labels: list
        """
        batch_scores_cls_boxes = get_pred_scores_classes_and_boxes_for_matric(tensor_preds, self._anchors.as_in_context(tensor_preds.context))
        parsed_detection_output = myutils.parse_batch_detection_outputs(batch_scores_cls_boxes, labels)
        self._metric.update(*parsed_detection_output)
    def get(self):
        return self._metric.get()


def generate_anchor(sizes, ratios, x=0.5, y=0.5, s_k=0.5, style='faster_rcnn'):
    """
    产生 Faster R-CNN style 的 anchors，因此如果 sizes 和 ratios 的 len 都是 3，那么会产生 9 个 anchors。  
    整个函数使用的全都是相对坐标。  
    x, y: anchor 的中心点的坐标  
    sizes：list，anchor 的大小  
    ratios：list，anchor 的宽长比例  
    s_k: 当 size = 1, ratio = 1 时 anchor 的边长与图片边长的比例。  
    return anchors: np.array, (N, 4), 返回 N 个相对坐标。
    """
    
    anchors = []
    if style == 'faster_rcnn':
        for s in sizes:
            for r in ratios:
                w, h = s_k * s * np.sqrt(r), s_k * s / np.sqrt(r)
                coor_x1 = x - w/2
                coor_y1 = y - h/2
                coor_x2 = x + w/2
                coor_y2 = y + h/2
                anchors.append([coor_x1, coor_y1, coor_x2, coor_y2])

    if style == 'ssd':
        for i, s in enumerate(sizes):
            if i == 0:
                for r in ratios:
                    w = s_k * s * np.sqrt(r)
                    h = s_k * s / np.sqrt(r)
                    coor_x1 = x - w/2
                    coor_y1 = y - h/2
                    coor_x2 = x + w/2
                    coor_y2 = y + h/2
                    anchors.append([coor_x1, coor_y1, coor_x2, coor_y2])
            else:
                w = s_k * s
                h = s_k * s
                coor_x1 = x - w/2
                coor_y1 = y - h/2
                coor_x2 = x + w/2
                coor_y2 = y + h/2
                anchors.append([coor_x1, coor_y1, coor_x2, coor_y2])
                
    anchors = np.array(anchors)
    return anchors


if __name__ == '__main__':
    # 产生一组 ssd style 的 anchor
    anchors = generate_anchor([1, 0.45, 0.2], [1, 0.5, 2], s_k=0.4, style='ssd')
    import dataset_utils
    root_path = r'D:\Documents\Data_Files\Datasets\Pascal\VOC2012'
    dataset = dataset_utils.Dataset(root_path)
    img, label = dataset[0]
    plt.imshow(img)
    myutils.visualize_boxes(anchors[:3], 'red', fig=plt.gcf(), is_rltv_cor=True, img_size=img.shape[:2])
    myutils.visualize_boxes(anchors[3:4], 'blue', fig=plt.gcf(), is_rltv_cor=True, img_size=img.shape[:2])
    myutils.visualize_boxes(anchors[4:5], 'green', fig=plt.gcf(), is_rltv_cor=True, img_size=img.shape[:2])

    if not plt.gca().yaxis_inverted():
        plt.gca().invert_yaxis()
    plt.show()
