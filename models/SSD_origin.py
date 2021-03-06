import numpy as np
import myutils
import mxnet as mx
import gluoncv as gcv

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

        # 如果有时间的话可以把换成 vgg16
        backbone = mx.gluon.model_zoo.vision.resnet50_v2(pretrained=True, ctx=self.ctx, root=backbone_root_path)

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

    # 有时间的话，其实这里可以改写成 get_transform_fn(*args)
    # 然后在函数内自己解析参数：self, img, label = args
    # 这样就可以使用多线程读取数据了
    # 下面的 get_transform_fn_val() 同理
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


class TargetGenerator(mx.gluon.Block):
    def __init__(self, iou_thresh=0.5, neg_thresh=0.5, negative_mining_ratio=3, stds=(0.1, 0.1, 0.2, 0.2), **kwargs):
        super(TargetGenerator, self).__init__(**kwargs)
        self._maximum_matcher = gcv.nn.matcher.MaximumMatcher(iou_thresh)
        self._ohem_sampler = gcv.nn.sampler.OHEMSampler(negative_mining_ratio, thresh=neg_thresh)
        self._cls_encoder = gcv.nn.coder.MultiClassEncoder()
        self._box_encoder = gcv.nn.coder.NormalizedBoxCenterEncoder(stds=stds)
        return
    
    def forward(self, x, anchors, gt_boxes, gt_ids):
        """
        x: cls_preds
        gt_ids: (b, N, 1)
        anchors: shape: (1, N, 4), relative  
        cls_preds: shape: (b, N, C+1)
        """
        cls_preds = x

        ious = mx.nd.transpose(mx.nd.contrib.box_iou(anchors.reshape((-1, 4)), gt_boxes), (1, 0, 2))
        matches_bip, _ = mx.nd.contrib.bipartite_matching(ious, is_ascend=False, threshold=1e-12)
        matches_max = self._maximum_matcher(ious)
        matches = matches_bip
        matches = mx.nd.where(matches_bip > -0.5, matches_bip, matches_max)
        pos_neg_samples = self._ohem_sampler(matches, cls_preds, ious)
        cls_targets = self._cls_encoder(pos_neg_samples, matches, gt_ids)
        box_targets, box_masks = self._box_encoder(pos_neg_samples, matches, anchors, gt_boxes)
        return cls_targets, box_targets, pos_neg_samples

    
def ssd_loss(cls_preds, box_preds, cls_targs, box_targs, pos_neg_samples):
    """
    使用 gluoncv 自带的 OHEM class 来做 negative mining，其余的部分基本都是按照 gluoncv 的
    MultiBoxLoss 来写的。
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
    cls_loss = -1 * mx.nd.pick(pred, cls_targs_copy, axis=-1, keepdims=False)

    # mask out if not positive or negative
    cls_loss = mx.nd.where((pos_mask + neg_mask) > 0, cls_loss, mx.nd.zeros_like(cls_loss))  # (b, N)
    cls_losses = mx.nd.sum(cls_loss, axis=0, exclude=True)  # 对 batch 中的每张图片计算 loss
    num_pos_all = mx.nd.sum(pos_mask, axis=0, exclude=True)
    cls_losses = cls_losses / mx.nd.sum(num_pos_all)  ## !!!!???

    bt = box_targs_copy
    bp = box_preds_copy
    box_loss = mx.nd.abs(bp - bt)
    box_loss = mx.nd.where(box_loss > _rho, box_loss - 0.5 * _rho,
                        (0.5 / _rho) * mx.nd.square(box_loss))
    # box loss only apply to positive samples
    box_loss = box_loss * pos_mask.expand_dims(axis=-1)
    box_losses = mx.nd.sum(box_loss, axis=0, exclude=True) / mx.nd.sum(num_pos_all)

    sum_losses = cls_losses + _lambd * box_losses

    return sum_losses, cls_losses, box_losses


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



