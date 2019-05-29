import mxnet as mx
import numpy as np
import gluoncv as gcv
# from .. import myutils
import myutils

class SSDBinMulti(mx.gluon.HybridBlock):
    def __init__(self, backbone_root_path=None, ctx=mx.cpu(), **kwargs):
        super(SSDBinMulti, self).__init__(**kwargs)

        if not backbone_root_path:
            raise Exception('backbone parameter path is None.')

        self.ctx = ctx
        self.model_img_size = (300, 300)
        self.anchors = np.array([])
        self.feature_map_shapes_list = []
        self.anchor_num_per_position = 6
        self.start_layer_idx = 6  # idx: 6~10 (included)
        self._num_feature_maps = 5  # build-in feature maps: 3, expanded feature maps: 2

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
            self.con_predictors = []
            for i in range(self.start_layer_idx, len(self.features)):
                setattr(self, 'con_predictor_' + str(i), 
                        mx.gluon.nn.Conv2D(self.anchor_num_per_position * 2, kernel_size=(3, 3), strides=1, padding=1))
                setattr(self, 'cls_predictor_' + str(i),
                        mx.gluon.nn.Conv2D(self.anchor_num_per_position * 20, kernel_size=(3, 3), strides=1, padding=1))
                setattr(self, 'box_predictor_' + str(i),
                        mx.gluon.nn.Conv2D(self.anchor_num_per_position * 4, kernel_size=(3, 3), strides=1, padding=1))
                eval('self.con_predictor_' + str(i) + '.initialize(mx.init.Xavier(), ctx=self.ctx)')
                eval('self.cls_predictor_' + str(i) + '.initialize(mx.init.Xavier(), ctx=self.ctx)')
                eval('self.box_predictor_' + str(i) + '.initialize(mx.init.Xavier(), ctx=self.ctx)')
                self.con_predictors.append(eval('self.con_predictor_' + str(i)))
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
            img = img.astype('float32') / 255  # deepcopy
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
            img = img.astype('float32') / 255  # deepcopy
            label = label.astype('float32')

            aug_img, aug_label = myutils.data_augment(img, label, size=self.model_img_size, rb=0.0, rc=0.0, rh=0.0, rs=0.0, 
                rflr=False, re=True, rcp=False)
            aug_img = mx.img.color_normalize(mx.nd.array(aug_img),
                                              mean=mx.nd.array(myutils.mean),
                                              std=mx.nd.array(myutils.std))
            mx_img = myutils.to_tensor(aug_img)
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
            cls_pred = self.cls_predictors[i - self.start_layer_idx](x)  # shape: (b, 5*C, h, w)
            box_pred = self.box_predictors[i - self.start_layer_idx](x)  # (b, 5*4, h, w)
            con_pred = self.con_predictors[i - self.start_layer_idx](x)  # (b, 2, h, w)
            tensor_pred_list.append(mx.nd.concat(
                con_pred.transpose((0, 2, 3, 1)).reshape((0, -1, self.anchor_num_per_position, 2)),
                cls_pred.transpose((0, 2, 3, 1)).reshape((0, -1, self.anchor_num_per_position, 20)),
                box_pred.transpose((0, 2, 3, 1)).reshape((0, -1, self.anchor_num_per_position, 4)),
                dim=3))

        tensor_pred = tensor_pred_list[0]
        for tensor in tensor_pred_list[1:]:
            tensor_pred = mx.nd.concat(tensor_pred, tensor, dim=1)

        return tensor_pred.reshape((0, -1, 26))  # (b, N, 26)

def generate_target2(anchors, tensor_preds, mx_labels, 
                    iou_thresh=0.5, neg_thresh=0.5, negative_mining_ratio=3, 
                    stds=(0.1, 0.1, 0.2, 0.2), dynamic_sampling=False):
    """
    anchors: shape: (1, num_anchors, 4), relative
    gt_ids: (b, M)
    """
    N = anchors.shape[1]
    con_preds = tensor_preds[:, :, :2]  # (b, N, 2)
    cls_preds = tensor_preds[:, :, 2:22]  # (b, N, 20)
    box_preds = tensor_preds[:, :, 22:]  # (b, N, 4)  这行好像没啥用

    gt_ids = mx_labels[:, :, 0:1]  # (b, M, 1)
    gt_boxes = mx_labels[:, :, 1:]  # (b, M, 4)
    
    _maximum_matcher = gcv.nn.matcher.MaximumMatcher(iou_thresh)
    _box_encoder = gcv.nn.coder.NormalizedBoxCenterEncoder(stds=stds)

    ious = mx.nd.transpose(mx.nd.contrib.box_iou(anchors[0], gt_boxes), (1, 0, 2))  # (b, N, M)

    # match confidence positives and negatives (without ohem) (dynamic rank)
    # match class positives
    ious_copy = ious.copy()  # (b, N, M)
    match_con = -mx.nd.ones(ious_copy.shape[:2], ctx=ious_copy.context)  # (b, N)
    match_cls = -1 * mx.nd.ones(cls_preds.shape[:2], ctx=ious_copy.context)  # (b, N)
    for i in range(ious_copy.shape[0]):
        for j in range(ious_copy.shape[2]):
            if gt_boxes[i, j, 0] == -1:
                break
            iou = ious_copy[i, :, j]  # (N, )
            anchor_iou_order = mx.nd.argsort(iou, is_ascend=False)  # (N, )
            # 这里需要再增加一些代码，检测是否前 40 个不为 -1。
            match_con[i, anchor_iou_order[:30]] = j
            match_cls[i, anchor_iou_order[:30]] = gt_ids[i, j, 0]
            ious_copy[i, anchor_iou_order[:30], :] = -1
    
    # 测试点：1. padding 部分都为 -1。2. gt box 所在地方都为 1， 每个 gt box 只有 30 个 1。3. 其余地方全为零。

    # generate pos samples (according to positive matches)
    pos_neg_sample = mx.nd.zeros(ious.shape[:2], ctx=tensor_preds.context)  # (b, N)
    pos_neg_sample = mx.nd.where(match_con >= 0, 
                                mx.nd.ones_like(pos_neg_sample), 
                                pos_neg_sample)  # (b, N)
    
    # generate neg samples (ohem on confidence)
    # 对于所有 gt_boxes 都有 iou < neg_thresh 的 anchor 作为 negatives
    # match_con == 0 是一个取反的 trick，因为 mxnet 里不能直接使用 ~ 符号取反
    all_negs = mx.nd.where(mx.nd.prod(ious < neg_thresh, axis=-1) * (match_con < 0), 
                            mx.nd.ones_like(match_con), 
                            mx.nd.zeros_like(match_con))  # (b, N)，大于 0 部分表示 neg
    # 这里的 con_loss 实际上取的是预测为 1 的 logits，即 false positive，logits 越大，意味着预测得越不对
    con_loss = mx.nd.where(all_negs, con_preds[:, :, 1], - np.inf * mx.nd.ones_like(all_negs))  # (b, N)
    # 这里是一个 trick，本来只能用 np.where(np.argsort() < neg_num) 实现找到 index 的方法，
    # 现在通过两个 argsort() 可以直接应用到矩阵上
    con_loss_rank = con_loss.argsort(axis=1, is_ascend=False).argsort(axis=1)  # (b, N)
    hard_negs = con_loss_rank < mx.nd.sum(pos_neg_sample > 0, axis=-1, keepdims=True) * negative_mining_ratio  # (b, N)
    pos_neg_sample = mx.nd.where(hard_negs, -1 * mx.nd.ones_like(pos_neg_sample), pos_neg_sample)  # (b, N)
    
    # generate box regression target (only pos)
    box_targs, _ = _box_encoder(pos_neg_sample, match_con, anchors, gt_boxes)  # (b, N, 4)

    # generate confidence target (pick pos and neg)
    con_targs = mx.nd.where(pos_neg_sample > 0, 
                            mx.nd.ones_like(pos_neg_sample),
                            -1 * mx.nd.ones_like(pos_neg_sample))  # (b, N)
    con_targs = mx.nd.where(pos_neg_sample < 0,
                            mx.nd.zeros_like(pos_neg_sample),
                            con_targs)  # (b, N)

    # generate class target (only pos)
    cls_targs = mx.nd.where(pos_neg_sample > 0, 
                            match_cls,
                            -1 * mx.nd.ones_like(pos_neg_sample)) # (b, N)
    
    return con_targs, cls_targs, box_targs, pos_neg_sample

def calc_loss2(tensor_preds, epoch, con_targs, cls_targs, box_targs, pos_neg_samples):
    """
    tensor_preds: (b, N, 26)
    con_targs: (b, N)
    cls_targs: (b, N)
    box_targs: (b, N, 4)
    pos_neg_samples: (b, N)
    """

    con_preds = tensor_preds[:, :, :2]  # (b, N, 2)
    cls_preds = tensor_preds[:, :, 2:22]  # (b, N, 20)
    box_preds = tensor_preds[:, :, -4:]  # (b, N, 4)

    loss_con = -mx.nd.pick(mx.nd.log_softmax(con_preds), con_targs)  # (b, N)
    loss_con = mx.nd.where(pos_neg_samples != 0, loss_con, mx.nd.zeros_like(loss_con))  # (b, N)
    loss_con = mx.nd.sum(loss_con, axis=0, exclude=True) / mx.nd.sum(pos_neg_samples != 0)  # (b, )

    loss_cls = -mx.nd.pick(mx.nd.log_softmax(cls_preds), cls_targs)  # (b, N)
    loss_cls = mx.nd.where(pos_neg_samples > 0, loss_cls, mx.nd.zeros_like(loss_cls))  # (b, N)
    loss_cls = mx.nd.sum(loss_cls, axis=0, exclude=True) / mx.nd.sum(pos_neg_samples > 0)  # (b, )

    # cls_logit_targs = mx.nd.one_hot(cls_targs, 20)  # (b, N, 20)  cls_targs 有 -1 所在的位置所有 one-hot logits 都为 0
    # loss_cls = (cls_preds - cls_logit_targs) ** 2  # (b, N, 20)
    # loss_cls = mx.nd.sum(loss_cls, axis=-1)  # (b, N)
    # loss_cls = mx.nd.where(pos_neg_samples != 0, loss_cls, mx.nd.zeros_like(loss_cls))  # (b, N)
    # loss_cls = mx.nd.sum(loss_cls, axis=0, exclude=True) / mx.nd.sum(pos_neg_samples != 0, axis=0, exclude=True)  # (b, )

    loss_box = mx.nd.abs(box_preds - box_targs)  # (b, N)
    loss_box = mx.nd.where(loss_box > 1, loss_box - 0.5, 0.5 * mx.nd.square(loss_box))
    loss_box = mx.nd.where(mx.nd.broadcast_like((pos_neg_samples > 0).expand_dims(axis=-1), loss_box), 
                            loss_box, 
                            mx.nd.zeros_like(loss_box))  # (b, N)
    loss_box = mx.nd.sum(loss_box, axis=0, exclude=True) / mx.nd.sum(pos_neg_samples > 0)  # (b, )

    loss = loss_con + loss_cls + loss_box
    loss = mx.nd.sum(loss)

    return loss, loss_con, loss_cls, loss_box

class SSDMetric2():
    def __init__(self, class_names, anchors):
        self._metric = gcv.utils.metrics.voc_detection.VOCMApMetric(iou_thresh=0.5, class_names=class_names)
        self._anchor_context = None
        self._anchors = anchors  # (N, 4)
    def reset(self):
        self._metric.reset()
    def update(self, tensor_preds, imgs, labels):
        """
        tensor_preds: mx.nd.NDArray, (b, N, 21)
        imgs: list
        labels: list
        """
        if not self._anchor_context:
            self._anchor_context = tensor_preds.context
            self._anchors = mx.nd.array(self._anchors, ctx=self._anchor_context).expand_dims(axis=0)
        # print(self._anchors.shape)
        batch_scores_cls_boxes = get_pred_scores_cls_boxes2(tensor_preds, self._anchors)
        parsed_detection_output = myutils.parse_batch_detection_outputs(batch_scores_cls_boxes, labels)
        self._metric.update(*parsed_detection_output)
    def get(self):
        return self._metric.get()
    def calc_ap(self, net, dataloader_val):
        self.reset()
        for mx_imgs, mx_labels in dataloader_val:
            tensor_preds = net(mx_imgs.as_in_context(net.ctx))
            self.update(tensor_preds, mx_imgs, mx_labels)
        return self.get()[1][-1]


def get_pred_scores_cls_boxes2(tensor_preds, anchors, is_sorted=False):
    """
    img_size: (2, ), 2:(height, width)
    anchors: np.array, (N, 4)
    """
    con_preds = tensor_preds[:, :, :2]  # (b, N, 2)
    cls_preds = tensor_preds[:, :, 2:22]  # (b, N, 20)
    box_preds = tensor_preds[:, :, -4:]  # (b, N, 4)

    cons = con_preds.softmax()[:, :, 1]  # (b, N)
    cls_probs = mx.nd.max(cls_preds.softmax(), axis=-1)  # (b, N)
    scores = cons * cls_probs  # (b, N)

    # scores = mx.nd.where(cons > 0.5, scores, mx.nd.zeros_like(scores))  # (b, N)
    scores = mx.nd.where(cls_probs > 0.2, scores, mx.nd.zeros_like(scores))  # (b, N)
    
    cls_ids = mx.nd.argmax(cls_preds, axis=-1)  # (b, N)

    _box_decoder = gcv.nn.coder.NormalizedBoxCenterDecoder(convert_anchor=True)
    boxes = _box_decoder(box_preds, anchors) # (b, N, 4)

    ids_scores_boxes = mx.nd.concatenate([cls_ids.expand_dims(axis=-1),
                                        scores.expand_dims(axis=-1),
                                        boxes], axis=2)  # (b, N, 6)  # 为啥这里的 axis 只能写 2，不能写 -1？
 
    ids_scores_boxes_nms = mx.nd.contrib.box_nms(ids_scores_boxes, overlap_thresh=0.5, valid_thresh=0.01, topk=200)

    temp = ids_scores_boxes_nms[:, :, 0].copy()
    ids_scores_boxes_nms[:, :, 0] = ids_scores_boxes_nms[:, :, 1]
    ids_scores_boxes_nms[:, :, 1] = temp  # (b, N, 6)
    scores_ids_boxes_nms = ids_scores_boxes_nms

    if is_sorted:
        sort_idx = mx.nd.argsort(scores_ids_boxes_nms[:, :, 0], axis=-1, is_ascend=False)  # (b, N)
        for i, idx in enumerate(sort_idx):   # (N, )
            scores_ids_boxes_nms[i] = scores_ids_boxes_nms[i, idx]
    return scores_ids_boxes_nms


def test_calc_loss2(mx_imgs, mx_labels, net, epoch):
    """
    要求 net 必须有 ctx, get_anchors()
    """
    anchors = net.get_anchors()  # (N, 4)
    anchors = mx.nd.array(anchors).expand_dims(axis=0).as_in_context(net.ctx)  # (1, N, 4)
    mx_imgs = mx_imgs.as_in_context(net.ctx)
    mx_labels = mx_labels.as_in_context(net.ctx)

    tensor_preds = net(mx_imgs)
    tensor_targs = generate_target2(anchors, tensor_preds, mx_labels, 
        iou_thresh=0.5, neg_thresh=0.5, negative_mining_ratio=3, dynamic_sampling=True)
    loss, loss_con, loss_cls, loss_box = calc_loss2(tensor_preds, epoch, *tensor_targs)
    print('loss sum:', loss.asscalar(), 'confidence loss:', loss_con.sum().asscalar(), 
        'class probability loss:', loss_cls.sum().asscalar(), 'box regression loss:', loss_box.sum().asscalar())
    return
