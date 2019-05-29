import mxnet as mx
import gluoncv as gcv
import numpy as np
import myutils
import ssd_utils
import matplotlib.pyplot as plt

def unreval_anchors(data, feature_map_shapes, num_anchor_per_position=6, ith_anchor=0):
    """
    假设一个位置有 a 个 anchor，这个函数取各个 feature map 上 [0, a) 个 anchors 的其中一个，并作为 list 返回数据  
    data: mx.nd.array, (N, ...)  注意这里没有 batch  
    feature_map_shapes: list, (M, 2)  
    return: list of arrays
    """
    data_on_feature_maps = []
    pos = 0
    for feature_map_shape in feature_map_shapes:
        feature_map_shape = mx.nd.array(feature_map_shape[-2:])
        start = int(pos + ith_anchor)
        end = int(pos + (mx.nd.prod(feature_map_shape) * num_anchor_per_position).asscalar())
        step = int(num_anchor_per_position)
        temp_data = data[start:end:step]
        pos = end
        
        if temp_data.ndim == 1:
            temp_data = temp_data.reshape(feature_map_shape.asnumpy().astype('int').tolist())  
        elif temp_data.ndim == 2:
            temp_data = temp_data.reshape((*tuple(feature_map_shape.asnumpy().astype('int').tolist()), -1))
        temp_data = temp_data.asnumpy()
        data_on_feature_maps.append(temp_data)

    return data_on_feature_maps


def test_generate_target(mx_imgs, mx_labels, net):
    """
    对传入的数据一张一张地计算 target，并统计结果
    net: attr: ctx, get_anchors(), get_map_shapes_list()
    """
    mx_imgs = mx_imgs.as_in_context(net.ctx)  # (b, c, h, w)
    mx_labels = mx_labels.as_in_context(net.ctx)  # (b, M, 5)
    anchors = net.get_anchors()  # (N, 4)
    anchors = mx.nd.array(anchors).expand_dims(axis=0).as_in_context(net.ctx)  # (1, N, 4)

    tensor_preds = net(mx_imgs)
    tensor_preds = tensor_preds.reshape((0, -1, 25))
    
    tensor_targs = ssd_utils.generate_target(anchors, tensor_preds, mx_labels[:, :, -4:], mx_labels[:, :, 0:1], 
        iou_thresh=0.5, neg_thresh=0.5, negative_mining_ratio=3, dynamic_sampling=True)
    cls_targs, box_targs, pos_neg_samples = tensor_targs

    for i, mx_img in enumerate(mx_imgs):
        print('img '+str(i)+':', 'positives:', str(mx.nd.sum(pos_neg_samples[i] > 0).asscalar())+',', 
            'negatives:', mx.nd.sum(pos_neg_samples[i] < 0).asscalar())

        # visualize original image and label boxes
        fig = plt.figure()
        plt.imshow(myutils.denormalize(mx_img.transpose((1, 2, 0)).asnumpy()))
        myutils.visualize_boxes(mx_labels[i][:, -4:].asnumpy(), 'blue', fig, is_rltv_cor=True, img_size=mx_imgs.shape[-2:])
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()        
        plt.show()

        # visualize samples
        # positives
        print('positive samples on each feature map:')
        for anchor_idx in range(net.anchor_num_per_position):
            print('anchor index:', anchor_idx)
            pos_on_fmaps = unreval_anchors(pos_neg_samples[i] > 0, net.get_feature_map_shapes_list(), 
                net.anchor_num_per_position, ith_anchor=anchor_idx)

            fig = plt.figure()
            axes = fig.subplots(nrows=1, ncols=len(net.get_feature_map_shapes_list()))
            for j, fmap_shape in enumerate(net.get_feature_map_shapes_list()):
                axes[j].imshow(pos_on_fmaps[j])
                axes[j].set_title('*'.join([str(elm) for elm in fmap_shape[-2:]]) + ', num: ' + 
                    str(np.sum(pos_on_fmaps[j] > 0)))
            fig.set_figwidth(16)
            fig.set_figheight(fig.get_figwidth())
            plt.show()

        # negatives
        print('negative samples on each feature map:')


def test_generate_target2(mx_imgs, mx_labels, net, check_opt=None):
    """
    对传入的数据一张一张地计算 target，并统计结果
    net: attr: ctx, get_anchors(), get_map_shapes_list()
    """
    mx_imgs = mx_imgs.as_in_context(net.ctx)  # (b, c, h, w)
    mx_labels = mx_labels.as_in_context(net.ctx)  # (b, M, 5)
    anchors = net.get_anchors()  # (N, 4)
    anchors = mx.nd.array(anchors).expand_dims(axis=0).as_in_context(net.ctx)  # (1, N, 4)

    tensor_preds = net(mx_imgs)
    tensor_targs = ssd_utils.generate_target2(anchors, tensor_preds, mx_labels, 
        iou_thresh=0.5, neg_thresh=0.3, negative_mining_ratio=3, dynamic_sampling=True)
    con_targs, cls_targs, box_targs, pos_neg_samples = tensor_targs

    for i, mx_img in enumerate(mx_imgs):
        print('img '+str(i)+':', 'positives:', str(mx.nd.sum(pos_neg_samples[i] > 0).asscalar())+',', 
            'negatives:', mx.nd.sum(pos_neg_samples[i] < 0).asscalar())

        # visualize original image and label boxes
        fig = plt.figure()
        plt.imshow(myutils.denormalize(mx_img.transpose((1, 2, 0)).asnumpy()))
        myutils.visualize_boxes(mx_labels[i][:, -4:].asnumpy(), 'blue', fig, is_rltv_cor=True, img_size=mx_imgs.shape[-2:])
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()        
        plt.show()
        
        if 'samples' in check_opt:
            # visualize samples
            # positives
            print('positive samples on each feature map:')
            for anchor_idx in range(net.anchor_num_per_position):
                print('anchor index:', anchor_idx)
                pos_on_fmaps = unreval_anchors(pos_neg_samples[i] > 0, net.get_feature_map_shapes_list(), 
                    net.anchor_num_per_position, ith_anchor=anchor_idx)

                fig = plt.figure()
                axes = fig.subplots(nrows=1, ncols=len(net.get_feature_map_shapes_list()))
                for j, fmap_shape in enumerate(net.get_feature_map_shapes_list()):
                    axes[j].imshow(pos_on_fmaps[j], cmap=plt.get_cmap('tab20'), vmin=0, vmax=1)
                    axes[j].set_title('*'.join([str(elm) for elm in fmap_shape[-2:]]) + ', num: ' + 
                        str(np.sum(pos_on_fmaps[j])))
                fig.set_figwidth(16)
                fig.set_figheight(fig.get_figwidth())
                plt.show()
        
            # negatives
            print('negative samples on each feature map:')
            for anchor_idx in range(net.anchor_num_per_position):
                print('anchor index:', anchor_idx)
                neg_on_fmaps = unreval_anchors(pos_neg_samples[i] < 0, net.get_feature_map_shapes_list(), 
                    net.anchor_num_per_position, ith_anchor=anchor_idx)

                fig = plt.figure()
                axes = fig.subplots(nrows=1, ncols=len(net.get_feature_map_shapes_list()))
                for j, fmap_shape in enumerate(net.get_feature_map_shapes_list()):
                    axes[j].imshow(neg_on_fmaps[j])
                    axes[j].set_title('*'.join([str(elm) for elm in fmap_shape[-2:]]) + ', num: ' + 
                        str(np.sum(neg_on_fmaps[j])))
                fig.set_figwidth(16)
                fig.set_figheight(fig.get_figwidth())
                plt.show()

        if 'ohem' in check_opt:
            # check out if thre prediction of negatives is 1.
            pred_con = tensor_preds[i, :, :2].argmax(axis=-1)  # (N, )
            pred_con = mx.nd.where(con_targs[i] == 0, pred_con, -mx.nd.ones_like(pred_con))  # (N, )
            correct_ratio = mx.nd.sum(pred_con == 0) / mx.nd.sum(pred_con >= 0)
            print('correct prediction in negatives:', correct_ratio.asscalar())

        if 'confidence' in check_opt:
            # visualize object confidence
            print('object confidence on each feature map:')
            for anchor_idx in range(net.anchor_num_per_position):
                print('anchor index:', anchor_idx)
                con_on_fmaps = unreval_anchors(con_targs[i], net.get_feature_map_shapes_list(), 
                    net.anchor_num_per_position, ith_anchor=anchor_idx)

                fig = plt.figure()
                axes = fig.subplots(nrows=1, ncols=len(net.get_feature_map_shapes_list()))
                for j, fmap_shape in enumerate(net.get_feature_map_shapes_list()):
                    axes[j].imshow(con_on_fmaps[j], cmap=plt.get_cmap('tab20'), vmin=-1, vmax=1)
                    axes[j].set_title('*'.join([str(elm) for elm in fmap_shape[-2:]]) + ', num: ' + 
                        str(np.sum(con_on_fmaps[j] > 0)))
                fig.set_figwidth(16)
                fig.set_figheight(fig.get_figwidth())
                plt.show()

        if 'class' in check_opt:
            # visualize class targets
            print('class targets on each feature map:')
            for anchor_idx in range(net.anchor_num_per_position):
                print('anchor index:', anchor_idx)
                cls_on_fmaps = unreval_anchors(cls_targs[i], net.get_feature_map_shapes_list(),
                    net.anchor_num_per_position, ith_anchor=anchor_idx)

                fig = plt.figure()
                axes = fig.subplots(nrows=1, ncols=len(net.get_feature_map_shapes_list()))
                for j, fmap_shape in enumerate(net.get_feature_map_shapes_list()):
                    axes[j].imshow(cls_on_fmaps[j], cmap=plt.get_cmap('terrain'), vmin=-1, vmax=20)
                    axes[j].set_title('*'.join([str(elm) for elm in fmap_shape[-2:]]) + ', num: ' + 
                        str(np.sum(cls_on_fmaps[j] >= 0)))
                fig.set_figwidth(16)
                fig.set_figheight(fig.get_figwidth())
                plt.show()

        if 'box' in check_opt:
            # visualize box targets
            print('box targets on each feature map:')
            # vmin = box_targs.min().asscalar()
            # vmax = box_targs.max().asscalar()
            for anchor_idx in range(net.anchor_num_per_position):
                print('anchor index:', anchor_idx)
                box_on_fmaps = unreval_anchors(box_targs[i], net.get_feature_map_shapes_list(),
                    net.anchor_num_per_position, ith_anchor=anchor_idx)

                fig = plt.figure()
                axes = fig.subplots(nrows=1, ncols=len(net.get_feature_map_shapes_list()))
                for j, fmap_shape in enumerate(net.get_feature_map_shapes_list()):
                    axes[j].imshow(box_on_fmaps[j][:, :, 0] != 0, cmap=plt.get_cmap('terrain'), vmin=0, vmax=1)
                    axes[j].set_title('*'.join([str(elm) for elm in fmap_shape[-2:]]) + ', num: ' + 
                        str(np.sum(box_on_fmaps[j][:, :, 0] != 0)))
                fig.set_figwidth(16)
                fig.set_figheight(fig.get_figwidth())
                plt.show()
    return


def test_calc_loss2(mx_imgs, mx_labels, net, epoch):
    """
    要求 net 必须有 ctx, get_anchors()
    """
    anchors = net.get_anchors()  # (N, 4)
    anchors = mx.nd.array(anchors).expand_dims(axis=0).as_in_context(net.ctx)  # (1, N, 4)
    mx_imgs = mx_imgs.as_in_context(net.ctx)
    mx_labels = mx_labels.as_in_context(net.ctx)

    tensor_preds = net(mx_imgs)
    tensor_targs = ssd_utils.generate_target2(anchors, tensor_preds, mx_labels, 
        iou_thresh=0.5, neg_thresh=0.5, negative_mining_ratio=3, dynamic_sampling=True)
    loss, loss_con, loss_cls, loss_box = ssd_utils.calc_loss2(tensor_preds, epoch, *tensor_targs)
    print('loss sum:', loss.asscalar(), 'confidence loss:', loss_con.sum().asscalar(), 
        'class probability loss:', loss_cls.sum().asscalar(), 'box regression loss:', loss_box.sum().asscalar())
    return


def test_preds(mx_imgs, mx_labels, net, tensor_targs, mode='mean'):
    """
    mode = {'mean' | 'single' | 'all'}
    """

    mx_imgs = mx_imgs.as_in_context(net.ctx)
    mx_labels = mx_labels.as_in_context(net.ctx)  # (b, M, 5)
    con_targs, cls_targs, box_targs, pos_neg_sample = tensor_targs

    anchors = mx.nd.array(net.get_anchors(), ctx=net.ctx).expand_dims(axis=0)  # (1, N, 4)
    tensor_preds = net(mx_imgs)  # (b, N, 26)
    con_preds = tensor_preds[:, :, :2]  # (b, N, 2)
    cls_preds = tensor_preds[:, :, 2:22]  # (b, N, 20)
    box_preds = tensor_preds[:, :, -4:]  # (b, N, 4)
    _box_decoder = gcv.nn.coder.NormalizedBoxCenterDecoder(convert_anchor=True)
    box_preds = _box_decoder(box_preds, anchors)  # (b, N, 4)
    gt_boxes = mx_labels[:, :, 1:]  # (b, M, 4)

    ious = mx.nd.transpose(mx.nd.contrib.box_iou(anchors[0], gt_boxes), (1, 0, 2))  # (b, N, M)
    neg_thresh = 0.5
    con_all_neg_targs = mx.nd.where(mx.nd.prod(ious < neg_thresh, axis=-1) * (pos_neg_sample <= 0), 
                            mx.nd.ones_like(pos_neg_sample), 
                            mx.nd.zeros_like(pos_neg_sample))  # (b, N)，+1: neg, 0: pos

    batch_con_pos_pre = []
    batch_con_pos_rec = []
    batch_con_neg_pre = []
    batch_con_neg_rec = []
    batch_hard_neg_rec = []
    batch_cls_rec = []
    batch_box_coarse_pre = []
    for i, mx_img in enumerate(mx_imgs):
        # confidence positives
        con_preds_rslt = mx.nd.argmax(con_preds[i], axis=-1)  # (N, )
        con_pos_preds = con_preds_rslt > 0  # (N, )
        con_neg_preds = con_preds_rslt == 0  # (N, )
        con_pos_targs = con_targs[i] > 0  # (N, )
        con_hard_neg_targs = con_targs == 0  # (N, )

        # pos precision
        con_pos_pre = (mx.nd.sum(con_pos_preds * con_pos_targs) / mx.nd.sum(con_pos_preds)).asscalar()
        # pos recall
        con_pos_rec = (mx.nd.sum(con_pos_preds * con_pos_targs) / mx.nd.sum(con_pos_targs)).asscalar()

        # confidence negatives
        # neg precision
        con_neg_pre = (mx.nd.sum(con_neg_preds * con_all_neg_targs[i]) / mx.nd.sum(con_neg_preds)).asscalar()
        # neg recall
        con_neg_rec = (mx.nd.sum(con_neg_preds * con_all_neg_targs[i]) / mx.nd.sum(con_all_neg_targs[i])).asscalar()

        # hard negatives recall
        con_hard_neg_rec = (mx.nd.sum(con_neg_preds * con_hard_neg_targs) / mx.nd.sum(con_hard_neg_targs)).asscalar()

        # class precision
        cls_preds_rslt = mx.nd.argmax(cls_preds[i], axis=-1)  # (N, )
        cls_rec = (mx.nd.sum(cls_preds_rslt == cls_targs[i]) / mx.nd.sum(cls_targs[i] >= 0)).asscalar()

        # regression precision
        gt_box = gt_boxes[i]  # (M, 4)
        box_pred = mx.nd.where(con_preds_rslt.expand_dims(axis=-1).broadcast_like(box_preds[i]), 
                    box_preds[i], 
                    mx.nd.zeros_like(box_preds[i]))  # (N, 4)
        pred_box_iou = mx.nd.contrib.box_iou(box_pred, gt_box)  # (N, M)
        box_coarse_pre = mx.nd.sum(pred_box_iou > 0.5) / mx.nd.sum(pred_box_iou > 0)  # 假设预测了 iou 肯定不等于 0
        box_coarse_pre = box_coarse_pre.asscalar()

        if mode == 'single' or mode == 'all':
            print(('img %d:, pos pre: %.4f, pos rec: %.4f, all neg pre: %.4f, all neg rec: %.4f, '
                  'hard neg rec: %.4f, cls pre: %.4f, box pre: %.4f') % (i, con_pos_pre, con_pos_rec, 
                  con_neg_pre, con_neg_rec, con_hard_neg_rec, cls_rec, box_coarse_pre))
        
        batch_con_pos_pre.append(con_pos_pre)
        batch_con_pos_rec.append(con_pos_rec)
        batch_con_neg_pre.append(con_neg_pre)
        batch_con_neg_rec.append(con_neg_rec)
        batch_hard_neg_rec.append(con_hard_neg_rec)
        batch_cls_rec.append(cls_rec)
        batch_box_coarse_pre.append(box_coarse_pre)

    if mode == 'mean' or mode == 'all':
        print(('mean: pos pre: %.4f, pos rec: %.4f, neg pre: %.4f, neg rec: %.4f, hard neg rec: %.4f, cls rec: %.4f ' + 
            'box pre: %.4f') % 
             (np.array(batch_con_pos_pre).mean(), np.array(batch_con_pos_rec).mean(),
              np.array(batch_con_neg_pre).mean(), np.array(batch_con_neg_rec).mean(),
              np.array(batch_hard_neg_rec).mean(), np.array(batch_cls_rec).mean(), 
              np.array(batch_box_coarse_pre).mean()))
        
    return

def test_box_coder():
    box_encoder = gcv.nn.coder.NormalizedBoxCenterEncoder()
    box_decoder = gcv.nn.coder.NormalizedBoxCenterDecoder(convert_anchor=True)

    boxes = ssd_utils.generate_random_gt_boxes(1, 3)  # (1, 3, 4)
    anchors = ssd_utils.generate_random_gt_boxes(1, 5)  # (1, 5, 4)
    matches = mx.nd.array(np.random.randint(0, 3, size=(1, 5)))  # (1, 5)
    samples = mx.nd.array(np.random.randint(0, 2, size=(1, 5)))  # (1, 5)
    boxes_enc, _ = box_encoder(samples, matches, anchors, boxes)
    boxes_dec = box_decoder(boxes_enc, anchors)

    print('boxes:------------------')
    print(boxes)
    print('------------------------')

    print('anchors:------------------')
    print(anchors)
    print('------------------------')

    print('matches:------------------')
    print(matches)
    print('------------------------')

    print('samples:------------------')
    print(samples)
    print('------------------------')

    print('boxes_enc:------------------')
    print(boxes_enc)
    print('------------------------')

    print('boxes_dec:------------------')
    print(boxes_dec)
    print('------------------------')

    return


def test_dataloader(dataloader, opt='first_batch'):
    if opt == 'first_batch':
        mx_imgs, mx_labels = next(iter(dataloader))  
        for i in range(len(mx_imgs)):
            plt.imshow(myutils.denormalize(mx_imgs[i].transpose((1, 2, 0)).asnumpy()))
            myutils.visualize_boxes(mx_labels[i, :, 1:].asnumpy(), 'blue', plt.gcf(), is_rltv_cor=True, img_size=mx_imgs[i].shape[-2:])
            if not plt.gca().yaxis_inverted():
                plt.gca().invert_yaxis()
            plt.show()
    if opt == 'all_batch':
        for mx_imgs, mx_labels in dataloader:
            for i in range(len(mx_imgs)):
                plt.imshow(myutils.denormalize(mx_imgs[i].transpose((1, 2, 0)).asnumpy()))
                myutils.visualize_boxes(mx_labels[i, :, 1:].asnumpy(), 'blue', plt.gcf(), is_rltv_cor=True, img_size=mx_imgs[i].shape[-2:])
                if not plt.gca().yaxis_inverted():
                    plt.gca().invert_yaxis()
                plt.show()

    return


def test_ssd_loss(mx_imgs, mx_labels, net, anchors, generate_targ_fn):
    """
    默认 mx_imgs，mx_labels，net，anchors 已经放在同一个计算环境中。
    """
    tensor_preds = net(mx_imgs)  # (b, N, 25)
    cls_preds = tensor_preds[:, :, :21]  # (b, N, 21)
    box_preds = tensor_preds[:, :, -4:]  # (b, N, 4)
    gt_boxes = mx_labels[:, :, -4:]  # (b, N, 4)
    gt_ids = mx_labels[:, :, 0:1]  # (b, N, 1)

    tensor_targs = generate_targ_fn(anchors, cls_preds, gt_boxes, gt_ids,
        iou_thresh=0.5, neg_thresh=0.3, negative_mining_ratio=3, dynamic_sampling=True)
    loss_sum, loss_cls, loss_box = ssd_utils.ssd_loss(cls_preds, box_preds, *tensor_targs)

    print('loss_sum:', loss_sum.sum().asscalar(), 'loss_cls:', loss_cls.sum().asscalar(), 
        'loss_box:', loss_box.sum().asscalar())
    return


def test_get_pred_scores_classes_and_boxes(mx_imgs, mx_labels, net, anchors):
    """
    anchors: (1, N, 4)
    """
    tensor_preds = net(mx_imgs)  # (b, N, 25)
    
    for i in range(len(mx_imgs)):
        scr_cls_box = ssd_utils.get_pred_scores_classes_and_boxes(tensor_preds[i].expand_dims(axis=0), None, anchors)
        plt.imshow(myutils.denormalize(mx_imgs[i].transpose((1, 2, 0)).asnumpy()))
        myutils.visualize_boxes(scr_cls_box[:, -4:], 'blue', plt.gcf(), 
            is_rltv_cor=True, img_size=[300, 300])
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.show()
        print(scr_cls_box.shape)
        print(scr_cls_box[:10])
        print('label:', mx_labels[i].asnumpy()[mx_labels[i].asnumpy()[:, 0] >= 0])
    return

    
