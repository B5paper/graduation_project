import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np
import gluoncv as gcv
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

root_path = r'D:\Documents\Data_Files\Datasets\Pascal\VOC2012'


# dataset
class Dataset(mx.gluon.data.Dataset):
    class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                 'train', 'tvmonitor']

    def __init__(self, root_path=None, mode='train'):
        super(Dataset, self).__init__()

        img_idx_directory = os.path.sep.join([root_path, 'VOCtrainval_11-May-2012', 'VOCdevkit', 'ImageSets', 'Main'])
        self.img_directory = os.path.sep.join([root_path, 'VOCtrainval_11-May-2012', 'VOCdevkit', 'JPEGImages'])
        self.annotation_directory = os.path.sep.join([root_path, 'VOCtrainval_11-May-2012', 'VOCdevkit', 'Annotations'])

        if mode == 'train':
            idx_file_name = 'train.txt'
        elif mode == 'val':
            idx_file_name = 'val.txt'
        else:
            raise Exception('Unknown mode.')

        self.img_indices = []
        with open(os.path.sep.join([img_idx_directory, idx_file_name]), 'r') as f:
            while True:
                line = f.readline()
                if line == '':
                    break
                self.img_indices.append(line.rstrip('\n'))
        self.len = len(self.img_indices)
        return

    def __getitem__(self, idx):
        img_path = os.path.sep.join([self.img_directory, str(self.img_indices[idx]) + '.jpg'])
        img = plt.imread(img_path)

        label_path = os.path.sep.join([self.annotation_directory, str(self.img_indices[idx]) + '.xml'])
        tree = et.parse(label_path)
        root = tree.getroot()
        obj_iter = root.iterfind('object')
        label = np.array([])
        for obj in obj_iter:
            temp_label = np.empty((5,))
            class_name = obj.find('name').text
            temp_label[0] = self._class_name_to_digit(class_name)
            bndbox = obj.find('bndbox')
            xmin = bndbox.find('xmin').text
            xmax = bndbox.find('xmax').text
            ymin = bndbox.find('ymin').text
            ymax = bndbox.find('ymax').text
            coord = [xmin, ymin, xmax, ymax]
            temp_label[1:5] = coord
            if len(label) == 0:
                label = temp_label.reshape((1, 5)).astype('int')
            else:
                label = np.concatenate((label, temp_label.reshape((1, 5)).astype('int')))

        return img, label

    def __len__(self):
        return self.len

    def _class_name_to_digit(self, class_name):
        class_names = Dataset._class_names
        if class_name in class_names:
            return class_names.index(class_name)
        else:
            raise Exception("Class name doesn't exist.")

    _class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                       'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                       'train', 'tvmonitor']

    def digit_to_class_name(self, digit):
        if digit in range(0, len(Dataset._class_names)):
            return Dataset._class_names[digit]
        else:
            raise Exception('The digit is out of range %d.' % (len(Dataset._class_names)))


# data processing
batchify_fn = gcv.data.batchify.Tuple((gcv.data.batchify.Stack(), gcv.data.batchify.Pad(pad_val=-1)))
# 仅用于 dataloader_val
batchify_fn_val = gcv.data.batchify.Tuple((gcv.data.batchify.Append(), gcv.data.batchify.Append()))
batchify_fn_val_2 = gcv.data.batchify.Tuple((gcv.data.batchify.Stack(), gcv.data.batchify.Pad(pad_val=-1)))


def resize_img_and_label(img, label, size=0):
    """
    resize the img and label, and transform the dtype of image from int8 to float32

    img: np.array, int8, (h, w, c)  
    label: absolute, np.array, (N, 5)  
    size: tuple or int, int for short side with keeping ratio, tuple format is (height, width) without keep ratio  
    return: (resz_img, resz_label), np.array, dtype of resz_img: float32, absolute coordinates  
    """
    if size == 0:
        return img / 255, label

    size = np.array(size)
    if size.size == 1:
        keep_ratio = True
    else:
        keep_ratio = False
        h, w = size

    img = img / 255
    if keep_ratio:
        resz_img = mx.img.resize_short(mx.nd.array(img), size)
        resz_img = resz_img.asnumpy()
    else:
        resz_img = mx.img.imresize(mx.nd.array(img), w, h)
        resz_img = resz_img.asnumpy()

    img_size = img.shape[:2]
    label_copy = label.copy().astype('float32')
    label_copy[:, 1:] = bbox_abs_to_rel(label_copy[:, 1:], img_size)
    label_copy[:, 1:] = bbox_rel_to_abs(label_copy[:, 1:], resz_img.shape[:2])
    resz_label = label_copy

    '''
    关于为什么要在这里变成 float32 的说明：
    因为插值必定会引入小数，所以原数据经过插值就会从int8类型变成float32类型，并且范围在0到255
    这样的一个数据放到 matplotlib 里面画图会有警告，说数据类型或者数值范围不对
    所以我要在这里把它在数值上归一化到0~1，保证出来的数据能直接用于画图
    这样就会带来一个问题，
    因为在 mxnet.gluon.data.vision.transforms.ToTensor() 里，不光变换维度，还会将 int 类型的数据转换成 float 型，
    这样就与我的操作产生了矛盾，因此用完我这个 resize 函数，就不要再用 ToTensor() 了，
    应该改成我自己写的 to_tensor() 函数（仅用于改变维度，不改变数据类型）
    '''
    return resz_img, resz_label


def data_augment(img, label, size=0, rb=0, rc=0, rh=0, rs=0, rflr=False, rftb=False, re=False, rcp=False):
    height, width = img.shape[:2]

    aug_img = img.copy()
    aug_label = label.copy()

    # random color jitter
    aug_img = mx.nd.array(aug_img)
    # random_color_jitter = mx.gluon.data.vision.transforms.RandomColorJitter(rb, rc, rs, rh)
    # aug_img = random_color_jitter(aug_img)
    # 我佛了，在 RandomColorJitter 里，rs 不能取 0。
    random_brightness = mx.gluon.data.vision.transforms.RandomBrightness(rb)
    random_contrast = mx.gluon.data.vision.transforms.RandomContrast(rc)
    random_saturation = mx.gluon.data.vision.transforms.RandomSaturation(rs)
    random_hue = mx.gluon.data.vision.transforms.RandomHue(rh)
    aug_funcs = [random_brightness, random_contrast, random_saturation, random_hue]
    import random
    random.shuffle(aug_funcs)
    for aug_func in aug_funcs:
        aug_img = aug_func(aug_img)

    # random expansion
    if re:
        if np.random.rand() > 0.5:
            aug_img, expand = gcv.data.transforms.image.random_expand(aug_img, fill=[m for m in mean])
            aug_label[:, 1:3] = aug_label[:, 1:3] + (expand[0], expand[1])
            aug_label[:, 3:5] = aug_label[:, 3:5] + (expand[0], expand[1])

    # random crop
    if rcp:
        rslt = gcv.data.transforms.experimental.bbox.random_crop_with_constraints(aug_label[:, 1:], aug_img.shape[:2][::-1])
        aug_label, crop = gcv.data.transforms.experimental.bbox.random_crop_with_constraints(aug_label[:, [1, 2, 3, 4, 0]], aug_img.shape[:2][::-1])
        aug_label = aug_label[:, [-1, 0, 1, 2, 3]]
        aug_img = mx.image.fixed_crop(aug_img, *crop)

    # random flip
    if rflr:
        if np.random.rand() > 0.5:
            aug_img = mx.nd.image.flip_left_right(aug_img)
            aug_label_copy = aug_label.copy()
            aug_label[:, 1:2] = aug_img.shape[1] - aug_label_copy[:, 3:4]
            aug_label[:, 3:4] = aug_img.shape[1] - aug_label_copy[:, 1:2]
            
    if rftb:
        if np.random.rand() > 0.5:
            aug_img = mx.nd.image.flip_top_bottom(aug_img)
            aug_label[:, 2:3] = height - label[:, 4:5]
            aug_label[:, 4:5] = height - label[:, 2:3]

    # resize
    aug_img, aug_label = resize_img_and_label(aug_img, aug_label, size)
    # aug_img = aug_img.asnumpy()

    return aug_img, aug_label


def normalize(img, mean=mean, std=std):
    norm_img = (img - np.array(mean)) / np.array(std)
    return norm_img


def denormalize(img, mean=mean, std=std):
    """
    :function: denormalize the image data.
    :param img: numpy.ndarray, (h, w, c)
    :param mean:
    :param std:
    :return: denormalized image
    """
    denorm_img = img * np.array(std) + np.array(mean)
    # if (denorm_img > 1.5).any():
    #     denorm_img[np.where(denorm_img < 0)] = 0
    #     denorm_img[np.where(denorm_img > 255)] = 255
    #     return denorm_img.astype('uint8')
    # else:
    #     denorm_img[np.where(denorm_img < 0)] = 0
    #     denorm_img[np.where(denorm_img > 1)] = 1
    #     return denorm_img
    denorm_img[np.where(denorm_img < 0)] = 0
    denorm_img[np.where(denorm_img > 1)] = 1
    return denorm_img


def bbox_abs_to_rel(bbox, pic_size):
    """
    :function: Transform absolute bbox coordinates to relative coordinates.
    :param bbox: numpy.adarray, uint8, (N, 4)
    :param pic_size: numpy.ndarray, (height, width)
    :return: relative coordiantes of bbox
    """
    height, width = pic_size
    rel_bbox = bbox.astype('float32') / np.array([width, height] * 2)
    return rel_bbox


def bbox_rel_to_abs(bbox, pic_size):
    """
    :function: Transform bbox coordinates from relative to absolute.
    :param bbox: numpy.ndarray, (N, 4)
    :param pic_size: numpy.ndarray or tuple, (height, width)
    :return: absolute coordinates of bbox
    """
    height, width = pic_size
    abs_bbox = bbox * np.array([width, height] * 2)
    return abs_bbox


def to_tensor(img):
    """
    :function: change the dimensions and the type of data (from np.array to mx.nd.array)
    :param img: np.array, (h, w, c)
    :return: mx.nd.array, (c, h, w)
    """
    return mx.nd.array(img).transpose(axes=(2, 0, 1))


def prepare_datum_for_net(img, label, transform_fn, size=0):
    """
    :func: image transform includes resizing, color normalizing, to tensor, expanding dimension
    label transform includes converting to relative coords, expanding dimension

    :param img: np.array, uint8, (h, w, c)
    :param label: np.array, uint8, (N, 5)
    :return:
    """
    img = img.astype('float32')  # deepcopy
    label = label.astype('float32')

    mx_img, mx_label = transform_fn(img, label)
    mx_img = mx_img.expand_dims(axis=0)
    mx_label = mx_label.expand_dims(axis=0)

    return mx_img, mx_label


def transform_data_to_batch(dataset, indices, model_img_size=0):  # 未完成
    """

    :param dataset:
    :param indices: list
    :param model_img_size:
    :return:
    """
    batchify_fn = gcv.data.batchify.Tuple(gcv.data.batchify.Stack(),
                                           gcv.data.batchify.Pad(pad_val=-1))
    batch_data = []
    for i in indices:
        img, label = dataset[i]

    batch_data = batchify_fn(batch_data)
    return


def transform_imgs_labels_to_train_batch(imgs, labels, transform_fn, batchify_fn):
    """

    :param imgs: list
    :param labels: list
    :param transform_fn:
    :param batchify_fn
    :return:
    """
    if len(imgs) != len(labels):
        raise Exception('the length of imgs is not equal to the length of labels.')

    mx_imgs, mx_labels = [], []
    for i in range(len(imgs)):
        img = imgs[i]
        label = labels[i]
        mx_img, mx_label = transform_fn(img, label)  # 注意，这里只是 transform，并没有 batchify
        mx_imgs.append(mx_img)
        mx_labels.append(mx_label)
    mx_imgs, mx_labels = batchify_fn(list(zip(mx_imgs, mx_labels)))
    return mx_imgs, mx_labels


def transform_imgs_labels_to_val_batch(imgs, labels, transform_fn, batchify_fn):
    """

    :param imgs:
    :param labels:
    :param transform_fn:
    :param batchify_fn:
    :return:
    """
    if len(imgs) != len(labels):
        raise Exception('the length of imgs is not equal to the length of labels.')
    mx_imgs = []
    img_sizes = []
    batch_labels = []
    for i in range(len(imgs)):
        img = imgs[i]
        label = labels[i]
        mx_img, _ = transform_fn(img, label)  # 注意，这里只是 transform，并没有 batchify
        img_sizes.append(img.shape[:2])
        mx_imgs.append(mx_img)
        batch_labels.append(label)
    mx_imgs, batch_labels = batchify_fn(list(zip(mx_imgs, batch_labels)))
    batch_labels = batch_labels.asnumpy()
    return mx_imgs, img_sizes, batch_labels


def transform_list_mx_to_list(list_mx_imgs, list_mx_labels):
    if len(list_mx_imgs) != len(list_mx_labels):
        raise Exception('the length of list_mx_imgs mismatch list_mx_labels ')
    list_imgs = []
    list_labels = []
    for i in range(len(list_mx_imgs)):
        mx_img = list_mx_imgs[i]
        mx_label = list_mx_labels[i]
        img = mx_img[0].asnumpy()
        label = mx_label[0].asnumpy()
        list_imgs.append(img)
        list_labels.append(label)
    return list_imgs, list_labels


# data visualization
def _add_rectangle(axes, relative_bbox, color='red'):
    """
    :param axes:
    :param relative_bbox: relative_bbox: numpy.ndarray, (x_left_top, y_left_top, x_right_bottom, y_right_bottom)
    :param color: color: 'red', 'blue', '...'
    :return: None
    """

    img_lst = axes.get_images()
    img = img_lst[0]
    pic_size = img.get_size()
    h, w = pic_size
    dx1, dy1 = relative_bbox[0].tolist(), relative_bbox[1].tolist()
    dx2, dy2 = relative_bbox[2].tolist(), relative_bbox[3].tolist()
    x1, y1 = dx1 * w, dy1 * h
    x2, y2 = dx2 * w, dy2 * h
    axes.add_patch(
        plt.Rectangle(xy=(x1, y1), width=x2 - x1, height=y2 - y1, fill=False, edgecolor=color, linewidth=2))
    return


def visualize_boxes(boxes, color, fig=None, is_rltv_cor=False, img_size=None):
    """
    img_size: tuple or array, (2, )
    boxes: nplarray, shape:(n, 4), 4:(xmin, ymin, xmax, ymax)
    is_rltv_cor: if the boxes use relative coordinates
    return: fig
    """
    if not isinstance(boxes, np.ndarray):
        raise Exception('parameter boxes is not an numpy array.')
        
    if not fig:
        fig = plt.figure()
    axes = fig.gca()

    if is_rltv_cor:
        if img_size is None:
            raise Exception('parameter img_size must be provided if if_rltv_cor is True.')
        boxes = bbox_rel_to_abs(boxes, img_size)
        axes.set_ylim(0, img_size[0])
        axes.set_xlim(0, img_size[1])
        axes.set_aspect('equal')
    
    for box in boxes:
        axes.add_patch(
            plt.Rectangle(xy=(box[0], box[1]), width=box[2]-box[0], height=box[3]-box[1], fill=False, edgecolor=color, linewidth=2))
    return fig


def data_visualize(img, bboxes):
    """
    :param img: numpy.ndarray, (h, w, c)
    :param bboxes: absolute position, numpy.ndarray, (N, 4), (x_left_top, y_left_top, x_right_bottom, y_right_bottom)
    :return: the figure that pyplot uses.
    """
    fig = plt.figure()
    plt.imshow(img)
    axes = fig.axes[0]

    for bbox in bboxes:
        rel_bbox = bbox_abs_to_rel(bbox, img.shape[:2])
        _add_rectangle(axes, rel_bbox)

    return fig


# validation
def validate(img, label, net, the_first_n_bboxes=3, std=(0.229, 0.224, 0.225),
             mean=(0.485, 0.456, 0.406)):
    """
    Inputs:

    img: numpy.array, shape (h, w, c)  without normalization
    label: np.array, shape (N, 5)
        n represents thd index of label boxes in one image.
    cls_preds: NDArray, shape (1, N, 2)
        N represents number of anchors. 2 represents 1 type of object
    anchors: NDArray, shape (1, N, 4)
        4 represents [rx1, ry1, rx2, ry2]. These are relative coordinates.
    bbox_preds: NDArray, shape (1, N*4)
        reshape(N, 4), equivalent to the shape of anchors

    Outputs:
        image with predicted bounding boxes.
    """
    height, width = img.shape[:2]
    import matplotlib.pyplot as plt
    fig = plt.imshow(img)
    axes = fig.axes

    mx_img, _ = resize_img_and_label(img, label)
    mx_img = mx.img.color_normalize(mx_img, mean=mean, std=std)
    mx_img = to_tensor(mx_img)
    mx_img = mx_img.expand_dims(axis=0)
    mx_img = mx_img.as_in_context(net.ctx)

    anchors, cls_preds, bbox_preds = net(mx_img)

    cls_probs = cls_preds.softmax().transpose(axes=(0, 2, 1))
    detect = mx.contrib.nd.MultiBoxDetection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(detect[0]) if row[0].asscalar() != -1]
    detection_output = detect[0, idx]
    print(detection_output)

    for relative_bbox in detection_output[0:the_first_n_bboxes, 2:6]:
        _add_rectangle(axes=axes, relative_bbox=relative_bbox.asnumpy(), color='blue')

    relative_label = bbox_abs_to_rel(label[0, 1:], np.array([height, width]))
    _add_rectangle(axes=axes, relative_bbox=relative_label, color='red')
    plt.show()


def validate_data_n (n, dataset, net, the_first_n_bboxes=3):
    """
    reset data_iter, and validate the n-th data. The index of data starts from 1.
    :param n:
    :param the_first_N_bboxes:
    :return:
    """

    img, label = dataset[n]
    validate(img, label, net, the_first_n_bboxes)


def visualize_pred(img, label=None, scores_class_boxes_nms=None, class_names_list=None, fig=None, show_label=False):
    """
    visualize the prediction of detector.

    :param img: np.array, (h, w, c)
    :param label: np.array, (N, 5), 5: (cls_id, xmin, ymin, xmax, ymax), absolute
    :param scores_class_boxes_nms: np.array, (n, 6), 6: (score, cls_id, xmin, ymin, xmax, ymax), absolute
    :param class_names_list: list, illustration of class id.
    :param fig: if the fig parameter is provided, the result will be plotted on the fig
    :param show_label: whether to show the ground-truth boxes.
    :return: fig, the figure that plot on.

    example:
    myutils.visualize_pred(img, label, scores_cls_boxes, dataset.class_names, show_label=True)
    plt.show()
    """
    scores_class_boxes_nms[:, -4:] = scores_class_boxes_nms[:, -4:] * 300
    # class_scores_boxes 假设这里已经 nms 过了
    if isinstance(scores_class_boxes_nms, mx.nd.NDArray):
        scores_class_boxes_nms = scores_class_boxes_nms.asnumpy()
    if fig is None:
        fig = plt.figure()

    axes = fig.gca()  # 如果 fig 不存在 axes，gca() 会自动创建一个。
    axes.imshow(img)
    scores_class_boxes = scores_class_boxes_nms[scores_class_boxes_nms[:, 0] != -1]

    for box in scores_class_boxes:
        # plot boundary box
        axes.add_patch(
            plt.Rectangle(xy=(box[2], box[3]), width=box[4] - box[2], height=box[5] - box[3],
                          fill=False, edgecolor='blue', linewidth=2))

        # plot prediction class label text
        font_size = img.shape[1] * 0.08
        if class_names_list is not None:
            if box[1].astype('int') in range(0, len(class_names_list)):
                cls_name = class_names_list[int(box[1])]
                axes.text(box[2], box[3], cls_name+' '+str(box[0])[:4], fontname='monospace', fontsize=font_size,
                          horizontalalignment='left', verticalalignment='bottom',
                          bbox={'boxstyle': 'square', 'pad': 0.2, 'alpha': 0.6,
                                'facecolor': plt.get_cmap('tab20').colors[int(box[1])],
                                'edgecolor': plt.get_cmap('tab20').colors[int(box[1])]})
            else:
                raise Exception('The digit is out of range %d.' % (len(class_names_list)))

    # plot ground-truth class label text
    if label is not None and show_label == True:
        for box in label:
            axes.add_patch(
                plt.Rectangle(xy=(box[1], box[2]), width=box[3] - box[1], height=box[4] - box[2],
                              fill=False, edgecolor='red', linewidth=2))
            cls_name = class_names_list[int(box[0])]
            axes.text(box[1], box[2], cls_name, fontname='monospace', fontsize=font_size,
                      horizontalalignment='left', verticalalignment='bottom',
                      bbox={'boxstyle': 'square', 'pad': 0.2, 'alpha': 0.6,
                            'facecolor': plt.get_cmap('tab20').colors[int(box[0])],
                            'edgecolor': plt.get_cmap('tab20').colors[int(box[0])]})

    axes.set_axis_off()
    return fig


def parse_batch_detection_outputs(batch_pred_scores_cls_boxes, batch_labels):
    batch_pred_bboxes = batch_pred_scores_cls_boxes[:, :, 2:]
    batch_pred_labels = batch_pred_scores_cls_boxes[:, :, 1]
    batch_pred_scores = batch_pred_scores_cls_boxes[:, :, 0]
    batch_gt_boxes = batch_labels[:, :, 1:]
    batch_gt_labels = batch_labels[:, :, 0]
    return batch_pred_bboxes, batch_pred_labels, batch_pred_scores, batch_gt_boxes, batch_gt_labels


def calc_batch_average_precision(labels, batch_scores_cls_boxes, metric):
    """
    这里要求 padding 的 labels 和 batch_scores_cls_boxes 都为 -1。
    参数 label 不需要经过 resize。可以在 dataset.transform() 里设置 transform_fn 的时候，不设置 resize。
    即创建一个专门用于 validation 的

    :param labels:
    :param batch_scores_cls_boxes:
    :param metric:
    :return:
    """
    if isinstance(labels, mx.nd.NDArray):
        labels = labels.asnumpy()
    if isinstance(batch_scores_cls_boxes, mx.nd.NDArray):
        batch_scores_cls_boxes = batch_scores_cls_boxes.asnumpy()

    batch_pred_boxes = batch_scores_cls_boxes[:, :, 2:].astype('int32')  # (b, N, 4)
    batch_pred_labels = batch_scores_cls_boxes[::, :, 1].astype('int32')  # (b, N)
    batch_pred_scores = batch_scores_cls_boxes[:, :, 0]  # (b, N)
    batch_gt_boxes = labels[:, :, 1:]  # (b, M, 4)
    batch_gt_labels = labels[:, :, 0]  # (b, M)

    metric.update(batch_pred_boxes, batch_pred_labels, batch_pred_scores, batch_gt_boxes, batch_gt_labels)
    return metric.get()


metric = gcv.utils.metrics.voc_detection.VOCMApMetric(class_names=Dataset.class_names)


def plot_result(save_fig=False):
    import pickle
    with open('./aps_dynamic_rank_matching', 'rb') as f:
        ap_rank = pickle.load(f)
    with open('./serial_arch_aps', 'rb') as f:
        ap_serial = pickle.load(f)
    with open('./result_no_dynamic_1', 'rb') as f:
        ap_original = pickle.load(f)[2]
        
    print('original aps:', ap_original)
    print('rank matching aps:', ap_rank)
    print('serial arch aps:', ap_serial)

    fig = plt.figure()
    plt.plot(ap_rank, 'r')
    plt.plot(ap_original, 'b')
    plt.plot(ap_serial, 'g')
    plt.show()

    if save_fig:
        fig.set_tight_layout(True)
        fig.savefig('dynamic_rank_matching.png')
