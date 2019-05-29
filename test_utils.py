import mxnet as mx
import numpy as np
import cv2
import dataset_utils
from models import SSD_rank_matching
import myutils
import matplotlib.pyplot as plt
import os

def test_and_save():
    # load model
    backbone_root_path = 'd:/Documents/Data_Files/Parameters/'
    ctx = mx.gpu()
    net = SSD_rank_matching.SSD(backbone_root_path=backbone_root_path, ctx=ctx)
    model_param_path = 'd:/Documents/Git/pascal_voc_benchmark/training_result/rank_matching_net_param'
    net.load_parameters(model_param_path)

    # prepare test dataset
    root_path = './input_images/'
    dataset = dataset_utils.VideoFrameDataset(root_path, net.model_img_size)
    dataloader = mx.gluon.data.DataLoader(dataset.transform(dataset.transform_val_fn), batch_size=5, shuffle=False, 
        last_batch='keep')
    print(next(iter(dataloader)).shape)

    frame_idx = 0
    fig = plt.figure()
    for idx, mx_imgs in enumerate(dataloader):
        mx_imgs = mx_imgs.as_in_context(net.ctx)
        tensor_preds = net(mx_imgs)
        scr_cls_boxes = SSD_rank_matching.get_pred_scores_classes_and_boxes(tensor_preds, 
            mx.nd.array(net.get_anchors(), ctx=net.ctx).expand_dims(axis=0))
        for i, scr_cls_box in enumerate(scr_cls_boxes):
            fig.clf()
            img = myutils.denormalize(mx_imgs[i].transpose((1, 2, 0)).asnumpy())
            plt.imshow(img)
            myutils.visualize_boxes(scr_cls_box[:, -4:], color='blue', fig=fig, is_rltv_cor=True, img_size=(300, 300))
            myutils.visualize_pred(img, None, scr_cls_box, class_names_list=dataset_utils.Dataset.class_names, fig=plt.gcf(), show_label=True)
            if not fig.gca().yaxis_inverted():
                fig.gca().invert_yaxis()
            fig.set_figwidth(10)
            fig.set_figheight(fig.get_figwidth())
            fig.set_tight_layout(True)
            fig.savefig(os.path.abspath(r'D:\Documents\Git\pascal_voc_benchmark\test_output'+'/'+str(frame_idx)+'.png'))
            frame_idx += 1
    return

if __name__ == '__main__':
    test_and_save()