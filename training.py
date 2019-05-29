import mxnet as mx
import numpy as np
import gluoncv as gcv
import myutils
import dataset_utils
import evaluation
import ssd_utils
import ssd_utils_test
import models
import time
import pickle
from dataset_utils import MiniSampler
from models import SSD_origin
from models import SSD_bin_multi
from models import SSD_rank_matching
import matplotlib.pyplot as plt

def prepare_model(model_name=None, ctx=mx.cpu()):
    if not model_name:
        raise Exception('please input model name.')
    
    if model_name == 'ssd_origin':
        backbone_root_path = r'd:/Documents/Data_Files/Parameters/'
        ssd = models.SSD_origin.SSD(backbone_root_path=backbone_root_path, ctx=ctx)
        ssd.hybridize()
        # initialize parameters
        tensor_pred = ssd(mx.nd.zeros((2, 3, 300, 300), ctx=ctx))
        return ssd

    if model_name == 'ssd_rank_matching':
        backbone_root_path = r'd:/Documents/Data_Files/Parameters/'
        ssd = models.SSD_rank_matching.SSD(backbone_root_path=backbone_root_path, ctx=ctx)
        ssd.hybridize()
        # initialize parameters
        tensor_pred = ssd(mx.nd.zeros((2, 3, 300, 300), ctx=ctx))
        return ssd

    if model_name == 'ssd_bin_multi':
        backbone_root_path = r'd:/Documents/Data_Files/Parameters/'
        ssd = models.SSD_bin_multi.SSDBinMulti(backbone_root_path=backbone_root_path, ctx=ctx)
        ssd.hybridize()
        # initialize parameters
        tensor_pred = ssd(mx.nd.zeros((2, 3, 300, 300), ctx=ctx))
        return ssd

    raise Exception('the model %s is not found in package models' % model_name)

def prepare_dataset(root_path=None):
    if not root_path:
        root_path = r'D:\Documents\Data_Files\Datasets\Pascal\VOC2012'
    
    # prepare dataset
    dataset = myutils.Dataset(root_path=root_path, mode='train')
    dataset_val = myutils.Dataset(root_path, mode='val')
    return dataset, dataset_val

def prepare_dataloader(dataset, dataset_val, batch_size_train, batch_size_val, transform_fn, transform_fn_val, batchify_fn_train, batchify_fn_val):
    # dataloader for training
    dataloader = mx.gluon.data.DataLoader(dataset.transform(transform_fn), 
                                        batch_size=batch_size_train, shuffle=True, last_batch='rollover', 
                                        batchify_fn=batchify_fn_train)

    # dataloader for validation
    dataloader_val = mx.gluon.data.DataLoader(dataset_val.transform(transform_fn_val), 
                                            batch_size=batch_size_val, shuffle=False, last_batch='keep',
                                            batchify_fn=batchify_fn_val)

    # # create mini dataset
    # file_name = r'./dataloder_sampler'
    # with open(file_name, 'rb') as f:
    #     sampler_mini, sampler_mini_val = pickle.load(f)
    
    # dataloader_mini = mx.gluon.data.DataLoader(dataset.transform(transform_fn), 
    #                                     batch_size=batch_size_train, sampler=sampler_mini, last_batch='rollover', 
    #                                     batchify_fn=myutils.batchify_fn)
    # dataloader = dataloader_mini

    # # sampler_mini_val = dataset_utils.MiniSampler(len(dataset_val), 100)
    # dataloader_val_mini = mx.gluon.data.DataLoader(dataset_val.transform(transform_fn_val), 
    #                                         batch_size=batch_size_val, sampler=sampler_mini_val, last_batch='keep',
    #                                         batchify_fn=myutils.batchify_fn_val_2)
    # dataloader_val = dataloader_val_mini

    dataloader.class_name = dataset.class_names
    dataloader.class_name_val = dataset.class_names

    return dataloader, dataloader_val
    

def train_ssd_origin(net, *args, **kwargs):

    """
    kwargs:
    class_names
    anchors
    init_lr
    epoch
    """
    class_names = kwargs['class_names']
    anchors = kwargs['anchors']
    init_lr = kwargs['init_lr']
    train_epoch = kwargs['epoch']

    # define trainer
    trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'wd': 0.0005, 'momentum':0.9})
    # train_epoch = 20
    # init_lr = 0.05
    trainer.set_learning_rate(init_lr)
    lr_scheduler = mx.lr_scheduler.PolyScheduler(train_epoch, init_lr, pwr=0.9)
    # metric = SSD_bin_multi.SSDMetric2(class_names, anchors)
    metric = evaluation.PascalVocMetric(class_names, anchors, SSD_origin.get_pred_scores_classes_and_boxes_for_matric)

    epoch_losses = []
    aps = []
    for epoch in range(train_epoch):
        tic = time.time()
        epoch_loss = 0
        for i, (mx_imgs, mx_labels) in enumerate(dataloader):
            mx_imgs = mx_imgs.as_in_context(net.ctx)
            mx_labels = mx_labels.as_in_context(net.ctx)
            with mx.autograd.record():
                tensor_preds = net(mx_imgs)  # (b, N, 25)
                cls_preds = tensor_preds[:, :, :21]  # (b, N, 21)
                box_preds = tensor_preds[:, :, -4:]  # (b, N, 4)
                tensor_targs = ssd_utils.generate_target(mx.nd.array(net.get_anchors(), ctx=net.ctx).expand_dims(axis=0), 
                                                        cls_preds, mx_labels[:, :, -4:], mx_labels[:, :, 0], 
                                                        iou_thresh=0.5, neg_thresh=0.3, negative_mining_ratio=3, 
                                                        dynamic_sampling=True)
                sum_loss, _, _ = SSD_origin.ssd_loss(cls_preds, box_preds, *tensor_targs)
            sum_loss.backward()
            trainer.step(1)
            trainer.set_learning_rate(lr_scheduler(epoch))
            epoch_loss += sum(sum_loss).mean().asscalar()
            
            # if i % 10 == 0:
            #     ssd_utils_test.test_ssd_loss(mx_imgs, mx_labels, net, 
            #                                 mx.nd.array(net.get_anchors(), ctx=net.ctx).expand_dims(axis=0), 
            #                                 ssd_utils.generate_target)
        aps.append(metric.calc_ap(net, dataloader_val))
        epoch_losses.append(epoch_loss)
        toc = time.time()
        print('epoch %d, loss: %.4f, ap: %.4f, time: %.2f seconds' % (epoch, epoch_loss, aps[-1], toc-tic))
    return epoch_losses, aps 

def train_ssd_bin_multi(net, *args, **kwargs):

    """
    kwargs:
    class_names
    anchors
    init_lr
    epoch
    lr_schedule
    """
    class_names = kwargs['class_names']
    anchors = kwargs['anchors']
    init_lr = kwargs['init_lr']
    train_epoch = kwargs['epoch']

    # define trainer
    trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'wd': 0.0005, 'momentum':0.9})
    # train_epoch = 20
    # init_lr = 0.05
    trainer.set_learning_rate(init_lr)
    lr_scheduler = mx.lr_scheduler.PolyScheduler(train_epoch, init_lr)
    metric = SSD_bin_multi.SSDMetric2(class_names, anchors)

    epoch_losses = []
    aps = []
    for epoch in range(train_epoch):
        tic = time.time()
        epoch_loss = 0
        for i, (mx_imgs, mx_labels) in enumerate(dataloader):
            mx_imgs = mx_imgs.as_in_context(net.ctx)
            mx_labels = mx_labels.as_in_context(net.ctx)
            with mx.autograd.record():
                tensor_preds = net(mx_imgs)  # (b, N, 26)
                tensor_targs = SSD_bin_multi.generate_target2(mx.nd.array(net.get_anchors(), ctx=net.ctx).expand_dims(axis=0), 
                                                        tensor_preds, mx_labels,
                                                        iou_thresh=0.5, neg_thresh=0.3, negative_mining_ratio=3, 
                                                        dynamic_sampling=True)
                sum_loss, _, _, _ = SSD_bin_multi.calc_loss2(tensor_preds, epoch, *tensor_targs)
            sum_loss.backward()
            trainer.step(1)
            trainer.set_learning_rate(lr_scheduler(epoch))
            epoch_loss += sum(sum_loss).mean().asscalar()
            
            # if i % 10 == 0:
            #     SSD_bin_multi.test_calc_loss2(mx_imgs, mx_labels, net, epoch)
        aps.append(metric.calc_ap(net, dataloader_val))
        epoch_losses.append(epoch_loss)
        toc = time.time()
        print('epoch %d, loss: %.4f, ap: %.4f, time: %.2f seconds' % (epoch, epoch_loss, aps[-1], toc-tic))
    return epoch_losses, aps


def train_ssd_rank_matching(net, *args, **kwargs):
    """
    kwargs:
    class_names
    anchors
    init_lr
    epoch
    """
    class_names = kwargs['class_names']
    anchors = kwargs['anchors']
    init_lr = kwargs['init_lr']
    train_epoch = kwargs['epoch']

    # define trainer
    trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'wd': 0.0005, 'momentum':0.9})
    # train_epoch = 20
    # init_lr = 0.05
    trainer.set_learning_rate(init_lr)
    lr_scheduler = mx.lr_scheduler.PolyScheduler(train_epoch, init_lr, pwr=0.9)
    metric = evaluation.PascalVocMetric(class_names, anchors, 
        SSD_rank_matching.get_pred_scores_classes_and_boxes_for_matric)

    epoch_losses = []
    aps = []
    for epoch in range(train_epoch):
        tic = time.time()
        epoch_loss = 0
        for i, (mx_imgs, mx_labels) in enumerate(dataloader):
            mx_imgs = mx_imgs.as_in_context(net.ctx)
            mx_labels = mx_labels.as_in_context(net.ctx)
            with mx.autograd.record():
                tensor_preds = net(mx_imgs)  # (b, N, 25)
                cls_preds = tensor_preds[:, :, :21]  # (b, N, 21)
                box_preds = tensor_preds[:, :, -4:]  # (b, N, 4)
                tensor_targs = SSD_rank_matching.generate_target(mx.nd.array(net.get_anchors(), ctx=net.ctx).expand_dims(axis=0), 
                                                        cls_preds, mx_labels[:, :, -4:], mx_labels[:, :, 0], 
                                                        iou_thresh=0.5, neg_thresh=0.3, negative_mining_ratio=3)
                sum_loss, _, _ = SSD_rank_matching.ssd_loss(cls_preds, box_preds, *tensor_targs)
            sum_loss.backward()
            trainer.step(1)
            trainer.set_learning_rate(lr_scheduler(epoch))
            epoch_loss += sum(sum_loss).mean().asscalar()
            
            # if i % 10 == 0:
            #     ssd_utils_test.test_ssd_loss(mx_imgs, mx_labels, net, 
            #                                 mx.nd.array(net.get_anchors(), ctx=net.ctx).expand_dims(axis=0), 
            #                                 ssd_utils.generate_target)
        aps.append(metric.calc_ap(net, dataloader_val))
        epoch_losses.append(epoch_loss)
        toc = time.time()
        print('epoch %d, loss: %.4f, ap: %.4f, time: %.2f seconds' % (epoch, epoch_loss, aps[-1], toc-tic))

    return epoch_losses, aps


def train(net, dataset_train, dataset_val, dataloader_train, dataloader_val, param_model, param_training, *args, **kwargs):
    return

if __name__ == '__main__':
    ctx = mx.gpu()
    net = prepare_model(model_name='ssd_origin', ctx=ctx)
    dataset, dataset_val = prepare_dataset()
    dataloader, dataloader_val = prepare_dataloader(dataset, dataset_val, batch_size_train=5, batch_size_val=5, 
        transform_fn=net.get_transform_fn(), transform_fn_val=net.get_transform_fn_val(), 
        batchify_fn_train=myutils.batchify_fn, batchify_fn_val=myutils.batchify_fn_val_2)

    # lr_params = {'init_lr':0.01, 'epoch':20, 'lr_schedule':'poly'}
    rtvl = train_ssd_origin(net, init_lr=0.01, epoch=35, lr_schedule='poly', class_names=dataset.class_names, anchors=net.get_anchors())
    
    # visualize training procedure
    losses, aps = rtvl
    fig = plt.figure()
    plt.plot(losses)
    fig = plt.figure()
    plt.plot(aps)
    
    with open('origin_loss_and_ap', 'wb') as f:
        pickle.dump([losses, aps], f)

    net.save_parameters('origin_net_param')

    plt.show()


