import random
import pickle
import myutils
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et
import cv2
import numpy as np
import mxnet as mx
import gluoncv as gcv

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


class MiniSampler(mx.gluon.data.sampler.Sampler):
    def __init__(self, dataset_len, mini_size, **kwargs):
        super(MiniSampler, self).__init__(**kwargs)
        self._len = mini_size
        
        idx = list(range(dataset_len))
        random.shuffle(idx)
        self._idx = idx[:self._len]
        self._pos = 0
        return
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._pos == self._len:
            self._pos = 0
            raise StopIteration
        rtvl = self._idx[self._pos]
        self._pos += 1
        return rtvl
    
    def __len__(self):
        return self._len

def generate_random_sampler():
    root_path = r'D:\Documents\Data_Files\Datasets\Pascal\VOC2012'
    
    # prepare dataset
    dataset = myutils.Dataset(root_path=root_path, mode='train')
    dataset_val = myutils.Dataset(root_path, mode='val')

    # sampler for training
    sampler_mini = MiniSampler(len(dataset), 300)
    # sampler for validation
    sampler_mini_val = MiniSampler(len(dataset_val), 100)

    file_name = 'dataloder_sampler'
    with open(file_name, 'wb') as f:
        pickle.dump([sampler_mini, sampler_mini_val], f)


class VideoFrameDataset(mx.gluon.data.Dataset):
    def __init__(self, root_path, img_size, **kwargs):
        super(VideoFrameDataset, self).__init__(**kwargs)

        self.img_size = img_size
        self._batchify_val_fn = gcv.data.batchify.Stack()

        self._img_paths = []
        for root, dirs, files in os.walk(os.path.abspath(root_path)):
            for fname in files:
                self._img_paths.append(os.path.sep.join([root, fname]))
        self._len = len(self._img_paths)
        return

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        img = plt.imread(self._img_paths[idx])
        return img

    def transform_val_fn(*args, **kwargs):
        self, img = args  # 这里的 label 为 None
        img = img.astype('float32') / 255
        img = cv2.resize(img, tuple(self.img_size))
        mx_img = myutils.normalize(img)  # (h, w, c)
        mx_img = np.transpose(mx_img, axes=(2, 0, 1))  # (c, h, w)
        mx_img = mx.nd.array(mx_img)  # (c, h, w)
        return mx_img

    def get_batchify_val_fn(self):
        return self._batchify_val_fn
            

if __name__ == '__main__':
    root_path = r'd:/Documents/Data_Files/Datasets/pedestrian_detection_dataset/skating/input/'
    img_size = (300, 300)
    vfd = VideoFrameDataset(root_path, img_size)
    batch_size = 5
    dataloader = mx.gluon.data.DataLoader(vfd.transform(vfd.transform_val_fn), batch_size, shuffle=False, last_batch='keep')
    mx_imgs = next(iter(dataloader))
    print(mx_imgs.shape)

