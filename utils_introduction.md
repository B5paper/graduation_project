# Utils Introduction

## myutils

**Numpy**

|Function|Description|
|-|-|
|resize_img_and_label|对图像和标签进行缩放，缩放后将图像类型转换成 float。<br/>这个函数支持按比例缩放短边，也支持指定宽度和高度缩放|
|data_augment|对图像增广，标签也跟着改变。图像仍是(h, w, c)格式，<br/>不会被归一化，但会被转换成 float。包含了缩放操作。|
|normalize|对 float 格式的图像进行归一化，使用内置的 mean 和 std。<br/>注意输入的数据是 float 格式的。|
|denormalize|反归一化，其它参数同 normalize。<br/>注意反归一化后的数据仍是 float 格式的。|
|bbox_abs_to_rel|将边界框的绝对坐标改成相对坐标|
|bbox_rel_to_abs|将相对坐标改成绝对坐标|

**MXNet**

|Function|Description|
|-|-|
|to_tensor|转换维度，(h, w, c) -> (c, h, w)<br/>改变数据格式，numpy -> mxnet|
|prepare_datum_for_net|主要对图片进行缩放，改变维度，将标签的绝对坐标改成相对的。<br/>改变数据格式，numpy -> mxnet。<br/>输出的图像数据可以直接送入网络中作 test（注意不是 train）<br/>train 的时候用对应模型 utils 中的 transform_fn。|

**Visualization**

|Function|Description|
|-|-|
|data_visualize|画出图像和边界框|

**Evaluation**

|Function|Description|
|-|-|
|validate||

## yolo_utils

**Numpy**

|Function|Description|
|-|-|
|get_center_coord_of_bboxes|得到输入框的中心的坐标|
|get_center_grid_of_bboxes|得到输入框的中心所在 grid|
|translate_box_yolo_to_abs|将 box 的坐标从 yolo 格式转换成绝对坐标|
|translate_box_abs_to_yolo|与上述函数的逆过程|
|_generage_random_pred_tensor|用于测试 generate_target|
|generate_target|用于生成计算 loss 的 target|

**MXNet**

|Function|Description|
|-|-|
|transform_fn|用于制作直接用于训练的增广数据集 mx_dataset|
|calc_yolo_loss|给定 tensor_pred 和 tensor_targ，计算 loss<br/>这里需要注意，tensor_pred 和 tensor_targ 都必须为 mx.nd.array 格式<br/>另外，因为输出中可能有负数，在计算 loss 时并没有取根号|
|generate_target_batch|适用于 bactch 模式|
|calc_batch_loss|适用于 batch 模式|

**Visualization**

|Function|Description|
|-|-|
|visualize_grids|在给定的图像上画出 grids，并标出 object 的中心点|
|visualize_pred|可视化 ground-truth 的框对应的预测框|

一个训练单张图片的例子：

```{.python .input}
import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np
import myutils
import yolo_utils

# 构建数据集
root_path = r'D:\Documents\Data_Files\Datasets\Pascal\VOC2012'
dataset = myutils.Dataset(root_path)
img, label = dataset[0]
yolo_utils.visualize_grids(img, label)
plt.show()

# 构建 YOLO 模型，初始化模型参数
ctx = mx.cpu()
img_size = (300, 300)
yolo_v1 = yolo_utils.YOLO_v1(ctx=ctx)
yolo_v1.hybridize()
output = yolo_v1(mx.nd.random.normal(shape=(2, 3, img_size[0], img_size[1]), ctx=ctx))
print(output.shape)

# 构建训练器。需要注意的是，如果 learning rate 大于 0.0001，会梯度爆炸
trainer = mx.gluon.Trainer(yolo_v1.collect_params(), 'sgd')
trainer.set_learning_rate(0.0001)

# 训练模型
img, label = dataset[0]
mx_img, mx_label = myutils.prepare_datum_for_net(img, label, img_size)
for i in range(10):
    with mx.autograd.record():
        tensor_preds = yolo_v1(mx_img.as_in_context(ctx))
        tensor_pred = tensor_preds[0]
        tensor_targ = yolo_utils.generate_target(img_size, mx_label[0].asnumpy(), tensor_pred.asnumpy())
        tensor_targ = mx.nd.array(tensor_targ, ctx=ctx)
        batch_loss = yolo_utils.calc_yolo_loss(tensor_pred, tensor_targ)
    batch_loss.backward()
    trainer.step(1)

    # 展示每一次的训练结果
    yolo_utils.visualize_pred(img, label, tensor_pred.asnumpy())
    plt.show()
```

一个训练 batch 的例子：

```python
import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np
import myutils
import yolo_utils

# 构建数据集
root_path = r'D:\Documents\Data_Files\Datasets\Pascal\VOC2012'
dataset = myutils.Dataset(root_path)
img, label = dataset[0]
yolo_utils.visualize_grids(img, label)
plt.show()

# 构建 YOLO 模型，初始化模型参数
ctx = mx.cpu()
img_size = (300, 300)
yolo_v1 = yolo_utils.YOLO_v1(ctx=ctx)
yolo_v1.hybridize()
output = yolo_v1(mx.nd.random.normal(shape=(2, 3, img_size[0], img_size[1]), ctx=ctx))
print(output.shape)

# 构建训练器。需要注意的是，如果 learning rate 大于 0.0001，会梯度爆炸
trainer = mx.gluon.Trainer(yolo_v1.collect_params(), 'sgd')
trainer.set_learning_rate(0.0001)

# 构建 batch
mx_imgs = mx.nd.array([])
mx_labels = mx.nd.array([])
for i in range(2):
    img, label = dataset[i]
    mx_img, mx_label = myutils.prepare_datum_for_net(img, label, img_size)
    if mx_imgs.size == 0:
        mx_imgs = mx_img
        mx_labels = mx_label
    else:
        mx_imgs = mx.nd.concat(mx_imgs, mx_img, dim=0)
        mx_labels = mx.nd.concat(mx_labels, mx_label, dim=0)
print(mx_imgs.shape)

# 训练，同时展示第一张图片的训练结果
img, label = dataset[0]

for i in range(20):
    with mx.autograd.record():
        tensor_preds = yolo_v1(mx_imgs.as_in_context(ctx))
        tensor_targs = yolo_utils.generate_target_batch(img_size, mx_labels, tensor_preds)
        batch_loss = yolo_utils.calc_batch_loss(tensor_preds, tensor_targs)
    batch_loss.backward()
    trainer.step(2)

    # visualize
    yolo_utils.visualize_pred(img, label, tensor_preds[0].asnumpy())
    plt.show()
```
