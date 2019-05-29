# 毕业设计代码

## 毕设过程说明

本仓库为本科毕业设计的代码，毕设的课题是《基于深度神经网络的单步目标检测算法及应用》，本来打算随便水一水，后来师兄说最好复现一个前几年的模型，然后想想办法提升点 AP。

之前曾简单复现过 Faster R-CNN, YOLO v1, 以及 SSD，感觉 SSD 的改进空间更大一些，所以就选择了 SSD 为基础模型。但 SSD 原版使用的 backbone net 是 VGG16，这个 backbone 占显存 500 多 M，而我的显存只有 2G，所以只能把 backbone 换成 ResNet-50 v2，这个只占 50 M 左右显存。训练时的图片大小只能设置为`300 * 300`，原版 SSD 还有个`512 * 512`版本的，我根本没有复现的希望。原版 SSD 训练时 batch size 设置为 32，我这里只能设置为 5。。。嘛，总之在我力所能及的范围内，简单地把 SSD 复现了一遍。

然后就是改进。改点什么好呢？复现 SSD 过程中发现样本匹配是个大问题，匹配不均匀的话就训练不明白，所以对正样本的匹配过程做了个改进。这个改进效果很明显，精度提升了很多，起码有 4% 的 mAP。后来实在想不到再改进些什么好，就把原来的单级分类器拆成了两级，刚开始时这个改进是灾难性的，精度直接下降了 20%，后来调了调阈值参数，最后效果和第一次稍稍持平，所以这个改进起码是没有坏处的，也可以写进论文里。

再后来感觉只说明这两点改进的话，论文写得比较单薄，且没结合课题里的“应用”这两个字，所以就拍了些视频，把改进后的 SSD 算法往这些视频里应用了一波，做了一番分析。当然这些分析都直接放在论文里了，代码里只有生成检测图片的部分。

无论 SSD 还是 YOLO，或者是 FPN，都属于有 anchor 的检测方法。最近无 anchor 的检测方法盛行，所以下一步我也会放弃 anchor 系列算法的研究了，可能会复现些无 anchor 的算法做做研究。

## 代码说明

刚开始的时候，没想到毕设会变得这么复杂，所以当时目录下只有三个文件：`myutils.py`，`yolo_utils.py`，`ssd_utils.py`，最多再加上一个`test.ipynb`。`myutils.py`负责数据集，图片迭代器，数据预处理，可视化，坐标转换等各个琐碎细节，`yolo_utils`及`ssd_utils`负责各个模型的实现，training target 的生成，loss 的计算等等。做实验，训练，查看训练结果之类的都在 jupyter notebook `test.ipynb`里做。

后来模型复杂了，代码变成了几千行，没法再用这几个文件简单地搞定了，所以就分了类，比如出现了专门准备数据集的`dataset_utils.py`，专门负责做 evaluation 的`evaluation.py`，专门用来实现模型的`model`文件夹，以及专门实现训练过程的`training.py`等。因为新的代码框架是慢慢实现的，所以到现在也没有完全把第一阶段的代码完全消除干净。可能随着复现模型的增多，这个目标检测框架最终会定下来吧。

这次毕设的代码实现中，最让我头疼，也让我收获最大的是矩阵运算。之前复现的模型规模小，碰到问题就写几个 for 循环就可以搞定，当时觉得也没啥。后来当规模做大后，越发觉得运算矩阵化的重要性。有一次在整个 Pascal VOC 2012 数据集上做 evaluation，发现需要时间为 12 小时，心想这也太长了吧。后来把大部分写 for 的地方都替换成了矩阵运算，evaluation 的时间变成了 3 分钟。emmmm。。。所以矩阵化真的很重要。但是矩阵化的代码也有点难写，只有有限的几种操作，还需要考虑 shape，还需要考虑特殊情况……就很难受。总之，在生成 target 和计算 loss 这些函数上我花的时间是最长的。。

其它的 jupyter notebook 文件大部分是临时做的实验的可视化结果，用来给论文里添加图片说明，或者用于代码调试之类的。

**最后给大家一句忠告：工作环境里湿度大时要少敲键盘，容易得关节炎，真的！疼疼疼疼疼……**