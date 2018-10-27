# Mask RCNN 我的理解

## 流程

![](/Users/liuxingyu/Pictures/markdown/mask-rcnn0.png)

1. covs extract image features
2. RPN generates region proposal
3. for each proposal(RoI), ROI aligh generates a fixed-length feature map(7 * 7 * 256)
4. compute the loss



## 1. convs 基础网络































































## 思路

mask rcnn 的思路很简洁：Faster Rcnn 针对每个候选区域有两个输出的：种类标签和bbox得偏移量。那么MRCNN就是在FasterRCNN的基础上增加一个分支进而再增加一个输出，即object mask



##### Fast R-CNN通过RoIPool层对每个候选区域提取特征，从而实现目标分类和bbox回归

![](/Users/liuxingyu/Pictures/markdown/frcnnn.png)

mrcnn 和 faster rcnn 产生相同的RPN layer， 后面发生了变化：

![](/Users/liuxingyu/Pictures/markdown/maskrcn11.png) 



## Primary task

RPN是为了找出多个重叠的anchors，尽可能的覆盖到所有区域。



采用的是多任务损失函数，针对每个ROI(不是所有）定义为

$L = L_{cls} + L_{box} + L_{mask}$

前两者就是faster_rcnn的输出

最后一个：

> 1. mask branch predict K 个种类的m*m的二值得掩膜，K为分类物体的种类数目
> 2. 依据预测的分支（faster RCNN部分)预测结果，当前Roi的物体种类
> 3. 第i个二值掩膜输出就是Roi的损失 $L_{mask}$