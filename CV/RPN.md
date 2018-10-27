# RPN(region proposal network)

## 自己的语言

目的：

> 扫描图像 寻找存在的目标

输入：

> shape=?   





RPN 是一个轻量的神经网络，它用滑动窗口来扫描图像，并寻找存在目标的区域。

RPN 扫描的区域被称为 anchor，这是在图像区域上分布的矩形在不同的尺寸和长宽比下，图像上会有将近 20 万个 anchor，并且它们互相重叠以尽可能地覆盖图像。

## Process

- select x region proposals(anchors)              ( x = k * conveluted pixel) 
- for each anchor output: 2cla 4reg feature (按照分数排序，然后选取前 N 个回归框特征)
- select fixed number1 ROI region (by NMS)
- select fixed number2 TRAIN_ROIS_PER_IMAGE (by IoU). 



在卷积特征图上，用 3*3 的窗口执行卷积操作。对特征图中的每个中心点，<!--中心点就是卷积后的像素点--> 选取 k 个不同 scale、aspect ratio 的 anchor。<!--k就是len(scale) * len(ratio)-->    

生成候选的 region proposals。特征图中的每个点会生成对应窗口区域的特征编码（。接着对该低维的特征编码做卷积操作，输出 2*k (2 bg or fg)分类特征和 4*k(4 中心点坐标 width height) 回归特征，分别对应每个点每个 anchor 属于目标的概率以及它所对应的物体的坐标信息。

> 例如特征图中每个点对应的 anchor 大小为 128*128像素，256*256像素，512*512 像素，每个 anchor 的长宽比为１:１，１:２，２:１，这样特征图中的每个点会对应 9 个 anchor，假设特征图大小为60*40，这样特征图大概会生成60*40*9个 anchor，大约 2 万个。同时特征图上的每个点会对应 60*40*9 region proposal box 的回归特征和分类特征。对 60*40*9 个分类特征，
>
> <!--每一个anchor只会对应bg or fg--> 按照分数排序，**[rpn_probs用于标记某个候选框为前景和背景的概率值，通过top-N可以筛选出概率最后的前N个候选框。]**    然后选取前 N 个回归框特征，比如前 5000 个。然后把回归框的值 (dy, dx, log(dh), log(dw)) 解码为 bounding box 的真实坐标 (y1, x1, y2, x2) 值。



接着通过非极大值一致算法 NMS 选择一定数量的 ROI region，比如说 2000 个。然后计算 ROI region 和 gt_boxes 的重叠覆盖情况，选择一个数量的 **TRAIN_ROIS_PER_IMAGE**，比如说 200 个进行训练。可以采用如下规则：

-  假如某 ROI 与任一目标区域的 IoU 最大，则该 anchor 判定为有目标。
- 假如某 ROI 与任一目标区域的 IoU>0.5，则判定为有目标；
-  假如某 ROI 与任一目标区域的 IoU<0.5，则判定为背景。

其中 **IoU，就是预测 box 和真实 box 的覆盖率**，其值等于两个 box 的交集除以两个 box 的并集。其它的 ROI 不参与训练。还可设定 ROI_POSITIVE_RATIO=0.33，比如说 33% 的 ROI 为正样本，其它为负样本。



RPN 扫描这些 anchor 的速度有多快呢？非常快。滑动窗口是由 RPN 的卷积过程实现的，可以使用 GPU 并行地扫描所有区域。此外，RPN 并不会直接扫描图像，而是扫描主干特征图。这使得 RPN 可以有效地复用提取的特征，并避免重复计算。

- 代码提示：RPN 在 rpn_graph() 中创建。anchor 的尺度和长宽比由 config.py 中的 RPN_ANCHOR_SCALES 和 RPN_ANCHOR_RATIOS 控制。

RPN 为每个 anchor 生成两个输出：

1. anchor 类别：前景或背景（FG/BG）。前景类别意味着可能存在一个目标在 anchor box 中。
2. 边框精调：前景 anchor（或称正 anchor）可能并没有完美地位于目标的中心。因此，RPN 评估了 delta 输出（x、y、宽、高的变化百分数）以精调 anchor box 来更好地拟合目标。(代码用什么表示？)



![](/Users/liuxingyu/Pictures/markdown/rpn.jpg)



使用 RPN 的预测，我们可以选出最好地包含了目标的 anchor，并对其位置和尺寸进行精调。如果有多个 anchor 互相重叠，我们将保留拥有最高前景分数的 anchor，并舍弃余下的（非极大值抑制）。然后我们就得到了最终的区域建议，并将其传递到下一个阶段。

- 代码提示：ProposalLayer 是一个自定义的 Keras 层，可以读取 RPN 的输出，选取最好的 anchor，并应用边框精调。

# ProposalLayer

该步骤用于获取最后的候选框（Proposal），形如[y1,x1,y2,x2]。<!-- center_point width height. -->

使用通过anchors机制获取的，和通过RPN网络获取的`rpn_probs[bg_prob,fg_prob]、rpn_bbox[dy,dx,log(dh),log(dw)]`作为输入数据，获取最终的候选框,但求得的候选框会超过2万+，可以通过以下三种方式将候选框的数量固定在一定范围。

1. top-N
   rpn_probs用于标记某个候选框为前景和背景的概率值，通过top-N可以筛选出概率最后的前N个候选框。
2. 去掉超出图片范围候选框
3. 非极大值抑制（NMS算法）

![这里写图片描述](http://p7n2owwza.bkt.clouddn.com/mask-lijie9.png)

如上图（左）中，虽然几个框都检测到了人脸，但是我不需要这么多的框，我需要找到一个最能表达人脸的框。通过NMS算法能够实现。

先假设有6个矩形框，根据分类器类别分类概率做排序，从小到大分别属于人脸的概率分别为A、B、C、D、E、F。
①从最大概率矩形框F开始，分别判断A~E与F的重叠度IOU是否大于某个设定的阈值;
②假设B、D与F的重叠度超过阈值，那么就扔掉B、D；并标记第一个矩形框F，是我们保留下来的
③从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，重叠度大于一定的阈值，那么就扔掉；并标记E是我们保留下来的第二个矩形框。
④就这样一直重复，找到所有被保留下来的矩形框。