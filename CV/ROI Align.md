# ROI Align

## Motivation

ROI Align 是在Mask-RCNN这篇论文里提出的一种==区域特征聚集方式==, 很好地解决了ROI Pooling操作中两次量化造成的区域不匹配(mis-alignment)的问题。

## **ROI Pooling 的局限性分析**

在常见的两级检测框架（比如Fast-RCNN，Faster-RCNN，RFCN）中，ROI Pooling 的作用是根据预选框的位置坐标在特征图中将相应区域池化为**固定尺寸的特征图**，以便于==后续的分类和包围框回归操作==。由于预选框的位置通常由模型回归得到的，一般来说是浮点数，而池化后的特征图要求尺寸固定，故ROI Pooling这一操作存在两次量化过程

- 将候选框边界量化为正数点坐标值。(量化就是取整)
- 将量化后的边界区域平均分割成k*k个单元（bin)对每一个单元进行边界量化。

事实上，经过上述两次量化，此时的候选框已经和最开始回归出来的位置有一定的偏差，这个偏差会影响检测或者分割的准确度。在论文里，作者把它总结为“不匹配问题（misalignment）。

> 下面我们用直观的例子具体分析一下上述区域不匹配问题。如 **图1** 所示，这是一个Faster-RCNN检测框架。输入一张800 * 800的图片，图片上有一个665 * 665的包围框(框着一只狗)。图片经过主干网络提取特征后，特征图缩放步长（stride）为32。因此，图像和包围框的边长都是输入时的1/32。800正好可以被32整除变为25。==但665除以32以后得到20.78==，带有小数，于是ROI Pooling **直接将它量化成20**。接下来需要把框内的特征池化7 * 7的大小，因此将上述包围框平均分割成7 * 7个矩形区域。显然，==每个矩形区域的边长为2.86==，又含有小数。于是ROI Pooling **再次把它量化到2**。经过这两次量化，候选区域已经出现了较明显的偏差（如图中绿色部分所示）。更重要的是，该层特征图上0.1个像素的偏差，缩放到原图就是3.2个像素。**那么0.8的偏差，在原图上就是接近30个像素点的差别**，这一差别不容小觑。
>
> 步骤总结：
>
> > 候选框---->（32 trides cons)得到一个新的边长----->pooling 成7*7区域（256d)---->FCs

![](/Users/liuxingyu/Pictures/markdown/ROIPool.png)



### problem

- 为什么ROI Pooling要量化成整数， 不能直接取浮点数吗？ 因为单位是1
- 为什么要在单元里面计算固定的四个为位置在做池化，不能直接选出中心位置吗？每个位置使用（x,y)这样表示吗？
- 候选区域分割成K*K个bin 这个候选区域就一层吗？ 256d是depth 一体的 不可分割



## **ROI Align 的主要思想和具体方法**

ROI Align的思路很简单：取消量化操作，使用双线性内插的方法获得坐标为浮点数的像素点上的图像数值,从而将整个特征聚集过程转化为一个连续的操作。值得注意的是，在具体的算法操作上，ROI Align并不是简单地补充出候选区域边界上的坐标点，然后将这些坐标点进行池化，而是重新设计了一套比较优雅的流程，如图所示：

- 遍历每个候选区域，保持浮点数边界不做量化（根据上层生成的候选区域）
- 将候选区域分割成k * k个单元，每个单元的边界也不做量化
- 在每个**单元**中计算的固定的四个坐标位置，用双线性内插的方法计算出这四个位置的值，然后进行最大池化操作。

![](/Users/liuxingyu/Pictures/markdown/roi align.png)

## **ROI Align 的反向传播**

常规的ROI Pooling的反向传播公式如下：

![图片标题](https://leanote.com/api/file/getImage?fileId=59fbd202ab644135b00006fa)

这里，xi代表池化前特征图上的像素点；$y_{rj}$代表池化后的第r个候选区域的第j个点；i*(r,j)代表点$y_{rj}$像素值的来源（最大池化的时候选出的最大像素值所在点的坐标）。由上式可以看出，只有当池化后某一个点的像素值在池化过程中采用了当前点Xi的像素值（即满足i=i*(r，j)），才在xi处回传梯度。

类比于ROIPooling，ROIAlign的反向传播需要作出稍许修改：首先，在ROIAlign中，xi*（r,j）是**一个浮点数的坐标位置(**前向传播时计算出来的采样点)，在池化前的特征图中，每一个与 xi*(r,j) 横纵坐标均小于1的点都应该接受与此对应的点yrj回传的梯度，故ROI Align 的反向传播公式如下: 
　　 
![图片标题](https://leanote.com/api/file/getImage?fileId=59fbe350ab644137db000a4e)

上式中，d(.)表示两点之间的距离，Δh和Δw表示 xi 与 xi*(r,j) 横纵坐标的差值，这里作为双线性内插的系数乘在原始的梯度上。

## **实现步骤**

1. 划分7*7的bin(我们可以直接精确的映射到feature map来划分bin，不用第一次量化) 

   ![](/Users/liuxingyu/Pictures/markdown/roi1.png)

2. 每个bin中采样4个点，双线性插值 

![](/Users/liuxingyu/Pictures/markdown/roi2.png)

3. 对每个bin4个点做max或average pool

   ```python
   # pytorch
   # 先采样到14*14 再Max pooling到7*7
   pre_pool_size = cfg.POOLING_SIZE * 2
   grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, pre_pool_size, pre_pool_size)))
   crops = F.grid_sample(bottom.expand(rois.size(0), bottom.size(1), bottom.size(2), bottom.size(3)), grid, mode=mode)
   crops = F.max_pool2d(crops, 2, 2)
   # tensorflow
   pooled.append(tf.image.crop_and_resize(
   tf.image.crop_and_resize(
   	feature_maps[i], level_boxes, boxes_indices, self.pool_shape, method='bilinear'
   	)
   ))
   ```

   ## sigmoid代替softmax

   利用分类的结果，在mask之路，只取对应类别的channel然后做sigmoid，减少类间竞争，避免出现一些洞之类





## 代码作用

正如 RPN 一样，它为每个 ROI 生成了两个输出：

1. 类别：ROI 中的目标的类别。和 RPN 不同（两个类别，前景或背景），这个网络更深并且可以**将区域分类为具体的类别**（人、车、椅子等）。它还可以生成一个背景类别，然后就可以弃用 ROI 了。
2. 边框精调：和 RPN 的原理类似，它的目标是进一步精调边框的位置和尺寸以将目标封装。

- 代码提示：分类器和边框回归器已在 fpn_classifier_graph() 中创建。

![](/Users/liuxingyu/Pictures/markdown/roi3.jpg)