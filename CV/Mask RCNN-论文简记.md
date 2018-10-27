## Mask RCNN-论文简记

## 网络结构

下图是Mask RCNN的一个网络结构示意图，其中黑色部分为原来的Faster R-CNN，红色部分为在Faster R-CNN网络上的修改，具体包括：（1）将ROI Pooling层替换成了ROIAlign；（2）添加了并列的FCN层（Mask层）。

![](/Users/liuxingyu/Pictures/markdown/mask-rcnn0.png)



下图是两种RCNN方法与Mask结合的示意图，其中灰色部分是原来的RCNN结合ResNet或FPN的网络，下面黑色部分为新添加的并联Mask层，这个图本身与上面的图也没有什么区别，旨在说明作者所提出的Mask RCNN方法的泛化适应能力-可以和多种RCNN框架结合，且表现都不错。

![](/Users/liuxingyu/Pictures/markdown/mask-rcnn1 (1).png)





### 技术要点

一、基础网络

> 原始图片在进入主干网络网络前，需要先 resize 成固定大小的图片，比如 1024*1024。 

在 ResNet50/101 的主干网络中，使用了 ResNet 中 Stage2，Stage3，Stage4，Stage5 的特征图，每个特征图对应的图片 Stride 为 [4, 8, 16, 32]，其中 stride 会影响图片大小的缩放比例。这样 [Stage2,Stage3,Stage4,Stage5] 对应的特征图大小为 [256 * 256,128 * 128,64 * 64,32 * 32]。ResNet 网络结构如下图所示**，其中 conv2_x，conv3_x，conv4_x，conv5_x 分别对应 Stage2, Stage3,Stage4,Stage5。** 



基于 [Stage2,Stage3,Stage4,Stage5] 的特征图，构建 FPN（feature pyramid networks，特征金字塔网络）结构。

> 在目标检测里面，低层的特征图信息量（depth可以代表信息量）比较少，但是特征图比较大，所以目标位置准确，所以容易识别一些小物体；高层特征图信息量比较丰富，但是目标位置比较粗略，特别是 stride 比较大（比如 32），图像中的小物体甚至会小于 stride 的大小，造成小物体的检测性能急剧下降。为了提高检测精度，Mask RCNN 中采用了如下的 FPN 的网络结构，一个自底向上的线路，一个自顶向下的线路以及对应层的链接。其中 1*1 的卷积核用于减少了 feature map 的个数；2up 为图片上采样，生成和上层 stage 相同大小的 feature map；最后相加对应元素，生成新的特征图。
>
> Mask RCNN 中自底向上的网络结构，为上述介绍的 ResNet50/101，对应的特征图为 [Stage2,Stage3,Stage4,Stage5]，自顶向下的网络结构，把上采样的结果和上层 Stage 的特征图进行元素相加操作，生成新的特征图 [P2, P3, P4, P5, P6], 如下所示：
>
> P5 对应 C5
>
> P4 对应 C4+ UpSampling2D（P5）
>
> P3 对应 C4+ UpSampling2D（P4）
>
> P2 对应 C4+ UpSampling2D（P3）
>
> P6 对应 MaxPooling2D(strides=2) (P5)
>
> 这样最后生成的 FPN 特征图集合为 [P2,P3,P4,P5,P6]，每个特征图对应的 Stride 为 [4, 8, 16, 32,64]，对应的特征图大小为 [256 * 256,128 * 128,64 * 64,32 * 32，16*16]，对应的 anchor 大小为 [32, 64, 128, 256, 512]，

**这样底层的特征图用于去检测较小的目标，高层的特征图用于去检测较大的目标。**



二、加入了ROIAlign层

ROIPool是一种针对每一个ROI的提取一个小尺度特征图（E.g. 7x7）的标准操作，它用以解决将**不同尺度的ROI提取成相同尺度的特征大小的问题**。ROIPool首先将浮点数值的ROI量化成离散颗粒的特征图，然后将量化的ROI分成几个空间的小块（Spatial Bins），最后对每个小块进行Max Pooling操作生成最后的结果。其示意图如下。

![](/Users/liuxingyu/Pictures/markdown/mask-rcnn-roipool (1).png)

通过计算[x/16]在连续坐标x上进行量化，其中16是特征图的步长，[ . ]表示四舍五入。这些量化引入了ROI与提取到的特征的不对准问题。由于分类问题对平移问题比较鲁棒，所以影响比较小。但是这在**预测像素级精度的掩模**时会产生一个非常的大的负面影响。

由此，作者提出ROIAlign层来解决这个问题，并且将提取到的特征与输入对齐。方法很简单，**避免对ROI的边界或者块（Bins）做任何量化**，例如**直接使用x/16代替[x/16]**。作者使用双线性插值（Bilinear Interpolation）在每个ROI块中4个采样位置上计算输入特征的精确值，并将结果聚合（使用Max或者Average）。示意图如下。

![](/Users/liuxingyu/Pictures/markdown/mask-rcnn-roialign (1).png)



三、改进了分割Loss

由原来的基于单像素Softmax的多项式交叉熵变为了基于单像素Sigmod二值交叉熵。该框架对每个类别独立地预测一个二值掩模，**没有引入类间竞争**，每个二值掩模的类别依靠网络ROI分类分支给出的分类预测结果。这与FCNs不同，FCNs是对每个像素进行多类别分类，它同时进行分类和分割，基于实验结果表明这样对于对象实例分割会得到一个较差的性能。

下面介绍一下更多的细节，在训练阶段，作者对于每个采样的ROI定义一个多任务损失函数**L=Lcls+Lbox+Lmask**，前两项不过多介绍。掩模分支对于每个ROI会有一个Km2Km2维度的输出，它编码了KK个分辨率为m×mm×m的二值掩模，分别对应着KK个类别。因此作者利用了A Per-pixel Sigmoid，并且定义为**平均二值交叉熵损失**（The Average Binary Cross-entropy Loss）。对于一个属于第KK个类别的ROI，仅仅考虑第KK个Mask（其他的掩模输入不会贡献到损失函数中）。这样的定义会允许对每个类别都会生成掩模，并且不会存在类间竞争。

四、掩模表示

一个掩模编码了一个输入对象的空间布局。作者使用了一个FCN来对每个ROI预测一个的掩模，这保留了空间结构信息。





## 损失函数

```
# Losses
rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
    [input_rpn_match, rpn_class_logits])
rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
    [input_rpn_bbox, input_rpn_match, rpn_bbox])
class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
    [target_class_ids, mrcnn_class_logits, active_class_ids])
bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
    [target_bbox, target_class_ids, mrcnn_bbox])
mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
    [target_mask, target_class_ids, mrcnn_mask])
```

`rpn_class_loss`：RPN网络分类损失函数
`rpn_bbox_loss`：==RPN网络回归损失函数.  [box_centers_y, box_centers_x, height, width] anchor==
`class_loss`：分类损失函数
`bbox_loss`：回归损失函数.
`mask_loss`：Mask回归损失函数



## 附录

### COCO数据集（2014）

![这里写图片描述](http://p7n2owwza.bkt.clouddn.com/mask-lijie13.png)

#### 6.1 JSON标注文件详解

images数组元素的数量等同于划入训练集（或者测试集）的图片的数量；
annotations数组元素的数量等同于训练集（或者测试集）中bounding box的数量；
categories数组元素的数量为92（2014年）；

```python
{
    "info": info,
    "licenses": [license],
    "images": [image], # 一个图片包含若干个annotations
"annotations": [annotation],
"categories": [categories]
}
```

— > Info

```python
"info": {
		"description": "This is stable 1.0 version of the 2014 MS COCO dataset.",
		"url": "http://mscoco.org",
		"version": "1.0",
		"year": 2014,
		"contributor": "Microsoft COCO group",
		"date_created": "2015-01-27 09:11:52.357475"
	}
```

— >Images是包含多个image实例的数组，对于一个image类型的实例：

```python
{
				"license": 5,
				"file_name": "COCO_train2014_000000057870.jpg",
				"coco_url": "http://mscoco.org/images/57870",
				"height": 480,
				"width": 640,
				"date_captured": "2013-11-14 16:28:13",
		    	"flickr_url":
                     "http://farm4.staticflickr.com/3153/2970773875_164f0c0b83_z.jpg",
				"id": 57870
			}
```

— > licenses是包含多个license实例的数组，对于一个license类型的实例：

```python
{
	"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
	"id": 1,
	"name": "Attribution-NonCommercial-ShareAlike License"
}
```

— > annotations字段是包含多个annotation实例的一个数组，annotation类型本身又包含了一系列的字段，如这个目标的category id和segmentation mask。**segmentation格式取决于这个实例是一个单个的对象（即iscrowd=0，将使用polygons格式）还是一组对象（即iscrowd=1，将使用RLE格式）**。如下所示：

```python
annotation{
    "id": int,    
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
"iscrowd": 0 or 1,
}
```

注意，只要是iscrowd=0那么segmentation就是polygon格式；只要iscrowd=1那么segmentation就是RLE格式。另外，每个对象（不管是iscrowd=0还是iscrowd=1）都会**有一个矩形框bbox** ，==矩形框左上角的坐标和矩形框的长宽会以数组的形式提供==。

==area是area of encoded masks，是标注区域的面积==。如果是矩形框，那就是高乘宽；如果是polygon或者RLE，那就复杂点。

annotation结构中的categories字段存储的是当前对象所属的category的id，以及所属的supercategory的name

```python
{
	"segmentation": [[510.66,423.01,511.72,420.03,510.45......]],
	"area": 702.1057499999998,
	"iscrowd": 0,
	"image_id": 289343,
	"bbox": [473.07,395.93,38.65,28.67],
	"category_id": 18,
	"id": 1768
}
```

**polygon格式比较简单，这些数按照相邻的顺序两两组成一个点的x y坐标**，如果有n个数（**必定是偶数**），那么就是n/2个点坐标。
如果iscrowd=1，那么segmentation就是**RLE格式(segmentation字段会含有counts和size数组)**，在json文件中gemfield挑出一个这样的例子，如下所示：

```python
{
    'counts': [272, 2, 4, 4, 4, 4, 2, 9, 1, 2, 16, 43, 143, 24......], 
    'size': [240, 320]}
```

**上面的segmentation中的counts数组和size数组共同组成了这幅图片中的分割 mask。其中size是这幅图片的宽高**，然后在这幅图像中，**每一个像素点要么在被分割（标注）的目标区域中，要么在背景中。很明显这是一个bool量**：如果该像素在目标区域中为true那么在背景中就是False；如果该像素在目标区域中为1那么在背景中就是0。对于一个240x320的图片来说，一共有76800个像素点，根据每一个像素点在不在目标区域中，我们就有了76800个bit，比如像这样（随便写的例子，和上文的数组没关系）：00000111100111110…；**但是这样写很明显浪费空间，我们直接写上0或者1的个数不就行了嘛（Run-length encoding)，于是就成了54251…，这就是上文中的counts数组。**

— > categories是一个包含多个category实例的数组，而category结构体描述如下：

```python
{
		"supercategory": "person",
		"id": 1,
		"name": "person"
	}
```

