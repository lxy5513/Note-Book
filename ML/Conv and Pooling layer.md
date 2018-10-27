# Conv and Pooling layer

## 1. Convolution

### math prepare

#### definition

> 我们称 $(f*g)(n)$ 为$f,  g$的卷积 
>
> 连续的定义为：$(f*g)(n) = \int_{-\infty} ^{+\infty}f(t)g(n-t)dt$
>
> 离散的定义为：$(f*g)(n) = \sum_{-\infty} ^{+\infty}f(t)g(n-t)dt$

直观理解就是，当结果由两个函数决定，且函数的自变量有关联时，卷积的意义就体现出来了。

#### application in image

> 图像上有很多噪点，这些噪点属于高频信号。高频信号，就好像平地耸立的山峰。平滑这座山峰的办法之一就是，把山峰刨掉一些土，填到山峰周围去。用数学的话来说，就是把山峰周围的高度平均一下
>
> 卷积可以帮助实现这个平滑算法。
>
> 图片的pixel位置就是上面的n  f是图片里截取的一个方阵 g是用来平滑f的（值比较小）



### CNN(用于特征提取)

> **卷积核放在神经网络里，就代表对应的权重（weight)**

假设现在 Convolution Kernel 大小是 ![3\times 3](https://www.zhihu.com/equation?tex=3%5Ctimes+3) ，我们就可以化简上式为

![](/Users/liuxingyu/Pictures/markdown/conv1.png)

  看公式不太容易明白，我们画个图看看，假如 Convolution Kernel 如下图

![](/Users/liuxingyu/Pictures/markdown/conv2.png)

 那么，从 Input Image 到 Output Image 的变化如下

![](/Users/liuxingyu/Pictures/markdown/conv3.png)

> 可以看出，其实二维卷积一样也是加权叠加/积分。需要注意的是，其中 Convolution Kernel 进行了水平和竖直方向的翻转。
>
> Convolution Kernel 具有的一个属性就是局部性。即它只关注局部特征，局部的程度取决于 Convolution Kernel 的大小。



特征图的大小（卷积特征）由下面三个参数控制，我们需要在卷积前确定它们：

- 深度（Depth）：深度对应的是卷积操作所需的滤波器个数。在下图的网络中，我们使用三个不同的滤波器对原始图像进行卷积操作，这样就可以生成三个不同的特征图。你可以把这三个特征图看作是堆叠的 2d 矩阵，那么，特征图的“深度”就是三。

![深度](https://wx4.sinaimg.cn/mw690/0065SY2ely1fhyia10lsxj30ma0b6mzm.jpg)

<!--有几个滤波器 那么生成的feature map就有几层-->

- 步长（Stride）：步长是我们在输入矩阵上滑动滤波矩阵的像素数。当步长为 1 时，我们每次移动滤波器一个像素的位置。当步长为 2 时，我们每次移动滤波器会跳过 2 个像素。步长越大，将会得到更小的特征图。
- 零填充（Zero-padding）：有时，在输入矩阵的边缘使用零值进行填充，这样我们就可以对输入图像矩阵的边缘进行滤波。零填充的一大好处是可以让我们控制特征图的大小。使用零填充的也叫做泛卷积，不适用零填充的叫做严格卷积。这个概念在下面的参考文献 14 中介绍的非常详细。

### Filter selection

在 CNN 的术语中，3x3 的矩阵叫做“滤波器（filter）”或者“核（kernel）”或者“特征检测器（feature detector）”，通过在图像上滑动滤波器并计算点乘得到矩阵叫做“卷积特征（Convolved Feature）”或者“激活图（Activation Map）”或者“特征图（Feature Map）”。记住滤波器在原始输入图像上的作用是特征检测器。

从上面图中的动画可以看出，对于同样的输入图像，不同值的滤波器将会生成不同的特征图。比如，对于下面这张输入图像：

![xiaolu](https://wx1.sinaimg.cn/mw690/0065SY2ely1fhyhtfwvv3j303q03o3yx.jpg)

In the table below, we can see the effects of convolution of the above image with different filters. As shown, we can perform operations such as Edge Detection, Sharpen and Blur just by changing the numeric values of our filter matrix before the convolution operation [8](https://en.wikipedia.org/wiki/Channel_(digital_image)) – this means that different filters can detect different features from an image, for example edges, curves etc. More such examples are available in Section 8.2.4 here.

在下表中，我们可以看到不同滤波器对上图卷积的效果。正如表中所示，通过在卷积操作前修改滤波矩阵的数值，我们可以进行诸如边缘检测、锐化和模糊等操作 —— 这表明不同的滤波器可以从图中检测到不同的特征，比如边缘、曲线等.

![卷积](https://wx3.sinaimg.cn/mw690/0065SY2ely1fhykfzx6wkj30ia0u2gtc.jpg)









> [通道](https://en.wikipedia.org/wiki/Channel_(digital_image)) 常用于表示图像的某种组成。一个标准数字相机拍摄的图像会有三通道 - 红、绿和蓝；你可以把它们看作是互相堆叠在一起的二维矩阵（每一个通道代表一个颜色），每个通道的像素值在 0 到 255 的范围内。
>
> [灰度](https://en.wikipedia.org/wiki/Grayscale)图像，仅仅只有一个通道。在本篇文章中，我们仅考虑灰度图像，这样我们就只有一个二维的矩阵来表示图像。矩阵中各个像素的值在 0 到 255 的范围内——零表示黑色，255 表示白色。





## 2.池化操作

空间池化（Spatial Pooling）（也叫做亚采用或者下采样）降低了各个特征图的维度，但可以保持大部分重要的信息。空间池化有下面几种方式：最大化、平均化、加和等等。

对于最大池化（Max Pooling），我们定义一个空间邻域（比如，2x2 的窗口），并从窗口内的修正特征图中取出最大的元素。除了取最大元素，我们也可以取平均（Average Pooling）或者对窗口内的元素求和。在实际中，最大池化被证明效果更好一些。

下面的图展示了使用 2x2 窗口在修正特征图（在卷积 + ReLU 操作后得到）使用最大池化的例子。

![ReLU](https://wx2.sinaimg.cn/mw690/0065SY2ely1fhyic5goncj30rg0ne436.jpg)

我们以 2 个元素（也叫做“步长”）滑动我们 2x2 的窗口，并在每个区域内取最大值。如上图所示，这样操作可以降低我们特征图的维度。

在下图展示的网络中，池化操作是分开应用到各个特征图的（注意，因为这样的操作，我们可以从三个输入图中得到三个输出图）。

![网络](https://wx1.sinaimg.cn/mw690/0065SY2ely1fhyklsq3fpj30ma0c7go0.jpg)

下图展示了在图 9 中我们在 ReLU 操作后得到的修正特征图的池化操作的效果。

![池化](https://wx4.sinaimg.cn/mw690/0065SY2ely1fhykmk1ehaj30z20eywht.jpg)

池化函数可以逐渐降低输入表示的空间尺度。特别地，池化：

- 使输入表示（特征维度）变得更小，并且网络中的参数和计算的数量更加可控的减小，因此，可以控制过拟合
- 使网络对于输入图像中更小的变化、冗余和变换变得不变性（输入的微小冗余将不会改变池化的输出——因为我们在局部邻域中使用了最大化/平均值的操作。
- 帮助我们获取图像最大程度上的尺度不变性（准确的词是“不变性”）。它非常的强大，因为我们可以检测图像中的物体，无论它们位置在哪里（参考 [18](https://wx4.sinaimg.cn/mw690/0065SY2ely1fhyia10lsxj30ma0b6mzm.jpg) 和 [19](https://wx4.sinaimg.cn/mw690/0065SY2ely1fhyid9loipj30tr09c3zj.jpg) 获取详细信息）。