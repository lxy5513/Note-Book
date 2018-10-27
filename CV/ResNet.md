## general structure

![](/Users/liuxingyu/Pictures/markdown/resnet.jpg)



![](/Users/liuxingyu/Pictures/markdown/resnet1.jpg)

ResNet使用两种残差单元，如图6所示。左图对应的是浅层网络，而右图对应的是深层网络 



### Sussessful reasons

第一，是它的shortcut connection增加了它的信息流动.

第二，就是它认为对于一个堆叠的非线性层，那么它最优的情况就是让它成为一个恒等映射，但是shortcut connection的存在恰好使得它能够更加容易的变成一个Identity Mapping。

## advantages

但是更深的网络其性能一定会更好吗？实验发现深度网络出现了退化问题（Degradation problem）：网络深度增加时，网络准确度出现饱和，甚至出现下降。深层网络存在着**梯度消失或者爆炸的问题**，这使得深度学习模型很难训练。

**残差学习**来解决退化问题。对于一个堆积层结构（几层堆积而成）当输入为 ![x](https://www.zhihu.com/equation?tex=x) 时其学习到的**特征记为** ![H(x)](https://www.zhihu.com/equation?tex=H%28x%29) ，现在我们希望其可以学习到残差 ![F(x)=H(x)-x](https://www.zhihu.com/equation?tex=F%28x%29%3DH%28x%29-x) ，这样其实原始的学习特征是 ![F(x)+x](https://www.zhihu.com/equation?tex=F%28x%29%2Bx) 。之所以这样是因为残差学习相比原始特征直接学习更容易。**当残差为0时，此时堆积层仅仅做了恒等映射，至少网络性能不会下降**，实际上残差不会为0，这也会使得堆积层在输入特征基础上学习到新的特征，从而拥有更好的性能。残差学习的结构如图4所示。这有点类似与电路中的“短路”，所以是一种短路连接（shortcut connection）。

![img](/Users/liuxingyu/Pictures/markdown/resnet2.jpg)

a building block

## keypoints

#### one

ResNet的一个重要设计原则是：当feature map大小降低一半时，feature map的数量增加一倍，这保持了网络层的复杂度。



#### two

对于短路连接，当输入和输出维度一致时，可以直接将输入加到输出上。但是当维度不一致时（对应的是维度增加一倍），这就不能直接相加。有两种策略：（1）采用zero-padding增加维度，此时一般要先做一个downsamp，可以采用strde=2的pooling，这样不会增加参数；（2）采用新的映射（projection shortcut），一般采用1x1的卷积，这样会增加参数，也会增加计算量。短路连接除了直接使用恒等映射，当然都可以采用projection shortcut。



## improve

改进了残差块，并提出了一种残差块的预激活变体 ，梯度可以在该模型中畅通无阻地通过快速连接到达之前的任意一层

![](/Users/liuxingyu/Pictures/markdown/resnet3.jpg) 

> a ----> resnet v1
>
> e ----> resnet v2



## **ResNet 的最新变体以及解读**

### **ResNeXt**

![](/Users/liuxingyu/Pictures/markdown/resnetx.jpg)

*左： ResNet 的构建块；右：ResNeXt 的构建块，基数=32*



ResNext 看起来和 [4] 中的 Inception 模块非常相似，它们都遵循了**「分割-转换-合并」**的范式。不过在 ResNext 中，不同路径的输出通过相加合并，而在 [4] 中它们是深度级联（depth concatenated）的。另外一个区别是，[4] 中的每一个路径互不相同（1x1、3x3 和 5x5 卷积），而在 ResNeXt 架构中，所有的路径都遵循相同的拓扑结构。

作者在论文中引入了一个叫作**「基数」（cardinality）**的超参数，指独立路径的数量，这提供了一种调整模型容量的新思路。实验表明，通过扩大基数值（而不是深度或宽度），准确率得到了高效提升。作者表示，与 Inception 相比，这个全新的架构更容易适应新的数据集或任务，因为它只有一个简单的范式和一个需要调整的超参数，而 Inception 需要调整很多超参数（比如每个路径的卷积层内核大小）。

这个全新的结构有三种等价形式：

![](/Users/liuxingyu/Pictures/markdown/resnetx1.jpg)

在实际操作中，「分割-变换-合并」范式通常通过「逐点分组卷积层」来完成，这个卷积层将输入的特征映射分成几组，并分别执行正常的卷积操作，其输出被深度级联，然后馈送到一个 1x1 卷积层中。



### **密集连接卷积神经网络**

Huang 等人在论文 [9] 中提出一种新架构 DenseNet，进一步利用快捷连接，将所有层直接连接在一起。在这种新型架构中，**每层的输入由所有之前层的特征映射组成**，其输出将传输给每个后续层。这些特征映射通过深度级联聚合。

![](/Users/liuxingyu/Pictures/markdown/densenet.jpg)

除了解决梯度消失问题，[8] 的作者称这个架构还支持特征重用，使得网络具备更高的参数效率。一个简单的解释是，在论文 [2] 和论文 [7] 中，恒等映射的输出被添加到下一个模块，如果两个层的特征映射有着非常不同的分布，那么这可能会阻碍信息流。因此，级联特征映射可以保留所有特征映射并增加输出的方差，从而促进特征重用。

### **深度随机的深度网络**

尽管 ResNet 的强大性能在很多应用中已经得到了证实，但它存在一个显著缺点：深层网络通常**需要进行数周的训练时间**。因此，把它应用在实际场景的成本非常高。为了解决这个问题，G. Huang 等作者在论文 [10] 中引入了一种反直觉的方法，即在训练过程中随机丢弃一些层，测试中使用完整的网络.

作者使用残差块作为他们网络的构建块。因此在训练期间，当特定的残差块被启用，它的输入就会同时流经恒等快捷连接和权重层；否则，就只流过恒等快捷连接。训练时，每层都有一个「生存概率」，==每层都有可能被随机丢弃==。在测试时间内，所有的块都保持被激活状态，并根据其生存概率进行重新校准。





**References:**

*[1]. A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems,pages1097–1105,2012.*

*[2]. K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. arXiv preprint arXiv:1512.03385,2015.*

*[3]. K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556,2014.*

*[4]. C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,pages 1–9,2015.*

*[5]. R. Srivastava, K. Greff and J. Schmidhuber. Training Very Deep Networks. arXiv preprint arXiv:1507.06228v2,2015.*

*[6]. S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Comput., 9(8):1735–1780, Nov. 1997.*

*[7]. K. He, X. Zhang, S. Ren, and J. Sun. Identity Mappings in Deep Residual Networks. arXiv preprint arXiv:1603.05027v3,2016.*

*[8]. S. Xie, R. Girshick, P. Dollar, Z. Tu and K. He. Aggregated Residual Transformations for Deep Neural Networks. arXiv preprint arXiv:1611.05431v1,2016.*

*[9]. G. Huang, Z. Liu, K. Q. Weinberger and L. Maaten. Densely Connected Convolutional Networks. arXiv:1608.06993v3,2016.*

*[10]. G. Huang, Y. Sun, Z. Liu, D. Sedra and K. Q. Weinberger. Deep Networks with Stochastic Depth. arXiv:1603.09382v3,2016.*

*[11]. N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever and R. Salakhutdinov. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. The Journal of Machine Learning Research 15(1) (2014) 1929–1958.*

*[12]. A. Veit, M. Wilber and S. Belongie. Residual Networks Behave Like Ensembles of Relatively Shallow Networks. arXiv:1605.06431v2,2016.*