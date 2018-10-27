## Rectified Linear Unit

### Introduce

**线性整流函数**（Rectified Linear Unit, **ReLU**）,又称**修正线性单元**, 是一种[人工神经网络](https://zh.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)中常用的激活函数（activation function），通常指代以[斜坡函数](https://zh.wikipedia.org/wiki/%E6%96%9C%E5%9D%A1%E5%87%BD%E6%95%B0)及其变种为代表的非线性函数。有一定的生物学原理。



### 定义

通常意义下，线性整流函数指代数学中的[斜坡函数](https://zh.wikipedia.org/wiki/%E6%96%9C%E5%9D%A1%E5%87%BD%E6%95%B0)，即

![R(x):={\begin{cases}x,&x\geq 0;\\0,&x<0\end{cases}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/ddc1da7c2934d2af5b5edafb534f57165bd8229c)

或者

![{\displaystyle f(x)=\max(0,x)}](https://wikimedia.org/api/rest_v1/media/math/render/svg/5fa5d3598751091eed580bd9dca873f496a2d0ac)



而在神经网络中，线性整流作为神经元的激活函数，定义了该神经元在线性变换![{\displaystyle \mathbf {w} ^{T}\mathbf {x} +b}](https://wikimedia.org/api/rest_v1/media/math/render/svg/7f9abf20c2e7f8813bf37a93378e53cf8f0f7461)之后的非线性输出结果。换言之，对于进入神经元的来自上一层神经网络的输入向量 ![x](https://wikimedia.org/api/rest_v1/media/math/render/svg/87f9e315fd7e2ba406057a97300593c4802b53e4)，使用线性整流激活函数的神经元会输出

![{\displaystyle f(x)=\max(0,x)}](https://wikimedia.org/api/rest_v1/media/math/render/svg/5fa5d3598751091eed580bd9dca873f496a2d0ac)

至下一层神经元或作为整个神经网络的输出（取决现神经元在网络结构中所处位置）。



### 优势

相比于传统的神经网络激活函数，诸如[逻辑函数](https://zh.wikipedia.org/wiki/%E9%80%BB%E8%BE%91%E5%87%BD%E6%95%B0)（Logistic sigmoid）和tanh等[双曲函数](https://zh.wikipedia.org/wiki/%E5%8F%8C%E6%9B%B2%E5%87%BD%E6%95%B0)，线性整流函数有着以下几方面的优势：

- 仿生物学原理：相关大脑方面的研究表明生物神经元的信息编码通常是比较分散及稀疏的。通常情况下，大脑中在同一时间大概只有1%-4%的神经元处于活跃状态。使用线性修正以及正则化（regularization）可以对机器神经网络中神经元的活跃度（即输出为正值）进行调试；相比之下，逻辑函数在输入为0时达到 ![{\frac {1}{2}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/a11cfb2fdb143693b1daf78fcb5c11a023cb1c55)，即已经是半饱和的稳定状态，不够符合实际生物学对模拟神经网络的期望。不过需要指出的是，一般情况下，在一个使用修正线性单元（即线性整流）的神经网络中大概有50%的神经元处于激活态。

- 更加有效率的[梯度下降](https://zh.wikipedia.org/wiki/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95)以及反向传播：避免了梯度爆炸和[梯度消失](https://zh.wikipedia.org/wiki/%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E9%97%AE%E9%A2%98)问题

- 简化计算过程：没有了其他复杂激活函数中诸如指数函数的影响；同时活跃度的分散性使得神经网络整体计算成本下降