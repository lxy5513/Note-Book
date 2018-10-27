## 1.条件随机场（condition random filed)

> CRF模型因其强大的序列建模能力，几乎成为了序列标注任务下最经典的算法
>
> 条件随机场是逻辑回归的序列化版本。逻辑回归是用于分类的对数线性模型，条件随机场是用于序列化标注的对数线性模型。





## 2.Markov property

> ###### 下一状态的概率分布只能由当前状态决定，在时间序列中它前面的事件均与之无关。这种特定类型的“无记忆性”称作马尔可夫性质

#### Markov process

> 马尔可夫过程的[条件概率](https://zh.wikipedia.org/wiki/%E6%9D%A1%E4%BB%B6%E6%A6%82%E7%8E%87)仅仅与系统的当前状态相关，而与它的过去历史或未来状态，都是[独立](https://zh.wikipedia.org/wiki/%E7%B5%B1%E8%A8%88%E7%8D%A8%E7%AB%8B%E6%80%A7)、不相关的

#### Markov chian

> 具备离散状态的马尔可夫过程，通常被称为马尔可夫链。马尔可夫链通常使用离散的时间集合定义



#### 隐马尔可夫模型 (Hidden Markov Model)

HMM是统计模型，它用来描述一个含有隐含未知参数的马尔可夫过程。

##### 三种应用

> 已知模型参数和某一特定输出序列，求最后时刻各个隐含状态的概率分布
>
> 已知模型参数和某一特定输出序列，求中间时刻各个隐含状态的概率分布
>
> 已知模型参数，寻找最可能的能产生某一特定输出序列的隐含状态的序列. 





## 4.word embedding

就是找到一个映射或者函数，生成在一个新的空间上的表达，该表达就是word representation。

降低训练所需要的数据量。从数据中自动学习到输入空间到Distributed representation空间的 映射![f](https://www.zhihu.com/equation?tex=f) 。



## 5.训练、测试、验证集

**Training set**: A set of examples used for learning, which is to fit the parameters [i.e., weights] of the classifier.

**Validation set:** A set of examples used to tune the parameters [i.e.,
 architecture, not weights] of a classifier, for example to choose the 
number of hidden units in a neural network.

**Test set**: A set of examples used only to assess the performance [generalization] of a fully specified classifier.



## 6.性能度量

错误率

> 分类错误占样本总量的比例

精度

> 分类正确的数量占样板总量的个数

| P代表查到的     | TP(真正例)      | FN (假反例）    |
| --------------- | --------------- | --------------- |
| **T代表正确的** | **FP (假正例)** | **TN (真反例)** |

查准率precision

> P = TP/(TP+FP)

查全率recall

> R = TP/(TP+FN) 

查准率和查全率是一对矛盾的度量.一般来说，查准率高时，查全率往往
偏低;而查全率高时，查准率往往偏低.例如，若希望将好瓜尽可能多地选出来，
则可通过增加选瓜的数量来实现，如果将所有西瓜都选上，那么所有的好瓜也

必然 都被 选上了，但这 样查准率 就会 较低;若希望选 出 的瓜中好瓜比 例尽 可能
高，则可只挑选最有把握的瓜， 但这样就难免会漏掉不少好瓜，使得查全率较
低.通常只有在一些简单任务中 7 才可能使查全率和查准率都很高.



## 7.pre-training

每次训练一层隐结点?训练时将上一层隐结点的输
出作为输入，向本层隐结点的输出作为下一层隐结点的输入，这称为"预训
练" (pre-training);



## 8.Transfer Learning

Transfer Learning关心的问题是：什么是“知识”以及如何更好地运用之前得到的“知识”。这可以有很多方法和手段。而**fine-tune**只是其中的一种实现手段, transfer learning却是一个很大的体系



## 9.RBM

受限玻尔兹曼机（Restricted Boltzmann Machine，RBM）是一种可用**随机神经网络**（stochastic neural network）来解释的**概率图模型**（probabilistic graphical model）。RBM是Smolensky于1986年在波尔兹曼机（Boltzmann Machine，BM）基础上提出的，所谓“随机”是指网络中的神经元是随机神经元，输出状态只有两种（未激活和激活），状态的具体取值根据概率统计法则来决定。



## 10.Monte Carlo Method

 **蒙特卡罗方法**又称统计模拟**法**、随机抽样技术，是一种随机模拟**方法**，以概率和统计理论**方法**为基础的一种计算**方法**，是使用随机数（或更常见的伪随机数）来解决很多计算问题的**方法**。 将所求解的问题同一定的概率模型相联系，用电子计算机实现统计模拟或抽样，以获得问题的近似解。



## 11.MaxPooling

MaxPooling有以下几点作用：1. 减少运算量；2. 一个卷积核可以认为是一种特征提取器，遇到符合的特征会得到较大的值，通过取max可以避免其他不相关特征的干扰； 3. 同样的特征，会存在强弱、位置等方面的差异，但都是相同的特征，通过取max可以避免位置带来的干扰。



## 12.DropOut

dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。注意是暂时，对于随机梯度下降来说，由于是随机丢弃，故而每一个mini-batch都在训练不同的网络。

dropout是CNN中防止过拟合提高效果的 一个大杀器 

训练时dropout keep prob为0.5 0.6
test时换成1.0



## 13.遇到Nan怎么办？

Nan问题，我相信大部分人都遇到过，一般可能是下面几个原因造成的：

1. 除0问题。这里实际上有两种可能，一种是被除数的值是无穷大，即Nan，另一种就是除数的值是0。之前产生的Nan或者0，有可能会被传递下去，造成后面都是Nan。请先检查一下神经网络中有可能会有除法的地方，例如softmax层，再认真的检查一下数据。我有一次帮别人调试代码，甚至还遇到过，训练数据文件中，有些值就是Nan。。。这样读进来以后，开始训练，只要遇到Nan的数据，后面也就Nan了。可以尝试加一些日志，把神经网络的中间结果输出出来，看看哪一步开始出现Nan。后面会介绍Theano的处理办法。
2. 梯度过大，造成更新后的值为Nan。特别是RNN，在序列比较长的时候，很容易出现梯度爆炸的问题。一般有以下几个解决办法。
   1. 对梯度做clip(梯度裁剪），限制最大梯度,其实是value = sqrt(w1^2+w2^2….),如果value超过了阈值,就算一个衰减系系数,让value的值等于阈值: 5,10,15。
   2. 减少学习率。初始学习率过大，也有可能造成这个问题。需要注意的是，即使使用adam之类的自适应学习率算法进行训练，也有可能遇到学习率过大问题，而这类算法，一般也有一个学习率的超参，可以把这个参数改的小一些。
3. 初始参数值过大，也有可能出现Nan问题。输入和输出的值，最好也做一下归一化





## 14.Xavier 初始化

既保证输入输出的差异性，又能让model稳定而快速的收敛

让model的训练速度和分类性能取得大幅提高



## 15.Bias，Error，和Variance(方差)有什么区别和联系

##### Error = Bias + Variance

Error反映的是整个模型的准确度，Bias反映的是模型在样本上的输出与真实值之间的误差，即模型本身的精准度，Variance反映的是模型每一次输出结果与模型输出期望之间的误差，即模型的稳定性

![](/Users/liuxingyu/Pictures/ML/basic/v2-286539c808d9a429e69fd59fe33a16dd_hd.png)

- 准：

  bias描述的是根据样本拟合出的模型的输出预测结果的期望与样本真实结果的差距，简单讲，就是在样本上拟合的好不好。要想在bias上表现好，low bias，就得复杂化模型，增加模型的参数，但这样容易过拟合 (overfitting)，过拟合对应上图是high variance，点很分散。low bias对应就是点都打在靶心附近，所以瞄的是准的，但手不一定稳。



- 确：

  varience描述的是样本上训练出来的模型在测试集上的表现，要想在variance上表现好，low varience，就要简化模型，减少模型的参数，但这样容易欠拟合(unfitting)，欠拟合对应上图是high bias，点偏离中心。low variance对应就是点都打的很集中，但不一定是靶心附近，手很稳，但是瞄的不准。





## 16.**遗传算法**（英语：genetic algorithm (GA) ）

​	是[计算数学](https://zh.wikipedia.org/wiki/%E8%AE%A1%E7%AE%97%E6%95%B0%E5%AD%A6)中用于解决[最优化](https://zh.wikipedia.org/wiki/%E6%9C%80%E4%BD%B3%E5%8C%96)的搜索[算法](https://zh.wikipedia.org/wiki/%E7%AE%97%E6%B3%95)，是[进化算法](https://zh.wikipedia.org/wiki/%E8%BF%9B%E5%8C%96%E7%AE%97%E6%B3%95)的一种。进化算法最初是借鉴了[进化生物学](https://zh.wikipedia.org/wiki/%E8%BF%9B%E5%8C%96%E7%94%9F%E7%89%A9%E5%AD%A6)中的一些现象而发展起来的，这些现象包括[遗传](https://zh.wikipedia.org/wiki/%E9%81%97%E4%BC%A0)、[突变](https://zh.wikipedia.org/wiki/%E7%AA%81%E5%8F%98)、[自然选择](https://zh.wikipedia.org/wiki/%E8%87%AA%E7%84%B6%E9%80%89%E6%8B%A9)以及[杂交](https://zh.wikipedia.org/wiki/%E6%9D%82%E4%BA%A4)等

##### 机理

​     	在遗传算法里，优化问题的解被称为个体，它表示为一个变量序列，叫做[染色体](https://zh.wikipedia.org/wiki/%E6%9F%93%E8%89%B2%E9%AB%94_(%E9%81%BA%E5%82%B3%E6%BC%94%E7%AE%97%E6%B3%95))或者[基因](https://zh.wikipedia.org/wiki/%E5%9F%BA%E5%9B%A0)[串](https://zh.wikipedia.org/wiki/%E5%AD%97%E7%AC%A6%E4%B8%B2)。染色体一般被表达为简单的字符串或数字符串，不过也有其他的依赖于特殊问题的表示方法适用，这一过程称为编码。首先，算法[随机](https://zh.wikipedia.org/wiki/%E9%9A%8F%E6%9C%BA%E5%87%BD%E6%95%B0)生成一定数量的个体，有时候操作者也可以干预这个随机产生过程，以提高初始种群的质量。在每一代中，都会评价每一个体，并通过计算[适应度函数](https://zh.wikipedia.org/w/index.php?title=%E9%80%82%E5%BA%94%E5%BA%A6%E5%87%BD%E6%95%B0&action=edit&redlink=1)得到[适应度](https://zh.wikipedia.org/wiki/%E9%80%82%E5%BA%94%E5%BA%A6)数值。按照适应度[排序](https://zh.wikipedia.org/wiki/%E6%8E%92%E5%BA%8F)种群个体，适应度高的在前面。这里的“高”是相对于初始的种群的低适应度而言

##### 算法 

- 选择初始生命种群
- 循环
  - 评价种群中的个体适应度
  - 以比例原则（分数高的挑中机率也较高）选择产生下一个种群（[轮盘法](https://zh.wikipedia.org/w/index.php?title=%E8%BC%AA%E7%9B%A4%E6%B3%95&action=edit&redlink=1)（roulette wheel selection）、[竞争法](https://zh.wikipedia.org/wiki/%E7%AB%B6%E7%88%AD%E6%B3%95)（tournament selection）及[档次轮盘法](https://zh.wikipedia.org/w/index.php?title=%E7%AD%89%E7%B4%9A%E8%BC%AA%E7%9B%A4%E6%B3%95&action=edit&redlink=1)（Rank Based Wheel Selection））。不仅仅挑分数最高的的原因是这么做可能收敛到局部的最佳点，而非整体的。
  - 改变该种群（交叉和变异）
- 直到停止循环的条件满足





## 17.为什么需要输入图片的大小固定

我们知道卷积层对于图像的大小是没有要求的，一般的卷积核都是3*3, 5*5等，而输入图像一般不会小于这个大小。所以问题就是出在全连接层。

链接层输入向量的维数对应全链接层的神经元个数，所以如果输入向量的维数不固定，那么全链接的权值参数个数也是不固定的

我们假设全连接层到输出层之间的参数是 ![W^{f*o}](https://www.zhihu.com/equation?tex=W%5E%7Bf%2Ao%7D)

- ![f](https://www.zhihu.com/equation?tex=f) 表示全连接层的节点个数
- ![o](https://www.zhihu.com/equation?tex=o) 表示输出层的节点个数

很显然 ![o](https://www.zhihu.com/equation?tex=o) 一般是固定的，而 ![f](https://www.zhihu.com/equation?tex=f) 则会随着输入图像大小的变化而变化。

#### 解决办法

##### 1).resize or crop

这种方法比较粗暴，而且会在预处理环节增加很大的计算量，一般而言是没有办法的办法。

##### 2).SPP(Spatial Pyramid Pooling 空间金字塔池化)

只需要在全连接层加上SPP layer就可以很好的解决题主的问题



### 