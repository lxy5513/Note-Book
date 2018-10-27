## **判别式（**discriminative**）模型 vs. 生成式(**generative**)模型**

在监督学习下，模型可以分为判别式模型与生成式模型。

==机器学习的主要任务就是：==

##### 	从 属性X 预测 标记Y

​		**方法：**

​		discriminative model 求概率$P（Y|X）​$   即后验概率

​		generative model $P(X, Y)$,  即联合概率 

#### **1. 首先区分生成/判别方法和生成/判别模型**

​	有监督机器学习方法可以分为生成方法和判别方法（常见的生成方法有LDA主题模型、朴素贝叶斯算法和隐式马尔科夫模型等，常见的判别方法有SVM、LR等），生成方法学习出的是生成模型，判别方法学习出的是判别模型。



![](/Users/liuxingyu/Pictures/markdown/Dis_Gen model.jpg)

### 2. Example

**判别式模型举例**：

> 要确定一个羊是山羊还是绵羊，用判别模型的方法是从历史数据中学习到模型，然后通过提取这只羊的特征来预测出这只羊是山羊的概率，是绵羊的概率。
>
> 对于二分类任务来说，实际得到一个score，当score大于threshold时则为正类，否则为反类

**生成式模型举例**：

> 利用生成模型是根据山羊的特征首先**学习出一个山羊的模型**，然后根据绵羊的特征**学习出一个绵羊的模型**，然后从这只羊中提取特征，放到山羊模型中看概率是多少，在放到绵羊模型中看概率是多少，==哪个大就是哪个==。

细细品味上面的例子，判别式模型是根据一只羊的特征可以直接给出这只羊的概率（比如logistic regression，这概率大于0.5时则为正例，否则为反例），而生成式模型是要都试一试，最大的概率的那个就是最后结果~



### 3. 特征

##### 判别式模型的特征总结如下：

1. 对 ![P(Y|X)](https://www.zhihu.com/equation?tex=P%28Y%7CX%29) 建模
2. 对所有的样本只构建一个模型，确认总体判别边界
3. 观测到输入什么特征，就预测最可能的label
4. 另外，判别式的优点是：对数据量要求没生成式的严格，速度也会快，小数据量下准确率也会好些。

##### 生成式总结下有如下特点：

1. 对 ![P(X,Y)](https://www.zhihu.com/equation?tex=P%28X%2CY%29) 建模
2. 这里我们主要讲分类问题，所以是要对每个label（ ![y_{i} ](https://www.zhihu.com/equation?tex=y_%7Bi%7D+) ）都需要建模，最终选择最优概率的label为结果，所以没有什么判别边界。（对于序列标注问题，那只需要构件一个model）
3. 中间生成联合分布，并可生成采样数据。
4. 生成式模型的优点在于，所包含的信息非常齐全，我称之为“上帝信息”，所以不仅可以用来输入label，还可以干其他的事情。生成式模型关注结果是如何产生的。但是生成式模型需要非常充足的数据量以保证采样到了数据本来的面目，所以速度相比之下，慢。

这一点明白后，后面讲到的HMM与CRF的区别也会非常清晰。



### 4. 求解思路

#### Discriminative Model

**判别模型求解的思路是：条件分布------>模型参数后验概率最大------->（似然函数![\cdot ](https://www.zhihu.com/equation?tex=%5Ccdot+)参数先验）最大------->最大似然**

**接着对生成模型和判别模型做更详细一点的解释。**
*这里定义训练数据为(C,X)，C={c1,c2,....cn}是n个训练样本的label，X={x1,x2....xn}是n个训练样本的feature。定义单个测试数据为(![\tilde{c} ](https://www.zhihu.com/equation?tex=%5Ctilde%7Bc%7D+),![\tilde{x} ](https://www.zhihu.com/equation?tex=%5Ctilde%7Bx%7D+)),![\tilde{c} ](https://www.zhihu.com/equation?tex=%5Ctilde%7Bc%7D+)为测试数据的lable，![\tilde{x} ](https://www.zhihu.com/equation?tex=%5Ctilde%7Bx%7D+)是测试样本的feature。*

训练完毕后，输入测试数据，判别模型**直接给出**的是![P(\tilde{c}|\tilde{x})](https://www.zhihu.com/equation?tex=P%28%5Ctilde%7Bc%7D%7C%5Ctilde%7Bx%7D%29)，即输出（label）关于输入（feature）的条件分布

#### Generative Model

给定输入*![\tilde{x} ](https://www.zhihu.com/equation?tex=%5Ctilde%7Bx%7D+)，*生成模型可以给出输入和输出的联合分布![P(\tilde{x},\tilde{c}) ](https://www.zhihu.com/equation?tex=P%28%5Ctilde%7Bx%7D%2C%5Ctilde%7Bc%7D%29+)，所以生成方法的目标是求出这个联合分布。这里**以朴素贝叶斯模型**为例，我们要求的目标可以通过：
![P(\tilde{x},\tilde{c}) ](https://www.zhihu.com/equation?tex=P%28%5Ctilde%7Bx%7D%2C%5Ctilde%7Bc%7D%29+)=![P(\tilde{x}|\tilde{c} )\cdot P(\tilde{c})](https://www.zhihu.com/equation?tex=P%28%5Ctilde%7Bx%7D%7C%5Ctilde%7Bc%7D+%29%5Ccdot+P%28%5Ctilde%7Bc%7D%29)
​	这样将求联合分布的问题转化成了求**类别先验概率和类别条件概率**的问题，朴素贝叶斯方法做了一个较强的假设--------feature的不同维度是独立分布的，简化了类别条件概率的计算，如果去除假设就是贝叶斯网络，这里不再赘述。
以朴素贝叶斯为例，**生成模型的求解思路是：联合分布------->求解类别先验概率和类别条件概率**



### **5. 优缺点：**

#### 生成模型：

优点：
1）生成给出的是联合分布![P(\tilde{x},\tilde{c}) ](https://www.zhihu.com/equation?tex=P%28%5Ctilde%7Bx%7D%2C%5Ctilde%7Bc%7D%29+)，不仅能够由联合分布计算条件分布![P(\tilde{c}|\tilde{x})](https://www.zhihu.com/equation?tex=P%28%5Ctilde%7Bc%7D%7C%5Ctilde%7Bx%7D%29)（反之则不行），还可以给出其他信息，比如可以使用![P(\tilde{x} )=\sum_{i=1}^{k}{P(\tilde{x}|\tilde{c}_{i})} ](https://www.zhihu.com/equation?tex=P%28%5Ctilde%7Bx%7D+%29%3D%5Csum_%7Bi%3D1%7D%5E%7Bk%7D%7BP%28%5Ctilde%7Bx%7D%7C%5Ctilde%7Bc%7D_%7Bi%7D%29%7D+)![*P(\tilde{c_i} )](https://www.zhihu.com/equation?tex=%2AP%28%5Ctilde%7Bc_i%7D+%29)来计算边缘分布![P(\tilde{x} )](https://www.zhihu.com/equation?tex=P%28%5Ctilde%7Bx%7D+%29)。如果一个输入样本的边缘分布![P(\tilde{x} )](https://www.zhihu.com/equation?tex=P%28%5Ctilde%7Bx%7D+%29)很小的话，那么可以认为学习出的这个模型可能不太适合对这个样本进行分类，分类效果可能会不好，这也是所谓的*outlier detection。*
2）生成模型收敛速度比较快，即当样本数量较多时，生成模型能更快地收敛于真实模型。
3）生成模型能够应付存在隐变量的情况，比如混合高斯模型就是含有隐变量的生成方法。

缺点：
1）天下没有免费午餐，联合分布是能提供更多的信息，但也需要更多的样本和更多计算，尤其是为了更准确估计类别条件分布，需要增加样本的数目，而且类别条件概率的许多信息是我们做分类用不到，因而如果我们只需要做分类任务，就浪费了计算资源。
2）另外，实践中多数情况下判别模型效果更好。

#### 判别模型：

优点：
1）与生成模型缺点对应，首先是==节省计算资源==，另外，需要的样本数量也少于生成模型。
2）准确率往往较生成模型高。
3）由于直接学习![P(\tilde{c}|\tilde{x} )](https://www.zhihu.com/equation?tex=P%28%5Ctilde%7Bc%7D%7C%5Ctilde%7Bx%7D+%29)，而不需要求解类别条件概率，所以允许我们对输入进行抽象（比如降维、构造等），从而能够简化学习问题。

缺点：

1）是没有生成模型的上述优点。