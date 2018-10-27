## Deep Feedforward Networks

**深度前馈网络（Deep Feedforward Networks）**：也被称为前馈神经网络（feedforward neural networks），或者多层感知机（multi-layer perceptrons， MLPs）是典型的深度学习模型。前馈网络的目标是去近似一个函数f∗f∗。模型之所以称为前馈，是因为信息只向前流动，没有反馈的连接。

**基于梯度的学习（Gradient Based Learning）**：神经网络模型和线性模型最大的区别在于神经网络的非线性使得损失函数**不再是凸函数**。这意味着神经网络的训练通常使用迭代的、基于梯度的优化，仅仅使得代价函数达到一个非常小的值；而不是像用于训练线性回归模型的线性方程求解器，或者用于训练逻辑回归或SVM的**凸优化算法**那样保证全局收敛。 凸优化从任何一种初始参数出发都会收敛（理论上如此——在实践中也很鲁棒但可能会遇到数值问题）。 用于非凸损失函数的随机梯度下降没有这种收敛性保证，并且对参数的初始值很敏感。 对于前馈神经网络，将所有的权重值初始化为小随机数是很重要的。 偏置可以初始化为零或者小的正值。

**输出单元：**

**1.用于高斯输出分布的线性单元（Linear Output Units）**：ŷ =WTh+by^=WTh+b，通常用来预测条件高斯分布：p(y|x)=N(y;ŷ ,I)p(y|x)=N(y;y^,I)

**2.用于Bernoulli输出分布的sigmoid单元（Sigmoid Output Units）**：二分类任务，可以通过这个输出单元解决。ŷ =σ(wTh+b)y^=σ(wTh+b)，其中，σ是sigmoid函数。

**3.用于 Multinoulli输出分布的softmax单元（Softmax Output Units）**：z=Wth+bz=Wth+b，而softmax(z)i=exp(zi)∑jexp(zj)softmax(z)i=exp(zi)∑jexp(zj)，如果说argmax函数返回的是一个onehot的向量，那么softmax可以理解成soft版的argmax函数。

**隐藏单元：**

**1.修正线性单元（Rectified Linear Units，ReLU）**：使用激活函数g(z)=max{0,z}g(z)=max{0,z}，有h=g(WTx+b)h=g(WTx+b)。通常b的初始值选一个小正值，如0.1。这样relu起初很可能是被激活的。relu的一个缺点是它不能在激活值是0的时候，进行基于梯度的学习。因此又产生了各种变体。

**1.1.maxout单元：整流线性单元的一种扩展**：g(z)i=maxj∈𝔾(i)zjg(z)i=maxj∈G(i)zj，其中，𝔾(i)G(i)是第i组的输入索引集{(i−1)k+1,…,ik}{(i−1)k+1,…,ik}。

**2.logistic sigmoid与双曲正切函数（Hyperbolic Tangent）单元**：使用logistic sigmoid：g(z)=σ(z)g(z)=σ(z)；使用双曲正弦函数：g(z)=tanh(z)g(z)=tanh(z)，其中, tanh(z)=2σ(2z)−1tanh(z)=2σ(2z)−1。 但是，在这两个函数的两端都很容易饱和，所以不鼓励用在隐藏单元中，一定要用可以优先选择双曲正弦函数。

**通用近似性质（Universal Approximation Properties）**：一个前馈神经网络如果具有线性输出层和至少一层具有激活函数（例如logistic sigmoid激活函数）的隐藏层，只要给予网络足够数量的隐藏单元，它可以以任意的精度来近似任何从一个有限维空间到另一个有限维空间的Borel可测函数。 虽然具有单层的前馈网络足以表示任何函数，但是网络层可能大得不可实现，并且可能无法正确地学习和泛化。 在很多情况下，使用更深的模型能够减少表示期望函数所需的单元的数量，并且可以减少泛化误差。

**MLP的深度（Depth）**：具有d个输入、深度为l、每个隐藏层具有n个单元的深度整流网络可以描述的线性区域的数量是O((nd)d(l−1)nd)O((nd)d(l−1)nd),意味着，这是深度l的指数级。

**后向传播算法（Back-Propagation）**：后向传播算法将偏差（cost）在网络中从后往前传播，用来计算关于cost的梯度。后向传播算法本身不是学习算法，而是学习算法，像SGD，使用后向传播算法来计算梯度。对于bp的生动理解，可以参考[知乎的这个回答](https://zhihu.com/question/27239198/answer/89853077)，“同样是利用链式法则，BP算法则机智地避开了这种冗余，它对于每一个路径只访问一次就能求顶点对所有下层节点的偏导值”；“BP算法就是主动还款。e把所欠之钱还给c，d。c，d收到钱，乐呵地把钱转发给了a，b，皆大欢喜”。