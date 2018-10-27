#### 为什么用双向 LSTM？

单向的 RNN，是根据前面的信息推出后面的，但有时候只看前面的词是不够的， 
 例如，

我今天不舒服，我打算*__*一天。

只根据‘不舒服‘，可能推出我打算‘去医院‘，‘睡觉‘，‘请假‘等等，但如果加上后面的‘一天‘，能选择的范围就变小了，‘去医院‘这种就不能选了，而‘请假‘‘休息‘之类的被选择概率就会更大。



#### 什么是双向 LSTM？

双向卷积神经网络的隐藏层要保存两个值， A 参与正向计算， A’ 参与反向计算。 
最终的输出值 y 取决于 A 和 A’：

![](/Users/liuxingyu/Pictures/markdown/BRNN.png)

即正向计算时，隐藏层的 s_t 与 s_t－1 有关；反向计算时，隐藏层的 s_t 与 s_t＋1 有关：

![](/Users/liuxingyu/Pictures/markdown/brnn1.png)

o-->output

![](/Users/liuxingyu/Pictures/markdown/brnn2.png)

