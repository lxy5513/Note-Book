### Introduce

Attention is simply **a vector**, often the outputs of dense layer using softmax function

==Before== Attention mechanism, translation relies on **reading a complete sentence** and **compress** all information into a **fixed-length** vector, as you can image, a sentence with hundreds of words represented by several words will surely lead to information loss, inadequate translation

**attention partially fixes this problem**

> It allows machine translator to look over all the information the original sentence holds, then generate the proper word according to current word it works on and the context



 Attention is just an **interface** formulated **by parameters and delicate math**. You could **plug it anywhere** you find it suitable, and potentially, the result may be enhanced



### Why Attention?

The core of Probabilistic Language Model is to assign a probability to a sentence by Markov Assumption(下一状态的概率分布只能由当前状态决定，在时间序列中它前面的事件均与之无关)

#### general rnn drawbacks

> 1. Structure Dilemma: in real world, the length of outputs and inputs can be totally different, while Vanilla RNN can only handle fixed-length problem which is difficult for the alignment
>
> 2. Mathematical Nature: it suffers from Gradient Vanishing/Exploding which means it is hard to train when sentences are long enough

#### encode-decode

Translation often requires arbitrary input length and out put length, to deal with the deficits above, encoder-decoder model is adopted and basic RNN cell is changed to GRU[1] or LSTM cell, hyperbolic tangent activation is replaced by ReLU

![](/Users/liuxingyu/Pictures/ML/NLP/encode-decode.png)

encode-decode model(with GRU)



> as picture shown, embedded word vectors are fed into encoder, aka(also known as) GRU cells sequentially. What happened during encoding? Information flows from left to right and each word vector is learned according to not only **current input** but also **all previous words.** 
>
> When the sentence is completely read, encoder generates **an output** and **a hidden state** at timestep 4 for further processing
>
> For encoding part, decoder (GRUs as well) **grabs the hidden state** from encoder, trained by teacher forcing (a mode that previous cell’s output as current input), then generate translation words sequentially.

one main deficit left unsolved: is one hidden state really **enough**?



#### Attention

Similar to the basic encoder-decoder architecture, this fancy mechanism plug **a context vector** into the gap between encoder and decoder

![](/Users/liuxingyu/Pictures/ML/NLP/attention.png)



that context vector takes **all cells’ outputs as input** to compute the probability distribution of source language words for **each single word** decoder wants to generate. 

> By utilizing this mechanism, it is possible for decoder to capture somewhat global information rather than solely to infer based on one hidden state.

##### build context vector

For a fixed target word, first, we loop over all encoders’ states to compare target and source states to generate scores for each state in encoders. 

Then we could use softmax to normalize all scores, which generates the probability distribution conditioned on target states.

At last, the weights are introduced to make context vector easy to train. That’s it. Math is shown below:

![](/Users/liuxingyu/Pictures/ML/NLP/attention_math.png)



### Conclusion

We hope you understand the reason why attention is one of the hottest topics today, and most importantly, the basic math behind attention. 

Implementing your own attention layer is encouraged. There are many variants in the cutting-edge researches, and they basically differ in the choice of score function and attention function, or of soft attention and hard attention (whether differentiable). But basic concepts are all the same.



### annotation

[1] 通过引入门（Gate） 的机制来解决RNN的梯度消失问题，从而学习到长距离依赖, GRU（Gated Recurrent Units）可以看成是LSTM的变种，GRU把LSTM中的`forget gate`和`input gate`用`update gate`来替代。 把`cell state`和隐状态htht进行合并