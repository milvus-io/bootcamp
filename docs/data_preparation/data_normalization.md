# 数据归一化

  * [L2 正则化（归一化）](#l2-正则化归一化)
  * [计算向量相似度](#计算向量相似度)
    + [内积（点积）](#内积点积)
    + [余弦相似度](#余弦相似度)
    + [欧氏距离](#欧氏距离)

## L2 正则化（归一化）

开始之前，建议先对测试数据进行归一化处理。归一化处理后，点积、余弦相似度，欧氏距离之间有等价关系。

假设 n 维原始向量空间：<img src="http://latex.codecogs.com/gif.latex?\\R^n(n>0)" title="\\R^n(n>0)" />

原始向量：<img src="http://latex.codecogs.com/gif.latex?\\X&space;=&space;(x_1,&space;x_2,&space;...,&space;x_n),X&space;\in&space;\reals^n" title="\\X = (x_1, x_2, ..., x_n),X \in \reals^n" />

向量<img src="http://latex.codecogs.com/gif.latex?$$X$$" title="$$X$$" />的 L2 范数（模长）：

<img src="http://latex.codecogs.com/gif.latex?\\\|&space;X&space;\|&space;=&space;\sqrt{\displaystyle\sum_{i=1}^n&space;(x_i)&space;^2}" title="\\| X \| = \sqrt{\displaystyle\sum_{i=1}^n (x_i) ^2}" />

归一化后的向量：<img src="http://latex.codecogs.com/gif.latex?X'&space;=&space;(x_1',&space;x_2',&space;...,&space;x_n'),X'&space;\in&space;\reals^n" title="X' = (x_1', x_2', ..., x_n'),X' \in \reals^n" />

其中每一维的 L2 正则化算法：

<img src="http://latex.codecogs.com/gif.latex?x_i'&space;=&space;\frac{x_i}{\|&space;X&space;\|}&space;=&space;\frac{x_i}{\sqrt{\displaystyle\sum_{i=1}^n&space;(x_i)&space;^2}}" title="x_i' = \frac{x_i}{\| X \|} = \frac{x_i}{\sqrt{\displaystyle\sum_{i=1}^n (x_i) ^2}}" />

归一化后，向量模长等于 1：<img src="http://latex.codecogs.com/gif.latex?\|&space;X'&space;\|&space;=&space;1" title="\| X' \| = 1" />



## 计算向量相似度

近似最近邻搜索（approximate nearest neighbor searching, ANNS）是目前针对向量搜索的主流思路。其核心理念在于只在原始向量空间的子集中进行计算和搜索，从而加快整体搜索速度。

假设搜索空间（即原始向量空间的子集）：<img src="http://latex.codecogs.com/gif.latex?\gamma,&space;\gamma&space;\subset&space;R^n" title="\gamma, \gamma \subset R^n" />



### 内积（点积）

向量<img src="http://latex.codecogs.com/gif.latex?$$A,&space;B$$" title="$$A, B$$" />的内积：

<img src="http://latex.codecogs.com/gif.latex?$$p(A,B)&space;=&space;A&space;\cdot&space;B&space;=&space;\displaystyle\sum_{i=1}^n&space;a_i&space;\times&space;b_i$$" title="$$p(A,B) = A \cdot B = \displaystyle\sum_{i=1}^n a_i \times b_i$$" />



### 余弦相似度

向量<img src="http://latex.codecogs.com/gif.latex?$$A,&space;B$$" title="$$A, B$$" />的余弦相似度：

<img src="http://latex.codecogs.com/gif.latex?$$\cos&space;(A,B)&space;=&space;\frac{A&space;\cdot&space;B}{\|A&space;\|&space;\|B\|}$$" title="$$\cos (A,B) = \frac{A \cdot B}{\|A \| \|B\|}$$" />

通过余弦判断相似度：数值越大，相似度越高。即

 <img src="http://latex.codecogs.com/gif.latex?$$TopK(A)&space;=&space;\underset{B&space;\in\&space;\gamma}{\operatorname{argmax}}&space;\big&space;(&space;cos(A,B)&space;\big&space;)$$" title="$$TopK(A) = \underset{B \in\ \gamma}{\operatorname{argmax}} \big ( cos(A,B) \big )$$" />

假设向量<img src="http://latex.codecogs.com/gif.latex?$$A,&space;B$$" title="$$A, B$$" />归一化后的向量分别是<img src="http://latex.codecogs.com/gif.latex?$$A',&space;B'$$" title="$$A, B$$" />：

<img src="http://latex.codecogs.com/gif.latex?$$cos(A,B)&space;=&space;\frac{A&space;\cdot&space;B}{\|A&space;\|&space;\|B\|}&space;=&space;\frac{&space;\displaystyle\sum_{i=1}^n&space;a_i&space;\times&space;b_i}{\|A\|&space;\times&space;\|B\|}&space;=&space;\displaystyle\sum_{i=1}^n&space;\bigg(\frac{a_i}{\|A\|}&space;\times&space;\frac{b_i}{\|B\|}\bigg)=cos(A',B')&space;$$" title="$$cos(A,B) = \frac{A \cdot B}{\|A \| \|B\|} = \frac{ \displaystyle\sum_{i=1}^n a_i \times b_i}{\|A\| \times \|B\|} = \displaystyle\sum_{i=1}^n \bigg(\frac{a_i}{\|A\|} \times \frac{b_i}{\|B\|}\bigg)=cos(A',B') $$" />

因此，归一化后两个向量之间的余弦相似度不变。特别的，

<img src="http://latex.codecogs.com/gif.latex?$$cos(A',B')&space;=&space;\displaystyle\sum_{i=1}^n&space;\bigg(\frac{a_i}{\|A\|}&space;\times&space;\frac{b_i}{\|B\|}\bigg)=p(A',B')$$" title="$$cos(A',B') = \displaystyle\sum_{i=1}^n \bigg(\frac{a_i}{\|A\|} \times \frac{b_i}{\|B\|}\bigg)=p(A',B')$$" />


因此，**向量归一化后，内积与余弦相似度计算公式等价**。



### 欧氏距离

向量<img src="http://latex.codecogs.com/gif.latex?$$A,&space;B$$" title="$$A, B$$" />的欧式距离：

<img src="http://latex.codecogs.com/gif.latex?$$d(A,B)&space;=&space;\sqrt{\displaystyle\sum_{i=1}^n&space;(a_i-b_i)&space;^2}$$" title="$$d(A,B) = \sqrt{\displaystyle\sum_{i=1}^n (a_i-b_i) ^2}$$" />

通过欧氏距离判断相似度：欧式距离越小，相似度越高。即

<img src="http://latex.codecogs.com/gif.latex?$$TopK(A)&space;=&space;\underset{B&space;\in\&space;\gamma}{\operatorname{argmin}}&space;\big&space;(&space;d(A,B)&space;\big&space;)$$" title="$$TopK(A) = \underset{B \in\ \gamma}{\operatorname{argmin}} \big ( d(A,B) \big )$$" />

如果进一步展开上面的公式：

<img src="http://latex.codecogs.com/gif.latex?$$d(A,B)&space;=&space;\sqrt{\displaystyle\sum_{i=1}^n&space;(a_i-b_i)&space;^2}\\\\&space;=\sqrt{\displaystyle\sum_{i=1}^n&space;(a_i^2-2a_i&space;\times&space;b_i&plus;b_i^2)}\\\\&space;=\sqrt{\displaystyle\sum_{i=1}^n&space;a_i^2&plus;\displaystyle\sum_{i=1}^n&space;b_i^2-2\displaystyle\sum_{i=1}^n&space;a_i&space;\times&space;b_i}\\\\&space;=\sqrt{2-2&space;\times&space;p(A,B)}&space;\\\\&space;\therefore&space;d(A,B)^2&space;=&space;-2&space;\times&space;p(A,B)&space;&plus;&space;2$$" title="$$d(A,B) = \sqrt{\displaystyle\sum_{i=1}^n (a_i-b_i) ^2}\\\\ =\sqrt{\displaystyle\sum_{i=1}^n (a_i^2-2a_i \times b_i+b_i^2)}\\\\ =\sqrt{\displaystyle\sum_{i=1}^n a_i^2+\displaystyle\sum_{i=1}^n b_i^2-2\displaystyle\sum_{i=1}^n a_i \times b_i}\\\\ =\sqrt{2-2 \times p(A,B)} \\\\ \therefore d(A,B)^2 = -2 \times p(A,B) + 2$$" />

因此，欧氏距离的平方与内积负相关。而欧式距离是非负实数，两个非负实数之间的大小关系与他们自身平方之间的大小关系相同。

<img src="http://latex.codecogs.com/gif.latex?\lbrace&space;a,b,c&space;\rbrace&space;\subset&space;\lbrace&space;x&space;\in&space;R&space;|&space;x&space;\geqslant&space;0&space;\rbrace" title="\lbrace a,b,c \rbrace \subset \lbrace x \in R | x \geqslant 0 \rbrace" />

<img src="http://latex.codecogs.com/gif.latex?\because&space;a&space;\leqslant&space;b&space;\leqslant&space;c" title="\because a \leqslant b \leqslant c" />

<img src="http://latex.codecogs.com/gif.latex?\therefore&space;a^2&space;\leqslant&space;b^2&space;\leqslant&space;c^2" title="\therefore a^2 \leqslant b^2 \leqslant c^2" />

所以，**向量归一化后，针对同一个向量，在同等搜索空间的条件下，欧氏距离返回的前 K 个距离最近的向量结果集与内积返回的前 K 个相似度最大的向量结果集是等价的**。
