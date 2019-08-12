- [硬件推荐配置](#硬件推荐配置)
- [测试数据准备](#测试数据准备)
  * [L2 正则化（归一化）](#l2-正则化归一化)
  * [数据文件](#数据文件)
  * [数据导入](#数据导入)
  * [保存向量 ID](#保存向量-id)
- [附录](#附录)
  * [计算向量相似度](#计算向量相似度)
    + [内积（点积）](#内积点积)
    + [余弦相似度](#余弦相似度)
    + [欧氏距离](#欧氏距离)

# 硬件推荐配置

| 系统组件           | 建议配置                                                    |
| ------------------ | ----------------------------------------------------------- |
| 操作系统           | Centos 7.2 及以上，或 Ubuntu LTS 16.04 及以上               |
| Docker 版本        | 18 及以上                                                   |
| Nvidia Docker 版本 | 2.0.3                                                       |
| Python 版本        | 3.6 及以上                                                  |
| CPU 配置           | Intel CPU                                                   |
| GPU 驱动           | Nvidia 显卡（Pascal 架构或以上），CUDA 10.1，Driver: 418.74 |
| 内存型号           | 8 GB 内存以上                                               |
| 存储型号           | 最少 SATA 3.0 SSD，推荐 PCIE NVMe SSD                       |


# 测试数据准备

## L2 正则化（归一化）
开始之前，建议先对测试数据进行归一化处理。

假设 n 维原始向量空间：<img src="http://latex.codecogs.com/gif.latex?\\R^n(n>0)" title="\\R^n(n>0)" />

原始向量：<img src="http://latex.codecogs.com/gif.latex?\\X&space;=&space;(x_1,&space;x_2,&space;...,&space;x_n),X&space;\in&space;\reals^n" title="\\X = (x_1, x_2, ..., x_n),X \in \reals^n" />

向量<img src="http://latex.codecogs.com/gif.latex?$$X$$" title="$$X$$" />的 L2 范数（模长）：

<img src="http://latex.codecogs.com/gif.latex?\\\|&space;X&space;\|&space;=&space;\sqrt{\displaystyle\sum_{i=1}^n&space;(x_i)&space;^2}" title="\\| X \| = \sqrt{\displaystyle\sum_{i=1}^n (x_i) ^2}" /><img src="http://latex.codecogs.com/gif.latex?\\\|&space;X&space;\|&space;=&space;\sqrt{\displaystyle\sum_{i=1}^n&space;(x_i)&space;^2}" title="\\\| X \| = \sqrt{\displaystyle\sum_{i=1}^n (x_i) ^2}" />

归一化后的向量：<img src="http://latex.codecogs.com/gif.latex?X'&space;=&space;(x_1',&space;x_2',&space;...,&space;x_n'),X'&space;\in&space;\reals^n" title="X' = (x_1', x_2', ..., x_n'),X' \in \reals^n" />

其中每一维的 L2 正则化算法：

<img src="http://latex.codecogs.com/gif.latex?x_i'&space;=&space;\frac{x_i}{\|&space;X&space;\|}&space;=&space;\frac{x_i}{\sqrt{\displaystyle\sum_{i=1}^n&space;(x_i)&space;^2}}" title="x_i' = \frac{x_i}{\| X \|} = \frac{x_i}{\sqrt{\displaystyle\sum_{i=1}^n (x_i) ^2}}" />

归一化后，向量模长等于 1：<img src="http://latex.codecogs.com/gif.latex?\|&space;X'&space;\|&space;=&space;1" title="\| X' \| = 1" />



## 数据文件

请以 csv 格式生成向量数据，每个文件建议不超过 10 万条向量。为加快数据的导入速度，建议事先为 csv 文件生成相应的 npy 二进制文件：

1. 通过 pandas.read_csv 方法读入一个 csv 文件，生成 pandas.dataframe 数据类型
2. 通过 numpy.array 方法，将上述 pandas.dataframe 转换成 numpy.array 数据类型
3. 将 numpy.array 数据类型，通过 numpy.save 方法存入一个 npy 二进制文件。

由于 npy 是二进制文件，因此不建议删除原始的 csv 文件，后续需要用到 csv 文件来核对向量查询的结果。

## 数据导入

目前 Milvus 数据库提供了一个 Python 客户端。当通过 Python 脚本导入向量数据的时候：

1. 通过 pandas.read_csv 方法读入一个 csv 文件，生成 pandas.dataframe 数据类型
2. 通过 numpy.array 方法，将上述 pandas.dataframe 转换成 numpy.array 数据类型
3. 通过 numpy.array.tolist 方法将 numpy.array 数据导转换成 2 维列表（形如，[[],[]...[]]）。
4. 通过 Milvus 提供的 Python API 将 2 维列表数据导入 Milvus 数据库，同时返回**向量 ID 列表**。

**Note**：

- 如果想验证查询精度，请自行生成每个查询向量的 ground truth 结果，以供后续核对。

## 保存向量 ID

在后续向量查询的时候，为降低内存占用，Milvus 只返回向量 ID。Milvus 当前版本暂时没有保存原始向量数据，因此需要用户自行保存向量数据和所返回的向量 ID。



# 附录
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

