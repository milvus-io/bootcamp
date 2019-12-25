# Data Normalization

- [L2 Normalization](#L2Normalization)
- [Compute Vector Similarity](#ComputeVectorSimilarity)
  - [Inner Product (Dot Product)](#InnerProduct)
  - [Cosine Simialrity](#CosineSimilarity)
  - [Euclidean Distance](#EuclideanDistance)

## L2 Normalization

Before you begin, it is recommended to normalize the data used in the test.

Assume there is an initial vector with n dimensions: <img src="http://latex.codecogs.com/gif.latex?\\R^n(n>0)" title="\\R^n(n>0)" />

Initial vector: <img src="http://latex.codecogs.com/gif.latex?\\X&space;=&space;(x_1,&space;x_2,&space;...,&space;x_n),X&space;\in&space;\reals^n" title="\\X = (x_1, x_2, ..., x_n),X \in \reals^n" />

The L2 norm of vector <img src="http://latex.codecogs.com/gif.latex?$$X$$" title="$$X$$" />is represented as:

<img src="http://latex.codecogs.com/gif.latex?\\\|&space;X&space;\|&space;=&space;\sqrt{\displaystyle\sum_{i=1}^n&space;(x_i)&space;^2}" title="\\| X \| = \sqrt{\displaystyle\sum_{i=1}^n (x_i) ^2}" />

Normalized vector:<img src="http://latex.codecogs.com/gif.latex?X'&space;=&space;(x_1',&space;x_2',&space;...,&space;x_n'),X'&space;\in&space;\reals^n" title="X' = (x_1', x_2', ..., x_n'),X' \in \reals^n" />

Each dimension is calculated as follows：

<img src="http://latex.codecogs.com/gif.latex?x_i'&space;=&space;\frac{x_i}{\|&space;X&space;\|}&space;=&space;\frac{x_i}{\sqrt{\displaystyle\sum_{i=1}^n&space;(x_i)&space;^2}}" title="x_i' = \frac{x_i}{\| X \|} = \frac{x_i}{\sqrt{\displaystyle\sum_{i=1}^n (x_i) ^2}}" />

After normalization, the L2 norm is 1: <img src="http://latex.codecogs.com/gif.latex?\|&space;X'&space;\|&space;=&space;1" title="\| X' \| = 1" />



## Compute Vector Similarity

ANNS (approximate nearest neighbor searching) is the mainstream application in vector search. The key concept is that computing and searching is done only in the sub-spaces of initial vector space. This significantly increases the overall search speed.

Assume the search space (sub-spaces of initial vector space) is: <img src="http://latex.codecogs.com/gif.latex?\gamma,&space;\gamma&space;\subset&space;R^n" title="\gamma, \gamma \subset R^n" />



### Inner Product (Dot Product)

The dot product of two vectors <img src="http://latex.codecogs.com/gif.latex?$$A,&space;B$$" title="$$A, B$$" /> is defined as follows：

<img src="http://latex.codecogs.com/gif.latex?$$p(A,B)&space;=&space;A&space;\cdot&space;B&space;=&space;\displaystyle\sum_{i=1}^n&space;a_i&space;\times&space;b_i$$" title="$$p(A,B) = A \cdot B = \displaystyle\sum_{i=1}^n a_i \times b_i$$" />



### Cosine Similarity

The cosine similarity of two vectors <img src="http://latex.codecogs.com/gif.latex?$$A,&space;B$$" title="$$A, B$$" />is represented as: 

<img src="http://latex.codecogs.com/gif.latex?$$\cos&space;(A,B)&space;=&space;\frac{A&space;\cdot&space;B}{\|A&space;\|&space;\|B\|}$$" title="$$\cos (A,B) = \frac{A \cdot B}{\|A \| \|B\|}$$" />

Similarity is measured by the cosine of the angle between two vectors: the greater the cosine, the higher the similarity:

 <img src="http://latex.codecogs.com/gif.latex?$$TopK(A)&space;=&space;\underset{B&space;\in\&space;\gamma}{\operatorname{argmax}}&space;\big&space;(&space;cos(A,B)&space;\big&space;)$$" title="$$TopK(A) = \underset{B \in\ \gamma}{\operatorname{argmax}} \big ( cos(A,B) \big )$$" />

Assume that after vector normalization, original vector<img src="http://latex.codecogs.com/gif.latex?$$A,&space;B$$" title="$$A, B$$" />is converted to <img src="http://latex.codecogs.com/gif.latex?$$A',&space;B'$$" title="$$A, B$$" />：

<img src="http://latex.codecogs.com/gif.latex?$$cos(A,B)&space;=&space;\frac{A&space;\cdot&space;B}{\|A&space;\|&space;\|B\|}&space;=&space;\frac{&space;\displaystyle\sum_{i=1}^n&space;a_i&space;\times&space;b_i}{\|A\|&space;\times&space;\|B\|}&space;=&space;\displaystyle\sum_{i=1}^n&space;\bigg(\frac{a_i}{\|A\|}&space;\times&space;\frac{b_i}{\|B\|}\bigg)=cos(A',B')&space;$$" title="$$cos(A,B) = \frac{A \cdot B}{\|A \| \|B\|} = \frac{ \displaystyle\sum_{i=1}^n a_i \times b_i}{\|A\| \times \|B\|} = \displaystyle\sum_{i=1}^n \bigg(\frac{a_i}{\|A\|} \times \frac{b_i}{\|B\|}\bigg)=cos(A',B') $$" />

Thus, the cosine similarity of two vectors remains unchanged after vector normalization. In particular,

<img src="http://latex.codecogs.com/gif.latex?$$cos(A',B')&space;=&space;\displaystyle\sum_{i=1}^n&space;\bigg(\frac{a_i}{\|A\|}&space;\times&space;\frac{b_i}{\|B\|}\bigg)=p(A',B')$$" title="$$cos(A',B') = \displaystyle\sum_{i=1}^n \bigg(\frac{a_i}{\|A\|} \times \frac{b_i}{\|B\|}\bigg)=p(A',B')$$" />

it can be concluded that **Cosine similarity equals Dot product for normalized vectors**.





### Euclidean Distance

The Euclidean distance of vectors <img src="http://latex.codecogs.com/gif.latex?$$A,&space;B$$" title="$$A, B$$" /> is represented as: 

<img src="http://latex.codecogs.com/gif.latex?$$d(A,B)&space;=&space;\sqrt{\displaystyle\sum_{i=1}^n&space;(a_i-b_i)&space;^2}$$" title="$$d(A,B) = \sqrt{\displaystyle\sum_{i=1}^n (a_i-b_i) ^2}$$" />

Similarity is measured by comparing the Euclidean distance between two vectors: the smaller the Euclidean distance, the higher the similarity:

<img src="http://latex.codecogs.com/gif.latex?$$TopK(A)&space;=&space;\underset{B&space;\in\&space;\gamma}{\operatorname{argmin}}&space;\big&space;(&space;d(A,B)&space;\big&space;)$$" title="$$TopK(A) = \underset{B \in\ \gamma}{\operatorname{argmin}} \big ( d(A,B) \big )$$" />

If you further unfold the above formula, you will get: 

<img src="http://latex.codecogs.com/gif.latex?$$d(A,B)&space;=&space;\sqrt{\displaystyle\sum_{i=1}^n&space;(a_i-b_i)&space;^2}\\\\&space;=\sqrt{\displaystyle\sum_{i=1}^n&space;(a_i^2-2a_i&space;\times&space;b_i&plus;b_i^2)}\\\\&space;=\sqrt{\displaystyle\sum_{i=1}^n&space;a_i^2&plus;\displaystyle\sum_{i=1}^n&space;b_i^2-2\displaystyle\sum_{i=1}^n&space;a_i&space;\times&space;b_i}\\\\&space;=\sqrt{2-2&space;\times&space;p(A,B)}&space;\\\\&space;\therefore&space;d(A,B)^2&space;=&space;-2&space;\times&space;p(A,B)&space;&plus;&space;2$$" title="$$d(A,B) = \sqrt{\displaystyle\sum_{i=1}^n (a_i-b_i) ^2}\\\\ =\sqrt{\displaystyle\sum_{i=1}^n (a_i^2-2a_i \times b_i+b_i^2)}\\\\ =\sqrt{\displaystyle\sum_{i=1}^n a_i^2+\displaystyle\sum_{i=1}^n b_i^2-2\displaystyle\sum_{i=1}^n a_i \times b_i}\\\\ =\sqrt{2-2 \times p(A,B)} \\\\ \therefore d(A,B)^2 = -2 \times p(A,B) + 2$$" />

It is obvious that the square of Euclidean distance has a negative correlation with the dot product. The Euclidean distance is a non-negative real number. And the size relationship between two non-negative real numbers is the same as the size relationship between their own squares.

<img src="http://latex.codecogs.com/gif.latex?\lbrace&space;a,b,c&space;\rbrace&space;\subset&space;\lbrace&space;x&space;\in&space;R&space;|&space;x&space;\geqslant&space;0&space;\rbrace" title="\lbrace a,b,c \rbrace \subset \lbrace x \in R | x \geqslant 0 \rbrace" />

<img src="http://latex.codecogs.com/gif.latex?\because&space;a&space;\leqslant&space;b&space;\leqslant&space;c" title="\because a \leqslant b \leqslant c" />

<img src="http://latex.codecogs.com/gif.latex?\therefore&space;a^2&space;\leqslant&space;b^2&space;\leqslant&space;c^2" title="\therefore a^2 \leqslant b^2 \leqslant c^2" />

Therefore, we can conclude that after vector normalization, if you search the same vector in the same vector spaces, the Euclidean distance equals Dot product for the top k results.
