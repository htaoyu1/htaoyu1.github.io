---
layout: post
title: 数据挖掘之关联规则分析（6）：关联规则的评估（二）
category: Data Mining 
author: Hongtao Yu
tags: 
  - data-mining 
  - association-analysis
comments: true
use_math: true
lang: zh
---

- TOC
{:toc}


# 客观度量的性质

在[关联规则的评估（二）](https://htaoyu1.github.io/data%20mining/2017/12/27/Association-Rule-5/)中，我们介绍了一系列评估项集和关联规则的客观兴趣度度量。但是对一组关联规则进行排序时，使用不同的度量我们往往会得到不同的排序结果。比如考虑两个不同的规则，当使用度量一评估时，规则一可能显得比规则二重要；但是使用度量二评估时，结果可能正好相反。那么对于这两个评估，我们更应该相信哪一个呢？要回答这个问题，我们需要了解不同度量的特点，然后根据待分析问题的性质来决定哪个度量更合适。

## 好度量的三个标准

Piatetsky-Shapiro 建议，一个好的度量 $O(A, B)$ 应该满足以下三个性质：

- **P1**: 如果 $A$ 和 $B$ 统计独立，应该有 $O = 0$。

- **P2**: 当 $P(A)$ 、$P(B)$ 不变时，度量 $O$ 的值应随着 $P(AB)$ 的升高单调增加。

   考虑列联表： 
   
   $$
   \mathbf{M} = 
   \begin{bmatrix}
        f_{11} & f_{10}  \\
        f_{01} & f_{00} 
   \end{bmatrix}
   \label{eq:p2m}
   $$
   
   如果
   
   $$
   \mathbf{M'} = \mathbf{M} +  
   \begin{bmatrix}
        k & -k  \\
        -k & k 
   \end{bmatrix}
   $$
   
   依据该性质应该有 $O(\mathbf{M'}) > O(\mathbf{M})$
   
   > 因为此时 $P(A) = (f_{11} + f_{10})/N $；$P(B) = (f_{11} + f_{01})/N$ 均不变，而 $P(AB) = (f_{11} + k)/N$ 变大了。


- **P3**: 当参数（$P(AB)$ 以及  $P(A)$ 或者 $P(B)$）不变时，度量 $O$ 的值应随着 $P(B)$ 或者 $P(A)$ 的升高单调减小。即如果：

   $$
   \mathbf{M'} = \mathbf{M} +  
   \begin{bmatrix}
        0 & k  \\
        0 & -k 
   \end{bmatrix}
   $$

   或者 
   
   $$
   \mathbf{M''} = \mathbf{M} +  
   \begin{bmatrix}
        0 & 0  \\
        k & -k 
   \end{bmatrix}
   $$
   
   应该有 $O(\mathbf{M'}) < O(\mathbf{M})$ 或者 $O(\mathbf{M''}) < O(\mathbf{M})$。
   
   > 在 $\mathbf{M'}$ 中，$P(A)$ 从 $(f_{11} + f_{10})/N $ 增加到 $(f_{11} + f_{10} + k)/N $; 而 $P(B)$ 和 $P(AB)$ 均不变。在 $\mathbf{M''}$ 中，$P(B)$ 从 $(f_{11} + f_{01})/N $ 增加到 $(f_{11} + f_{01} + k)/N $; 而 $P(A)$ 和 $P(AB)$ 保持不变。

## 变量交换对称性（Symmetry under variable permutation））

我们之前已经讨论过对称性度量。所谓对称性，表示度量在交换 $A$ 和 $B$ 的情况下保持不变。如果用数学语言描述，一个度量是**对称的**，如果有 $O(\mathbf{M}^T) = O(\mathbf{M})$。


## 行／列缩放不变性（Row/column scaling invariance）

考虑矩阵

 $$
   \mathbf{R} = \mathbf{C} =  
   \begin{bmatrix}
        k_1 & 0  \\
        0 & k_2 
   \end{bmatrix}
   $$
   
其中 $k_1 $ 和 $k_2$ 为正的常数。乘积 $\mathbf{R} \times \mathbf{M}$ 表示对矩阵 $\mathbf{M}$ 的第一和第二行分别使用 $k_1$ 和 $k_2$ 缩放。乘积 $\mathbf{M} \times \mathbf{C}$ 表示对矩阵 $\mathbf{M}$ 的第一和第二列分别使用 $k_1$ 和 $k_2$ 缩放。我们称度量 $O$ 具有**行列缩放不变性**，如果有 $O(\mathbf{RM}) = O(\mathbf{M})$ 和 $O(\mathbf{MC}) = O(\mathbf{M})$。

![6-1 成绩和性别的例子](/assets/blog-images/Association-Rule-6.1.png)
**Fig. 6-1.** 成绩和性别的例子。

考虑 **Fig. 6-1** 的例子。表中的数据显示 2004 年男生和女生的数量比 1993 年分别翻了两倍和三倍。但是成绩好和成绩差的男生的比例仍然是 3:4, 而女生成绩好坏的比例也保持不变，仍然是 2:1。因此虽然抽样分布发生了变化，但是性别和成绩之间的关联预期保持不变。 


## 行／列交换反对称性（Antisymmetry under row/column permutation）

令

 $$
   \mathbf{S}  =  
   \begin{bmatrix}
        0 & 1  \\
        1 & 0 
   \end{bmatrix}
   $$

对于*归一化*的度量 $O$, 如果有 $O(\mathbf{SM}) = -O(\mathbf{M})$, 则称 $O$ 是**行交换反对称**的；如果 $O(\mathbf{MS}) = -O(\mathbf{M})$, 则称 $O$ 是**列交换反对称**的。  
   

> 如果 $O$ 的取值范围为 $[-1, 1]$, 则称度量 $O$ 是归一化的。对于一个 $O \in [0, \infty)$ 的非归一化度量，可以使用 $(O-1)/(O+1)$ 或者 $\frac{2}{\pi}(\tan ^ {-1} \log(O))$ 等函数进行归一化。

 

## 反演不变性（Inversion invariance）

仍然令

 $$
   \mathbf{S}  =  
   \begin{bmatrix}
        0 & 1  \\
        1 & 0 
   \end{bmatrix}
   $$

如果有 $O(\mathbf{SMS}) = O(\mathbf{M})$, 则称 $O$ 具有**反演不变性**。  



![6-2 反演操作的影响](/assets/blog-images/Association-Rule-6.2.png)
**Fig. 6-2.** 反演操作的影响。矢量 $C$ 和 $D$ 分别是 $A$ 和 $B$ 的反演。 


考虑 **Fig. 6-2** 的例子，每个列向量表示一个项集是否在事务集中出现。比如对于 $A$, 只有第一个元素和最后一个元素为1，表示 $A$ 只在第一个和最后一个事务中出现。向量 $C$ 是 $A$ 的`反演`（inversion），即 $A$ 中的值 1 在 $C$ 中被反转为 0，0 被 反转为 1。所谓反演不变性就是有$O(A, B) = O(C, D)$。反演不变性度量不适合分析`对称的二元数据`。

> 对称二元数据是指一个变量取值为 0 和 1 的重要性是一样的。

## 零加不变性（Null invariance）

`零加`（null addition）操作是指在事务集中添加不相关的事务。比如在评估规则 $A \Rightarrow B$ 时，我们向事务集添加不包含  $A$ 和 $B$ 的事务。**零加不变性**是指度量在零加操作下是不变的。如果用数学的语言描述，我们令

 $$
   \mathbf{C}  =  
   \begin{bmatrix}
        0 & 0  \\
        0 & k 
   \end{bmatrix}
   $$
 
 对于一个零加不变的度量有 $O(\mathbf{M} + \mathbf{C}) = O(\mathbf{M})$。

下图总结了上面我们对列联表的五种操作结果。

![6-3 对列联表的五种操作结果](/assets/blog-images/Association-Rule-6.3.png)
**Fig. 6-3.** 对列联表的五种操作结果。（a）变量交换操作。（b）行列缩放操作。（c）行列交换操作。（d）反演操作。（e）零加操作。（图片来自于参考文献 2）。 


下图是对客观度量性质的一个总结，其中没有一个度量可以满足前面我们讨论的所有性质。

![6-4 客观度量的性质总结](/assets/blog-images/Association-Rule-6.4.png)
**Fig. 6-4.** 客观度量的性质总结（图片来自于参考文献 2）。 


# 辛普森悖论与数据分层

现实中的数据和关联规则是复杂的。例如，根据特定变量的值，一个关联规则可能出现，也可能不出现，甚至出现反转。这个问题就是`辛普森悖论`（Simpson's paradox）。**Fig. 6-5** 显示了 HDTV 电视和健身器材销售的列联表。

![6-5 高清电视和健身器材销售的列联表](/assets/blog-images/Association-Rule-6.5.png)
**Fig. 6-5.** 高清电视和健身器材销售的列联表。 

从数据中我们可以得到下列关联规则的置信度：

$$
c(买 \text{HDTV} = 是 \Rightarrow 买健身器材 = 是) = \frac{99}{180} = 55\%
$$

$$
c(买 \text{HDTV} = 否 \Rightarrow 买健身器材 = 是) = \frac{99}{180} = 45\%
$$

规则暗示，与没有买高清电视的顾客相比，购买电视的顾客更有可能购买健身器材。然而，进一步分析发现，购买这些商品的顾客分为两类，一类是大学生，一类是在职人员。**Fig. 6-6** 分别给出了这两类人员的列联表。

![6-6 大学生和在职人员购买高清电视和健身器材的列联表](/assets/blog-images/Association-Rule-6.6.png)
**Fig. 6-6.** 大学生和在职人员购买高清电视和健身器材的列联表。 

根据上述数据，我们可以分别计算出这两类人员购买电视机和健身器材的关联规则。对于大学生，我们有：

$$
c(买 \text{HDTV} = 是 \Rightarrow 买健身器材 = 是) = \frac{1}{10} = 10.0\%
$$

$$
c(买 \text{HDTV} = 否 \Rightarrow 买健身器材 = 是) = \frac{4}{34} = 11.8\%
$$

对于在职人员有：

$$
c(买 \text{HDTV} = 是 \Rightarrow 买健身器材 = 是) = \frac{98}{170} = 57.7\%
$$

$$
c(买 \text{HDTV} = 否 \Rightarrow 买健身器材 = 是) = \frac{50}{86} = 58.1\%
$$

这里我们得到了与刚才完全相反的结论！不论是大学生，还是在职人员，都是没有购买电视机的更有可能购买健身器材。

这里的悖论可以进行如下解释。假设 

$$ 
\frac{a}{b} < \frac{c}{d} \qquad 并且 \qquad \frac{p}{q} < \frac{r}{s}
$$

其中 $a/b$ 和 $p/q$ 是规则 $A \Rightarrow B$ 在分层数据下的置信度，$c/d$ 和 $r/s$ 是规则 $\overline{A} \Rightarrow B$ 在分层数据下的置信度。当数据合并到一起时，如果有

$$
\frac{a+p}{b+q} > \frac{c+r}{d+s}
$$

那么悖论就出现了。所以这里的教训是，<u>需要适当的数据分层才能避免因辛普森悖论产生的虚假模式</u>。比如在分析病人的数据时，首先按照混杂因素（如年龄，性别等）对数据分层。


# 倾斜支持度分布


具有`倾斜支持度分布`的数据集对关联分析算法提取模式的质量提出了挑战。一个事务集中大多数项具有较低或者中等的频率，但是有少数项具有极高的频率，这样的数据集叫做具有倾斜支持度分布的数据集。下图展示了这样的一个数据集。

![6-7 一个具有倾斜支持度分布的数据集](/assets/blog-images/Association-Rule-6.7.png)
**Fig. 6-7.** 一个具有倾斜支持度分布的数据集。 


在 Apriori 算法 和 FP-Growth 算法中，我们使用支持度和置信度来对数据集进行挖掘。然而，如果使用较大的置信度，可能会遗漏数据集中的一些重要规则。比如， **Fig 6-7** 的数据集包含了 30 条事务，其中 $q$ 和 $r$ 的支持度均为 $16.7\%$。图中我们可以看到 $q$ 和 $r$ 具有很强的关联性。但是如果是用 $20\%$ 的支持度阈值，我们就会错失这条重要规则。采用较低的支持度阈值可以克服这个问题。但是对于一个较大的数据集，如果是用更小的支持度，那么需要处理的规则将会成倍的增加。

降低置信阈值的另一个副作用是会产生大量的`交叉支持`（cross-support）模式。所谓交叉支持模式，就是算法会挖掘出大量的高频项与低频项相关联的虚假模式。比如在 **Fig 6-7** 中 $q \Rightarrow p$ 就是一个交叉支持模式。就算使用置信度度量也不足以消除交叉支持模式，因为 $c(q \Rightarrow p) = 80\%$。在进行进一步分析之前，我们先给交叉支持模式下一个定义：

**交叉支持模式**： 交叉支持模式是一个项集 $X = $ {$i_1, i_2, \cdots, i_k$} 的支持度比率

$$
r(X) = \frac{\min [s(i_1), s(i_2), \cdots, s(i_k)]}{\max [s(i_1), s(i_2), \cdots, s(i_k)]}
\label{eq:sr}
$$

小于用户指定的阈值 $h_c$。

假定我们使用 $h_c = 0.3$, 那么上面数据集中的 {$p, q$}，{$p, r$}，以及 {$p, q, r$} 都是交叉支持模式，因为他们的支持度比率为 $0.2$。


在上面的例子中我们注意到，虽然 $c(q \Rightarrow p) $ 的支持度很高，但是规则 $c(p \Rightarrow q) $ 的支持度却很低，只有 $4/25 = 16\%$。这就意味着，我们可以使用项集的**最低置信度规则来检测交叉支持模式**。


因为置信度具有如下反单调性

$$
c(\{i_1, i_2\} \Rightarrow \{i_3, i_4, \cdots, i_k\}) \leq c(\{i_1, i_2, i_3\} \Rightarrow \{i_4, i_5, \cdots, i_k\})
$$

即将关联规则左边的项移到右边不会增加规则的置信度。所以我们可以通过寻找左边仅包含一个项的规则来寻找最低置信度。因为置信度

$$
c(\{i_j\} \Rightarrow \{ i_1, i_2, \cdots, i_{j-1}, i_{j+1}, \cdots, i_k\}) = \frac{s(i_1, i_2, \cdots, i_k)}{s(i_j)}
$$

不管第一项是什么，只要项集给定，上式的分子是不变的。为了最小化置信度，我们需要最大化分母，即寻找项 $i_m$， 满足

$$
s(i_m) = \max [s(i_1), s(i_2), \cdots, s(i_k)]
$$

则规则

$$
c(\{i_m\} \Rightarrow \{ i_1, i_2, \cdots, i_{m-1}, i_{m+1}, \cdots, i_k\}) = \frac{s(i_1, i_2, \cdots, i_k)}{s(i_m)}
$$

就是具有最小置信度的规则。这个度量就是之前我们讲到的全置信度度量

$$
\begin{eqnarray}
\text{all_conf}(\{ i_1, i_2, \cdots, i_k \}) & = & \frac{s(\{i_1, i_2, \cdots, i_k\})}{\max [s(i_1), s(i_2), \cdots, s(i_k)]} \\
& \leq & \frac{\min [s(i_1), s(i_2), \cdots, s(i_k)]}{\max [s(i_1), s(i_2), \cdots, s(i_k)]}
\end{eqnarray}
$$

上式中的第二步我们用到了支持度的反单调性。因此， 全置信度的上界其实就是公式 $\ref{eq:sr}$ 中的支持度比率。因为交叉支持模式的支持度比率总是小于 $h_c$, 所以这类模式的全置信度也一定小于 $h_c$。因此通过确保模式的全置信度超过 $h_c$ 就可以消除交叉支持模式。另外，全置信度也是反单调的，即

$$
\text{all_conf}(\{ i_1, i_2, \cdots, i_k \}) \geq \text{all_conf}(\{ i_1, i_2, \cdots, i_k, i_{k+1} \})
$$

全置信度能够确保项集中的项之间是强关联的。假定一个项集 $X$ 的全置信度是 $80\%$。那么如果 $X$ 中的一个项出现在某个事务中，那么 $X$ 中的其他项至少也有 $80\%$ 的概率出现在该事务中。这种强关联模式又称`超团模式`（hyperclique pattern）。

# References


1. [数据挖掘导论](https://book.douban.com/subject/5377669/)，P.-N. Tan, Michael Steinbach, and V. Kumar 著；范明，范宏建 译， 人民邮电出版社, **2006.**

2. P.-N. Tan, V. Kumar, and J. Srivastava, "[Selecting the right objective measure for association analysis,](https://doi.org/10.1016/S0306-4379(03)00072-3)" *Information Systems*, vol. 29, pp. 293-313, **2004.**

2. [数据挖掘：概念与技术](https://book.douban.com/subject/2038599/)，Jiawei Han, Micheline Kamber, and Jian Pei 著；范明，孟小峰 译， 机械工业出版社, **2007.**

3. [Apriori导论](http://www.voidcn.com/article/p-whbfxrod-vp.html)


4. [A Probabilistic Comparison of Commonly Used Interest Measures for Association Rules](http://michael.hahsler.net/research/association_rules/measures.html)









