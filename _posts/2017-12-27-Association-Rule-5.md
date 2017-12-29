---
layout: post
title: 数据挖掘之关联规则分析（5）：关联规则的评估（一）
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



# 引言

关联分析可以产生大量的规则，但是在实际的应用场景中，很大一部分可能是我们不感兴趣的。例如在购物篮数据中，我们认为规则 {黄油} $\Rightarrow$ {面包} 是显而易见并且无趣的。在产生规则的过程中，我们使用了支持度和置信度的度量。支持度的一个缺点是`稀有项问题`(rare item problem)， 即如果一个项出现的次数非常少，那么它就会被立即剪去。然而在实际生活中，这些稀有的项仍然是有可能产生一些有意思并且有价值的规则的。例如在购物篮数据中，购买珠宝的事务可能是稀有项，但是我们仍然对能否从中挖掘出一些有意义的规则感兴趣。

如何从大量的规则中筛选出有意义的模式呢？第一种方法是使用统计学的`客观兴趣度度量`（objective interesting measure）来评估模式的质量。例如之前我们讨论的支持度，置信度，以及提升度。第二种方式是结合领域内的专家知识，使用可视化技术（visualization）、基于模版的方法（template-based approach）、或者基于领域知识的主观兴趣度度量（subjective interesting measure）来评估。这里我们只讨论不依赖于领域专业知识的、数据驱动的客观度量方法。

# 列联表

`列联表`（contingency table, 又译相依表）是统计多个类别变量观测值的表格。如果我们用 $A$ 表示事件 $A$ 发生，$\overline{A}$ 表示 $A$ 没有发生。那么对于两个变量 $A$ 和  $B$, 其联列表可以表示为:

|       |   $B$ | $\overline{B}$ |     |
|:-----:|:------:|:-------------:|:----|
| $A$  	| $f_{11}$ | $f_{10}$ | $f_{1+}$|
| $\overline{A}$  	| $f_{01}$ | $f_{00}$ | $f_{0+}$ |
|      	| $f_{+1}$ | $f_{+0}$ | $N$ |

其中 $f_{11}$ 表示数据集中事件 $A$ 和 $B$ 同时发生的次数；$f_{10}$ 表示 $A$ 发生 $B$ 没发生的事件次数；$f_{1+} = f_{11} +  f_{10}$ 表示 $A$ 发生的总次数，$f_{0+} = f_{01} +  f_{00}$ 表示 $A$ 没发生的事件总次数。$N = f_{1+} + f_{0+} = f_{+1} + f_{+0}$ 表示数据集中包含的事件总数。 客观度量常常基于联列表中列出的频度计数来计算。



# 兴趣因子

`兴趣因子`（Interest Factor）其实就是我们[之前](http://localhost:4000/data%20mining/2017/12/11/Association-Rule-1/#%E6%8F%90%E5%8D%87%E5%BA%A6)介绍的提升度最初的名字，这里不再赘述。兴趣因子的取值范围为 $[0, \infty)$。

兴趣因子是一种`对称度量`： 

$$
I(A \Rightarrow B) = I(B \Rightarrow A) = \frac{P(AB)}{P(A) \cdot P(B)} = \frac{N f_{11}}{f_{1+} f_{+1}}
$$


兴趣因子（提升度）不存在稀有项问题，但是它**对数据集中的噪声敏感**，两个稀有项同时出现几次就有可能产生很大的兴趣因子。考虑下面两个联列表：

|       |   $p$ | $\overline{p}$ |     |
|:-----:|:------:|:-------------:|:----|
| $q$  	| 880 | 50 | 930 |
| $\overline{q}$  	| 50 | 20 | 70  |
|      	| 930 | 70 | 1000 |


|       |   $r$ | $\overline{r}$ |     |
|:-----:|:------:|:-------------:|:----|
| $s$  	| 20 | 50 | 70 |
| $\overline{s}$  	| 50 | 880 | 930  |
|      	| 70 | 930 | 1000 |

**Table 5-1**: {p, q} 和 {r, s}的列联表。

在 **Table 5-1** 中，$p$ 和 $q$ 同时出现和不出现的次数与 $r$ 和 $s$ 正好相反。但是 {$r$, $s$} 的兴趣因子远大于 {$p$, $q$}:

$$
I(p \Rightarrow q) = I(q \Rightarrow p)= \frac{0.88}{0.93 \times 0.93} = 0.12
$$ 

$$
I(r \Rightarrow s) = I(s \Rightarrow r)= \frac{0.02}{0.07 \times 0.07} = 4.08
$$ 

可以看到尽管 {$p$, $q$} 在 1000 条事务中同时出现了 880 次， 其兴趣因子只有 1.02； 而 {$r$, $s$} 只同时出现了 20 次，兴趣因子竟然达到了 4.08。这里的原因是 $r$ 和 $s$ 都是稀有项，发生的概率只有 0.07, 虽然两者一块发生的概率只有 0.02, 但是由于分母非常小，导致了兴趣因子的值非常大。在这样的情况下，置信度是一个更好的衡量标准。


# 相关分析

相关分析是统计两个变量之间关系的一种方法。对于连续变量，相关度用`皮尔逊相关系数`定义，对于二元变量，相关度可以用 $\phi$ `系数度量`：

$$
\begin{eqnarray}
\phi(A,B) 
& = & \frac{P(AB)-P(A)P(B)}{\sqrt{P(A)P(B)(1-P(A))(1-P(B))}} \\
& = & \frac{N f_{11} - f_{1+} f_{+1}}{\sqrt{f_{1+} f_{+1} f_{0+} f_{+0}}} \\
& = &
\frac{ f_{11}  f_{00} - f_{10} f_{01}
}{\sqrt{f_{1+} f_{+1} f_{0+} f_{+0} }}
\end{eqnarray}
$$


> $$
> \begin{eqnarray*}
> \frac{N f_{11} - f_{1+} f_{+1}}{\sqrt{f_{1+} f_{+1} f_{0+} f_{+0}}} 
> 
> & = & \frac{N f_{11} - (f_{11} + f_{10}) (f_{11} + f_{01})}{\sqrt{f_{1+} f_{+1} f_{0+} f_{+0}}} \\
> 
> & = & \frac{(N - f_{11} - f_{10} - f_{01}) f_{11} - f_{10} f_{01}}{\sqrt{f_{1+} f_{+1} f_{0+} f_{+0}}} \\
> 
> & = & \frac{ f_{11}  f_{00} - f_{10} f_{01}
}{\sqrt{f_{1+} f_{+1} f_{0+} f_{+0} }}
> \end{eqnarray*}
> $$

$\phi$ 相关系数的涵义是：两个二元变数 $A$ 和 $B$, 若观察值大多落在列联表的对角线上，即 $A$ 和 $B$ 要么同时发生，要么同时不发生，那么 $A$ 和 $B$ 正相关；反之，那么 $A$ 和 $B$ 负相关。$\phi$ 相关系数的取值范围为 $[-1, 1]$。$\phi$ 相关系数与`皮尔逊卡方检验`密切相关 $\phi^2 = \chi^2/N$。

$\phi$ 系数把项在事务中同时出现和同时不出现视为同等重要，比如在 **Table 5-1** 中, $\phi(p, q) = \phi (r, s) = 0.232$。因此 $\phi$ 系数更适合分析对称的二元变量。


# $\chi^2$ 分析（卡方分析）

因为 $\phi$ 系数度量与卡方检验密切相关，这里我们就顺便介绍一下 $\chi^2$ 检验。卡房系数可以帮组我们确定两个变量是否相关，定义为:

$$
\chi^2 = \sum\limits_x \frac{(x - \overline{x})^2}{\overline{x}}
$$

上式中 $x$ 为一个观测值，$\overline{x}$ 为 $x$ 的期望值，求和表示对所有的观测求和。以 **Table 5-2** 中的 {g, v} 数据为例：

|       |   $g$ | $\overline{g}$ |     |
|:-----:|:------:|:-------------:|:----|
| $v$  	| 400(450) | 350(300) | 750 |
| $\overline{v}$  	| 200(150) | 50(100) | 250  |
|      	| 600 | 400 | 1000 |

**Table 5-2**: {g, v} 的列联表

假设 $g$ 和 $v$ 独立，那么 $g$ 发生的概率为 $750/1000 = 0.75$, 所以在 $v$ 发生的 600 次里，同时包含 $g$ 的事务计数应该为 $600 \times 0.75 = 450$, 即 $f_{gv}$ 的期望值为 450。同理，我们可以求出表中其他的频度计数(见括号中的数据)。因此 $g$ 和 $v$ 的卡房系数为：

$$
\chi^2 = \frac{(400-450)^2}{450} + \frac{(350-300)^2}{300} + \frac{(200-150)^2}{150} + \frac{(50-100)^2}{100} = 555.6
$$

卡方系数需要查表才能确定值的意义。由于数据的自由度为 = (行数-1) $\times$ (列数-1)=1，在置信度为(1-0.001)的条件下，查表得到值为6.63。由于 555.6 远大于该值，所以我们拒绝 $g$ 和 $v$ 相互独立的假设，因为这两者是有关联的。由于 $f_{gv}$ 的观测值 400 小于独立假设下的期望值 $450$, 所以认为 $g$、$v$ 负相关。


# IS 度量

`IS 度量`综合考虑了兴趣因子和支持度。 IS 度量定义为兴趣因子和支持度的几何平均值：

$$
IS(A,B) = \sqrt{I(A, B) \times s(A, B)} = \frac{P(AB)}{\sqrt{P(A)P(B)}}
$$

当模式的兴趣因子和支持度都很大时，IS 也很大。IS 的取值范围为 $[0, 1]$。比如在 **Table 5-1** 中, $IS(p, q) = 0.946$，$IS(r, s) = 0.286$。

IS 度量在数学上等价于余弦相似度。将 $A$ 和 $B$ 看成一对位向量。例如对于长度为 $N$ 的事务集 $T$ = {$t_1, t_2, \cdots, t_N$}, 如果 $A$ 在事务 $t_i$ 中出现，那么记 $a_i = 1$, 否则 $a_i = 0$。这样 $A$ 可以用矢量来表示 $\mathbf{A}$ = {$a_1, a_2, \cdots, a_N$}。项集 {A, B} 的支持度计数为 $\sigma(A \cup B) = \mathbf{A} \cdot \mathbf{B}$, 支持度为 $s(A, B) = \frac{\mathbf{A} \cdot \mathbf{B}}{N}$。 A 的支持度为 $s(A) = \frac{\sigma(A)}{N} = \frac{\mathbf{A} \cdot \mathbf{A}}{N} = \frac{\vert \mathbf{A}\vert^2}{N}$。我们有

$$
\cos(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\vert \mathbf{A}\vert \vert \mathbf{B}\vert} = \frac{s(A, B)}{\sqrt{s(A)\times s(B)}}
$$


IS 度量也可以表示为从一对二元变量中提取出的关联规则置信度的几何平均值：

$$
IS(A, B) = \sqrt{\frac{s(A, B)}{s(A)} \times \frac{s(A, B)}{s(B)}} = \sqrt{c(A \Rightarrow B) \times c(B \Rightarrow A)}
$$

由于集合均值总是接近于较小的数，所以只要规则 $A \Rightarrow B$ 和 $B \Rightarrow A$ 中有一个的置信度较低，那么项集 {$A, B$} 的 IS 值就较低。从上面的关系我们也可以看出，IS 是一个对称度量。

IS 度量与置信度度量存在类似的缺陷，**即使不相关或则负相关的模式，其 IS 值也可能相当大**。考虑统计独立的两个项 $A$ 和 $B$, 我们有:

$$
IS_{indep}(A, B) = \frac{s(A, B)}{\sqrt{s(A)\times s(B)}} = \frac{s(A)\times s( B)}{\sqrt{s(A)\times s(B)}} = \sqrt{S(A) \times S(B)}
$$

即统计独立情况下，如果 $A$ 和 $B$ 各自的支持度都很大的话，那么他们的 IS 度量也会很大。


# 全置信度

给定两个项集 $A$ 和 $B$, 他们的`全置信度`（all-confidence, 又称 `h置信度`, h-confidence）定义为：

$$
\text{all_conf}(A, B) = \frac{s(A \cup B)}{\max \{s(A), s(B)\}}  = \min \{ P(A \vert B), P(B \vert A) \}
$$

可以看出 $A$ 和 $B$ 的全置信度等价于关联规则 $A \Rightarrow B$ 和  $B \Rightarrow A$ 的最小置信度。

# 最大置信度

`最大自信度`与全自信度相反，求的是规则 $A \Rightarrow B$ 和  $B \Rightarrow A$ 的最大置信度：

$$
\text{max_conf}(A, B) = = \max \{ P(A \vert B), P(B \vert A) \}
$$

# Kulczynski (Kluc) 度量

`Kulczynski 度量` 为规则 $A \Rightarrow B$ 和  $B \Rightarrow A$ 的置信度的算术平均值：

$$
\text{Kulc}(A, B) = = \frac{1}{2} ( P(A \vert B) + P(B \vert A) )
$$

# 客观兴趣度度量总结

除了上面提到的几种度量，还有很多其他的度量客观兴趣度的方法。参考文献 4 罗列了很多兴趣度度量的方法以及相应的参考文献。下图是对一部分客观兴趣度度量方法在概率意义上的一个总结：

![5-1 关联模式评估的客观兴趣度度量总结](/assets/blog-images/Association-Rule-5.1.png)
**Fig. 5-1.** 关联模式评估的客观兴趣度度量总结（图片来自于参考文献 5）。

度量可以是对称的，也可以是非对称的。对于一个度量 $M$, 如果有 $M(A \Rightarrow B) = M (B \Rightarrow A)$，那么就称 $M$ 为`对称的度量`，否则就是`非对称的度量`。**对称度量一般用来评价项集，非对称度量适合用来分析关联规则。** 下图是使用列联表中的频度计数表示的一部分度量的计算公式：

![5-2 关联模式评估的客观兴趣度度量的对称性](/assets/blog-images/Association-Rule-5.2.png)
**Fig. 5-2.** 关联模式评估的客观兴趣度度量的对称性（原图来自于参考文献 6）。



# References


1. [数据挖掘导论](https://book.douban.com/subject/5377669/)，P.-N. Tan, Michael Steinbach, and V. Kumar 著；范明，范宏建 译， 人民邮电出版社, **2006.**

2. [数据挖掘：概念与技术](https://book.douban.com/subject/2038599/)，Jiawei Han, Micheline Kamber, and Jian Pei 著；范明，孟小峰 译， 机械工业出版社, **2007.**

3. [Apriori导论](http://www.voidcn.com/article/p-whbfxrod-vp.html)


4. [A Probabilistic Comparison of Commonly Used Interest Measures for Association Rules](http://michael.hahsler.net/research/association_rules/measures.html)

5. P.-N. Tan, V. Kumar, and J. Srivastava, "[Selecting the right objective measure for association analysis,](https://doi.org/10.1016/S0306-4379(03)00072-3)" *Information Systems*, vol. 29, pp. 293-313, **2004.**

6. M. Steinbach, P.-N. Tan, H. Xiong, and V. Kumar, "[Objective Measures for Association Pattern Analysis,](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.76.4177&rep=rep1&type=pdf)" in *AMS-IMS-SIAM Joint Summer Research Conference, Machine and Statistical Learning, Prediction and Discovery,* Snowbird, Utah, **2006.**







