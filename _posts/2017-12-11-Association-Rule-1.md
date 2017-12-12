---
layout: post
title: 数据挖掘之关联规则分析（1）：基本概念
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


> 之前找工作面试的时候被问到非平衡数据的问题，所以面试回来后恶补了一些非平衡数据集处理的方法。最近工作上需要拿出一些创新的玩意儿，就顺着这个思路走了下去。拿着做学术的劲头看了一些非平衡数据集处理的论文，主要是 SMOTE 类的方法。仔细研究了下业界的一些算法，以及现有算法的一些缺陷，形成了一个很小的 idea。今天把材料整理好提交了出去，所以决定在这个方向上暂告一个段落。接下来的项目需要做一些关联分析的东西，正好边学习边写点笔记。因为还在学习阶段，有些地方表述可能不是很准确，或者有纰漏，欢迎大家批评指正。学习的过程中参考了很多网路上的资料，特别是有些 blog 写的相当好，对我的帮助很大，摘取了其中一些收录到了参考文献里，这里一并表示感谢。

# 引言

关联规则（Association Rules）分析是数据挖掘研究中的一个重要分支，用于从大量数据中挖掘出有价值的事物之间的相互依存性和关联性，如果两个或多个事物之间存在一定的关联性，那么，其中一个事物就能通过其他事物预测到。关联规则分析属于**无监督学习**。IBM 的 Agrawal, Imielinski 和 Swami 于 1993 年在对市场购物篮问题（Market Rule Analysis）的分析中首次提出了该问题，用于发现销售中商品间的关联性。沃尔玛“啤酒与尿布”的故事就是关联规则挖掘的一个经典案例。

> 1998 年的《哈佛商业评论》上刊登了“啤酒与尿布”的案例。沃尔玛的超市管理人员分析销售数据时发现了一个有意思的现象：某些情况下，“啤酒”与“尿布”这两件看上去毫无关系的商品经常会同时出现在同一个购物篮中。后续调查发现，美国妇女们经常会叮嘱丈夫下班后为孩子买尿布，30%-40%的丈夫同时会顺便购买自己喜爱的啤酒。基于这个发现，超市调整了货架设置，把尿布和啤酒摆放在一起，大大增加了销售额。虽然该故事是哈弗商学院杜撰的，但确实很好地解释了关联规则挖掘的意义。

关联规则挖掘在实际生活中存在着广泛的应用，比如目录设计、商品推荐、定向广告、追加销售、货架设计、仓储规划、网络用法挖掘、入侵检测、网络故障分析、以及生物信息等。这里我们首先对关联规则挖掘中的一些基本概念做一个简单的介绍。以下面的超市购物篮数据集为例：

| TID   |      Items       |
|:-----:|:----------------:|
| $t_1$	| 牛奶,面包          |
| $t_2$	| 面包,尿布,啤酒,鸡蛋 |
| $t_3$ | 牛奶,尿布,啤酒,可乐 |
| $t_4$ | 面包,牛奶,尿布,啤酒 |
| $t_5$ | 面包,牛奶,尿布,可乐 |

表中的每一行代表一次购买清单。


# 基本定义

上表中的每一条记录被称为一个`事务`（Transaction), 包含一个唯一标识 TID 和给定顾客购买的商品集合（注意这里我们不考虑顾客购物的顺序，也不考虑购买的数量，只考虑某件商品在本次购物中出现与否）。
所有事务的集合被称为`事务集`，记为 $T$。

表中的每一个商品被称为一个 `项`（Item）。每一个事务中，顾客购买的商品构成一个项目的集合，称为`项集`(Itemset)。比如事务 $t_1$ 包含了项集 {牛奶,面包}。包含 $k$ 个项的集合被称为 `k-项集`（k-Itemset）。例如 {牛奶,面包} 是一个 2-项集，{尿布,可乐,鸡蛋} 是一个 3-项集。事务集 $T$ 中包含的全体项的集合我们记为 $I$。比如上表中 $I$ = {牛奶,面包,尿布,啤酒,鸡蛋,可乐}。

给定一个事务集 $T$ = {$t_1, t_2, \cdots, t_n$}， 以及所有项的集合 $I$ = {$i_1, i_2, \cdots, i_m$}。每个事务 $t_k$ 包含的项集都是 $I$ 的子集，即 $t_k \subseteq I $。事务的宽度定义为事务中出现项的个数。如果项集 $X$ 是事务 $t_k$ 的子集，则称事务 $t_k$ 包括项集 $X$。例如上表中事务 $t_2$ 包括项集 {面包，尿布}，但不包括项集 {面包，牛奶}。

关联规则是形如 $X \Rightarrow Y$ 的蕴含式。其中 $X$ 和 $Y$ 是项集，且有 $X \bigcap Y = \varnothing $。$X$ 称为关联规则的先导或前提，$Y$ 称为关联规则的后继或结论。例如上面的例子中，关联规则就是形如 {啤酒} $\Rightarrow$ {尿布} 的蕴含式。关联规则的强度可以使用它的`支持度`和`置信度`来度量。

# 支持度

支持度度量了**关联规则的普适性**。比如我们想分析关联规则$X \Rightarrow Y$， 其对应的支持度为:

  $$
  \text{Support}(X \Rightarrow Y) = P(XY) = \frac{\text{count}(X \cap Y)}{N}
  $$
  
  其中 $N$ 为整个数据集的事务总数。支持度越高，表明某一关联规则的适用性就越大。依次类推，如果我们想分析关联规则 $\\{X, Y\\} \Rightarrow Z$ 的支持度，则对应的计算公式为:
  
 $$
  \text{Support}(X, Y \Rightarrow Z) = P(XYZ) = \frac{\text{count}(X \cap Y \cap Z)}{N}
  $$
  
支持度是关联规则强度的一种重要度量，因为一方面支持度低的规则可能只是偶然出现，另一方面支持度低也表明该关联规则具有较低的实用性，不具备挖掘的价值。

# 置信度
置信度度量了**关联规则的准确度**。体现了一个数据出现后，另一个数据出现的概率。比如我们想分析关联规则$X \Rightarrow Y$， 其对应的置信度为:

  $$
  \text{Confidence}(X \Rightarrow Y) = P(Y \vert X) = \frac{ P(X \cap Y)}{P(X)}
  $$
  
置信度的本质就是条件概率，置信度越高，说明 $X$ 出现则 $Y$ 也出现的可能性越高。类推到多个数据的置信度，比如对于规则$\\{X, Y\\} \Rightarrow Z$ ，其置信度为：

  $$
  \text{Confidence}(X, Y \Rightarrow Z) = P(Z \vert X, Y) = \frac{ P(X \cap Y \cap Z)}{P(X \cap Y)}
  $$

# 提升度

对于一个关联规则 $X \Rightarrow Y$, 如果其满足最小支持度阈值和最小置信度阈值，即 $\text{Support}(X \Rightarrow Y) > \text{min_sup.}$, $\text{Confidence}(X \Rightarrow Y) > \text{min_conf.}$，则我们称其为强规则，否则为弱规则。关联规则挖掘的目的就是为了发现数据集中的强规则。但是发现的强关联规则是真实有效的么？这个问题可以通过提升度来衡量。[Ref. 6](http://fufeng.iteye.com/blog/1755125) 对这个问题讲解的非常清晰明了：

假设某家商场只卖两种商品：CD 机和 Mp3 两种音乐播放器，每天的交易量有 10000 单，其中 6000 单中包含 CD 机， 7000 单中包含 Mp3， 3000 单中既包含 CD 又包含 Mp3。假定 min_sup.=0.2， min_conf.=0.4。

规则 CD $\Rightarrow$ Mp3 的支持度和置信度分别为

$$
\text{Support}(\text{CD} \Rightarrow \text{MP3}) = \frac{6000}{10000} = 0.6 > \text{min_sup.}
$$

$$
\text{Confidence}(\text{CD} \Rightarrow \text{MP3}) = \frac{3000}{6000} = 0.5 > \text{min_conf.}
$$


因此 CD $\Rightarrow$ Mp3 属于强关联规则，按照 Apriori 算法的思路可以推导出购买 CD 机的顾客通常愿意再购买 Mp3 。
 

但实际上真的是这样吗？我们尝试做这样一个假设，如果该商店中如果不卖 CD 机的话，那么 mp3 的交易量会不会下降，如果下降说明 CD 机的销售会促进 mp3 的交易量；如果持平，则认为 Mp3 的销售和 CD 机的销售是两个独立事件，互不影响；如果上升，则认为 CD 机的的销售阻碍 Mp3 的交易量。


在概率论中，假如两个事件 $X$ 和 $Y$ 相互独立，那么 $X$ 和 $Y$ 同时发生的概率应该等于 $X$ 事件发生的概率乘以 $Y$ 事件发生的概率，即 $P(XY)=P(X)P(Y)$，$X$ 和 $Y$ 事件的存在互不影响。
 

- 如果 $P(XY)>P(X)P(Y)$, 说明 $X$ 和 $Y$ 同时发生的概率大于 $X$ 和 $Y$ 单独发生概率的乘积，那么 

  $$
  P （ X \vert Y ） = \frac{P(XY)}{P(Y)} > \frac{P(X)P(Y)}{P(Y)}=P(X) 
  $$ 

  也就是说当 $Y$ 发生时， $X$ 发生的概率会大于 $X$ 单独发生时的概率，即 $X$ 事件的发生会促进 $Y$ 事件的发生 。

 
- 如果 $P(XY)<P(X)P(Y)$, 说明 $X$ 和 $Y$ 同时发生的概率小于 $X$ 和 $Y$ 单独发生概率的乘积，那么 

  $$
  P （ X \vert Y ） = \frac{P(XY)}{P(Y)}< \frac{P(X)P(Y)}{P(Y)}=P(X)
  $$

  也就是说当 $Y$ 发生时， $X$ 发生的概率会小于 $X$ 单独发生时的概率，即 $X$ 事件的发生会阻碍 $Y$ 事件的发生 。



回到刚才的例子，我们可以发现


$$
\begin{eqnarray}
P(\text{CD} \& \text{Mp3}) & = & \frac{3000}{10000} = 0.3  \\
P(\text{CD})P(\text{Mp3}) & = & \frac{6000}{10000} \times \frac{7000}{10000}=0.42
\end{eqnarray}
$$

即 $P(\text{CD}\&\text{Mp3})< P(\text{CD})P(\text{Mp3})$，CD 机的交易其实会阻碍 Mp3 的交易量。

所以说，只凭支持度和置信度去衡量规则之间的关联关系是具有欺骗性的。因此，在前面两种度量标准的前提下还引入了第三种度量概念，称为提升度， 用来体现规则 $X \Rightarrow Y$ 中 $X$ 和 $Y$ 的相关性：

$$
\text{Lift}(X \Rightarrow Y) = \frac{\text{Confidence}(X \Rightarrow Y)}{P(Y)} = \frac{P(X \cap Y)}{P(X)P(Y)}
$$

如果 $\text{Lift}(X,Y)<1$, 则 $X$ 的出现和 $Y$ 的出现是负相关的，即相互阻碍；如果 $\text{Lift}(X,Y)>1$, 则 $X$ 和 $Y$ 是正相关，意味着一个的出现蕴含另一个的出现；如果 $\text{Lift}(X,Y)=1$，说明 $X$ 和 $Y$ 是相互独立的。
  

# References

1. *[数据挖掘导论](https://book.douban.com/subject/5377669/)*，P.-N. Tan, Steinbach, and V. Kumar 著；范明，范宏建 译， 人民邮电出版社, **2006.**

2. R. Agrawal, T. Imieliski, and A. Swami, "[Mining association rules between sets of items in large databases,](https://doi.org/10.1145/170035.170072)" in *Proceedings of the 1993 ACM SIGMOD international conference on Management of data,* Washington, D.C., USA, **1993,** pp. 207-216.

3. [Apriori 算法原理总结](http://www.cnblogs.com/pinard/p/6293298.html)

4. [数据挖掘经典案例分享：啤酒与尿布](http://www.jianshu.com/p/a22890153f93)

5. [数据挖掘系列（1）关联规则挖掘基本概念与Aprior算法](http://www.cnblogs.com/fengfenggirl/p/associate_apriori.html)

6. [关联规则（二）强关联规则一定就是用户感兴趣的规则吗](http://fufeng.iteye.com/blog/1755125)

