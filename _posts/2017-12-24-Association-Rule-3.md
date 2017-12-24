---
layout: post
title: 数据挖掘之关联规则分析（3）：FP-Growth 算法
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

`Apriori 算法`可以有效地减小发现频繁项集（又称频繁模式）的搜索空间。但是如果数据集中的频繁模式非常多，或者频繁模式特别长，又或者我们设置的支持度阈值很低，那么 Apriori 算法的效率就会变得低下。比如对于包含 $10^4$ 个频繁 1-项集的数据集，Apriori 算法需要产生超过 $10^7$ 个候选 2-项集。为了发现一个长度为 100 的频繁模式 $\{a_1, a_2, \cdots, a_{100}\}$，Apriori 算法总共需要产生 $2^{100}-2 \approx 10^{30}$ 个候选项集。
此外，Apriori 算法需要多次扫描数据库。如果数据集中的最大频繁模式长度为 $k_{max}$, 那么 Apriori 算法就需要扫描 $k_{max} + 1$ 次数据集。

# FP-Tree 
与 Apriori 算法不同，`FP-Growth 算法`不需要产生候选集，并且只需要对数据集扫描两次。 FP-Growth 算法基于一种叫做 `FP-Tree` （频繁模式树，Frequent-Pattern Tree）的数据结构。通过第一遍扫描数据库，首先建立一个支持度大于 min_sup 的频繁 1-项集，并按照每项的支持度计数进行降序排列。这个排列好的项集被称作`项头表`（item header table），或`频繁项头表`（frequent item header table）。然后第二次扫描数据库，逐条读取每条事务，并将其中的项目首先按照其在项头表中的顺序进行排列，然后逐条插入 FP-Tree 中。**Fig. 3-1**显示了 FP-Tree 的构建过程。接下来我们会对每一步进行详细的解释。

![3-1 FP-Tree 算法概览](/assets/blog-images/Association-Rule-3.1.png)
**Fig. 3-1.** FP-Tree 算法概览。

在构建 FP-Tree 之前，我们先看一下树的内部节点的结构。
```py
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue       # 项的名称
        self.count = numOccur       # 项的计数
        self.nodeLink = None        # 兄弟指针
        self.parent = parentNode    # 父指针
        self.children = {}          # 子指针（可以有多个）
```
每个树的节点包括 5 个域，即项的名称，项的计数，Node Link（兄弟节点指针），父节点指针和子节点指针。项头表的结构包括 2 个域， 项的名称和 Node Link 的头指针（head of node link）（见 **Fig. 3-1**的左下图）。算法开始的时候，所有的这些头指针都为空。


![3-2 构造 FP-Tree：事务 $t_1$](/assets/blog-images/Association-Rule-3.2.png)
**Fig. 3-2.** 构造 FP-Tree：事务 $t_1$。读入事务。 $t_1$ 包含项 {a, b}，由于项头表中项的排序为 $abcde$, 所以 $t_1$ 中项目的按降序排序后为 {a, b}。接下来创建两个节点 a 和 b, 并建立一个路径 $null \to a \to b$, 将 a 和 b 的计数都设置为 1。
由于 a 和 b 都是第一次出现，所以将项头表中项 a 和 b 的 Node Link 的头指针分别指向 FP-Tree 中的 a 节点 和 b 节点（红色虚线）。
  
  
![3-3 构造 FP-Tree：事务 $t_2$](/assets/blog-images/Association-Rule-3.3.png)
**Fig. 3-3.** 构造 FP-Tree：事务 $t_2$。读入事务。 $t_2$ 包含项 {c, b, d}， 按降序排序后为 {b, c, d}。由于路径 $null \to b \to c \to d$ 不与树中已有的任何路径重合，所以我们在树中构建三个新的节点 b, c, d, 并将他们的计数都设为1。由于节点 c 和 d 都是第一次出现，所以将项头表中项 c 和 d 的 Node Link 的头指针分别指向他们（红色虚线）。由于节点 b 之前已经出现过，所以将上一次出现的 b 节点的 Node Link 指向它（绿色虚线）。这里使用指针将相同的项连接起来的目的是为将来挖掘频繁模式做准备。


![3-4 构造 FP-Tree：事务 $t_3$](/assets/blog-images/Association-Rule-3.4.png)
**Fig. 3-4.** 构造 FP-Tree：事务 $t_3$。读入事务。 $t_3$ 包含项 {a, d, e, c}， 按降序排序后为 {a, c, d, e}。由于路径 $null \to a \to c \to d \to e$ 的第一部分 $null \to a$ 已经存在于树中，所以将该 a 节点的计数增加1， 并将 cde 添加到其后面，计数都置为1。此时节点 e 第一次出现，将项头表中 e 的指针指向它。由于节点 c 和 d 又出现在一条新的路径上，所以用上一次新出现的 c, d 节点指向他们。


![3-5 构造 FP-Tree：事务 $t_4$](/assets/blog-images/Association-Rule-3.5.png)
**Fig. 3-5.** 构造 FP-Tree：事务 $t_4$。读入事务。 $t_4$ 包含项 {a, e, d}， 降序排序后为 {a，d, e}。由于路径 $null \to a \to d \to e$ 的 $null \to a$ 部分已经存在于树中，所以将该 a 节点的计数增加1， 并将 de 添加到其后面，计数都置为1。由于节点 d 和 e 又出现在一条新的路径上，所以用上一次新出现的 d, e 节点指向他们。


![3-6 构造 FP-Tree：事务 $t_5$](/assets/blog-images/Association-Rule-3.6.png)
**Fig. 3-6.** 构造 FP-Tree：事务 $t_5$。读入事务。 $t_5$ 包含项 {a, c, b}， 降序排序后为 {a，b, c}。路径 $null \to a \to b \to c$ 的 $null \to a \to b$ 部分已经存在于树中，所以将该 a 和 b 节点的计数增加1， 并将 c 添加到其后面，计数置为1。由于 c 又出现在一条新的路径上，所以用上一次新出现的 c 节点指向它。


![3-7 构造 FP-Tree：事务 $t_6$](/assets/blog-images/Association-Rule-3.7.png)
**Fig. 3-7.** 构造 FP-Tree：事务 $t_6$。读入事务。 $t_6$ 包含项 {a, c, d, b}， 降序排序后为 {a，b, c, d}。路径 $null \to a \to b \to c \to d$ 的 $null \to a \to b \to c$ 部分已经存在于树中，所以将重合部分节点的计数增加1， 并将 d 添加到其后面，计数置为1。由于 d 又出现在一条新的路径上，所以用上一次新出现的 d 节点指向它。


![3-8 构造 FP-Tree：事务 $t_7$](/assets/blog-images/Association-Rule-3.8.png)
**Fig. 3-8.** 构造 FP-Tree：事务 $t_7$。读入事务。 $t_7$ 包含项 {a, f}， 由于 f 没有出现在项头表中，是非频繁项，所以剪除。此时 $t_7$ 的频繁项只剩下 a，路径 $null \to a$ 已经存在于树中，所以将路径上的节点计数都增加1 （这里只有a）。


![3-9 构造 FP-Tree：事务 $t_8$](/assets/blog-images/Association-Rule-3.9.png)
**Fig. 3-9.** 构造 FP-Tree：事务 $t_8$。读入事务。 $t_8$ 包含项 {a, c, b}, 降序排序后为 {a，b, c}。路径 $null \to a \to b \to c$ 已经存在于树中，所以将路径上的节点计数都增加1。


![3-10 构造 FP-Tree：事务 $t_9$](/assets/blog-images/Association-Rule-3.10.png)
**Fig. 3-10.** 构造 FP-Tree：事务 $t_9$。读入事务。 $t_9$ 包含项 {d, b, a, g}, 删除非频繁项 g, 并将剩余的项目按降序排列得到 {a，b, d}。路径 $null \to a \to b \to d$ 的 $null \to a \to b$ 部分已经存在于树中，所以将该部分的节点计数都增加1， 并将 d 添加到后面，计数置为1。由于 d 又出现在一条新的路径上，所以用上一次新出现的 d 节点指向它。


![3-11 构造 FP-Tree：事务 $t_{10}$](/assets/blog-images/Association-Rule-3.11.png)
**Fig. 3-11.** 构造 FP-Tree：事务 $t_{10}$。读入事务。 $t_{10}$ 包含项 {c，e, b}, 按降序排序后得到 {b, c, e}。路径 $null \to b \to c \to e$ 的 $null \to b \to c$ 部分已经存在于树中，所以将该部分的节点计数都增加1， 并将 e 添加到后面，计数置为1。由于 e 又出现在一条新的路径上，所以用上一次新出现的 e 节点指向它。


至此，通过2次扫描数据集，我们把 FP-Tree 构建好了。FP-Tree 将相同的项尽可能地保存到一个节点，从而节省了存储空间。接下来就是如何从这个树中挖掘我们想要的频繁模式了。


# FP-Growth

`FP-Growth` 是对 FP-Tree 进行频繁模式挖掘的一种算法。还记得 FP-Tree 的每个节点包含了三个指针：指向父节点的指针，指向子节点的指针，以及指向兄弟节点的指针。指向子节点的指针主要是在构建树的过程中使用，使得我们可以沿着每条路径搜索；而指向兄弟节点和父节点的指针主要用在挖掘阶段。兄弟节点的指针帮助我们遍历所有以该项结尾的模式，而父节点指针则帮助我们至下而上地对树进行挖掘。下面以我们之前的例子详细说明。

FP-Growth 算法从项头表的底部向上开始挖掘。对于项头表的每一项，从项头表中 Node Link 的头指针开始，沿着 Node Link 找到所有以该项为叶子结点的路径，基于此路径建立前缀路径子树，以及该项的条件模式基，并最终建立该项的条件 FP-Tree。在条件 FP-Tree 的基础上采用分治方法递归找到所有以该项结尾的频繁模式。好了，废话不说，直接开挖。

![3-12 FP-Growth：挖掘以项 e 结尾的频繁模式](/assets/blog-images/Association-Rule-3.12.png)
**Fig. 3-12.** FP-Growth：挖掘以项 e 结尾的频繁模式。

从项头表的底部开始，首先沿着 **Fig. 3-12a** 中的蓝色箭头找到所有以 e 结尾的子树（**Fig. 3-12b**）。这个子树被称为以 e 结尾的`前缀路径子树`（prefix path sub-tree）。由于 e 的总计数为 3，大于最小支持度计数 2，所以 e 是一个频繁项，因此我们必须解决发现以 de, ce, be, 和 ae 结尾的频繁项集的子问题。为了解决这些子问题，我们必须首先将前缀路径子树转化为 `条件 FP 树`（conditional FP-Tree）。


从图中可以看出，我们有 3 条路径，分别为 $\langle a:8, c:1, d:1, e:1 \rangle$，$\langle a:8, d:1, e:1 \rangle$ 和 $\langle b:2, c:2, e:1 \rangle$。第一条路径，$\langle a:8, c:1, d:1, e:1 \rangle$，表示模式 acde 在数据集中出现一次。该路径同时也表示 a 在数据集中总共出现了 8 次，但是与 e 一起，它只出现了一次。为了挖掘与 e 一起出现的模式，我们这里只考虑 e 的`前缀路径`(prefix path) $\langle a:1, c:1, d:1 \rangle$, 简记为 $\langle acd:1 \rangle$。第二条路径， $\langle a:8, d:1, e:1 \rangle$，表示模式 ade 在数据集中出现一次, e 的前缀路径为 $\langle ad:1 \rangle$。第三条路径，$\langle b:2, c:2, e:1 \rangle$，表示模式 bce 在数据集中也出现了一次。bc 在数据集中出现了两次，但是与 e 一起形成的 bce 只出现了一次。因此 e 的前缀路径为 $\langle bc:1 \rangle$。 集合 {$\langle acd:1 \rangle, \langle ad:1 \rangle, \langle bc:1 \rangle$} 构成了 e 的`条件模式基`（conditional pattern base）,即 e 存在条件下的子模式基。

接下来我们需要在此条件模式基的基础上构建 e 的条件 FP 树。我们首先统计 e 的条件模式基中各项出现的次数，有 a:2, c:2, d:2, b:1。由于 b 的计数小于最小支持计数 2，我们将其删除，得到 e 的条件 FP 树（**Fig. 3-12c**）。 可以看到 e 的条件 FP 树与原来的前缀路径略有不同，频度计数已经更新，并且删除了 b。由于我们此时我们已经不在需要 e 节点的信息，所以 e 也被删除了。


为了发现以 de 结尾的频繁项集，从 e 的条件 FP 树中搜集 d 的所有前缀路径子树（**Fig 3-12d**）。由于 d 的计数为 2， 所以 {d,e} 的支持度计数也为 2， 不小于最小支持度计数，所以为频繁项。此时我们需要解决发现以 cde 和 ade 结尾的频繁项集的问题(我们此时不再考虑 b，因为 b 已经不存在于以 de 结尾的前缀路径子树中了)。为此，我们构造 de 的条件 FP 树，更新频度计数以及删除非频繁项 c 之后， **Fig 3-12e** 显示了 de 的条件 FP 树。因为该 FP 树只包含一个频繁项 a，所以我们直接写出以 de 结尾的频繁项集 {a, d, e}。

接下来计算以 ce 结尾的频繁项，处理 ce 的前缀路径子树后(**Fig. 3-12f**)，发现 {c, e} 是频繁项。因此要构建 ce 的条件 FP 树，但是在构建的时候我们发现，a 的计数需要更新为 1， 小于最小支持度计数，因此 a 被删除。所以以 ce 结尾的频繁项集只有 {c, e}。

接下来算法继续发现以 ae 结尾的频繁项集 {a, e}（**Fig. 3-12g**）。通过这些步骤我们最终找到了包含 e 的频繁项集 {e}, {d, e}, {a, d, e}, {c, e}, {a, e}。通过这个例子我们看到， FP-Growth 算法使用了`分治`(devide and conquer)策略，递归找到了所有以 e 结尾的频繁项集。

在进行下一步挖掘之前，我们先整理一下思路，回想一下针对 e 的挖掘过程：

1. 首先统计 e 的计数，如果其不小于最小支持度计数，我们找到了第一个频繁项集 {e}。

2. 如果 e 是频繁项，那么找到以 e 结尾的前缀路径子树。

3. 更新 e 的前缀路径子树中各个节点的计数，得到 e 的条件模式基。

4. 寻找在项头表中排在 e 之前的所有项与 e 形成的 2-项集的频繁模式。即 de, ce, be, 和 ae. 如果在条件模式基中，某个项的计数小于最小支持度计数。比如上例中的 b，那么我们可以将其从候选 2-项集中直接删除。所以我们得到了三个频繁项 {d, e}, {c, e}, {a, e}。 

5. 基于 e 的条件模式基建立条件 FP 树。如果某个项已经删除，那么其在条件 FP 树中的节点也一并删除。

6. 然后使用此 FP 树，回到步骤1的流程，分别挖掘包含 {d, e}, {c, e}, {a, e} 的更高阶频繁项。

接下来的过程不在赘述，直接看图。

![3-13 FP-Growth：挖掘以项 d 结尾的频繁模式](/assets/blog-images/Association-Rule-3.13.png)
**Fig. 3-13.** FP-Growth：挖掘以项 d 结尾的频繁模式。得到频繁项集 {d}, {c, d}, {b, d}, {a, d}, {b, c, d}, {a, c, d}, {a, b, d}。


![3-14 FP-Growth：挖掘以项 c 结尾的频繁模式](/assets/blog-images/Association-Rule-3.14.png)
**Fig. 3-14.** FP-Growth：挖掘以项 c 结尾的频繁模式。得到频繁项集 {c}, {b, c}, {a, c}, {a, b, c}。


![3-15 FP-Growth：挖掘以项 b 结尾的频繁模式](/assets/blog-images/Association-Rule-3.15.png)
**Fig. 3-15.** FP-Growth：挖掘以项 b 结尾的频繁模式。得到频繁项集 {b}, {a, b}。


![3-16 FP-Growth：挖掘以项 a 结尾的频繁模式](/assets/blog-images/Association-Rule-3.16.png)
**Fig. 3-16.** FP-Growth：挖掘以项 a 结尾的频繁模式。得到频繁项集 {a}。


至此，我们对整个数据集挖掘完毕。得到的频繁项为：

| 后缀  |      频繁项集       |
|:-----:|:----------------:|
| e     | {e},{d,e},{a,d,e},{c,e},{a,e}|
| d     | {d}, {c,d}, {b,c,d}, {a,c,d}, {b,d}, {a,b,d}, {a,d}|
| c     | {c},{b,c},{a,b,c},{a,c}|
| b     | {b},{a,b} |
| a     | {a}|


# References

1. J. Han, J. Pei, Y. Yin, and R. Mao, "[Mining Frequent Patterns without Candidate Generation: A Frequent-Pattern Tree Approach,](https://link.springer.com/article/10.1023/B:DAMI.0000005258.31418.83)" *Data Mining and Knowledge Discovery,* vol. 8, pp. 53-87, **2004.**

2. [机器学习实战](https://book.douban.com/subject/24703171/)，Peter Harrington 著；李锐，李鹏，曲亚东，王斌译， 人民邮电出版社, **2013.**


3. [数据挖掘导论](https://book.douban.com/subject/5377669/)，P.-N. Tan, Michael Steinbach, and V. Kumar 著；范明，范宏建 译， 人民邮电出版社, **2006.**


4. [数据挖掘：概念与技术](https://book.douban.com/subject/2038599/)，Jiawei Han, Micheline Kamber, and Jian Pei 著；范明，孟小峰 译， 机械工业出版社, **2007.**

5. [FP Tree算法原理总结](http://www.cnblogs.com/pinard/p/6307064.html)








