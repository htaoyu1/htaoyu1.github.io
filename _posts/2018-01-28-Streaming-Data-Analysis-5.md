---
layout: post
title: 流数据聚类分析(5)：HPStream 算法
category: Machine Learning
author: Hongtao Yu
tags: 
  - machine-learning
  - data-streaming
comments: true
use_math: true
lang: zh
---

- TOC
{:toc}

# 概述

高维度数据通常具有`稀疏性`（sparsity）。高维度情况下，所有数据点之间具有近似相等的距离，因此对基于距离的聚类方法提出了挑战。这就是所谓的`高维诅咒`，又称`维数灾难`（curse of dimensionality）。对于高维数据，一个解决的方案是使用一种称为`投影聚类`（projected clustering）的算法，即对数据在某一些维度子集下进行聚类。所以投影聚类的结果就是一些在不同子维度下的簇。不同簇使用的子维度可以不相同。这样聚类形成的簇叫做`投影簇`（projected clusters）。

假定 $k$ 是簇的目标数目，$l$ 是子空间的维度。投影聚类包含如下两个方面：

1. 将数据集划分为 $k+1$ 个部分 $\\{\mathcal{C}_1, \mathcal{C}_2, \cdots, \mathcal{C}_k, \mathcal{O}\\}$, 其中前 $k$ 个 $\mathcal{C}$ 为  $k$ 个簇，最后一个 $\mathcal{O}$ 为离群点的集合。

2. 每个簇 $\mathcal{C}_i$ 的投影子空间 $\mathcal{E}_i$ （$1 \leq i \leq k$）。子空间维度的大小为 $l$。

`HPStream`算法（**H**igh-dimensional **P**rojected **Stream** clustering）[^HPStream] 通过对子空间包含的维度持续优化，以达到对流数据实施投影聚类的目的。在 HPStream 中，每个簇的维度随着时间持续更新。与其他的算法相比，HPStream 算法

1. 对流数据引入了投影聚类的概念。现实世界中大部分的数据都是高维的，因此有必要设计一种可以对高维流数据进行高质量聚类的方法。`STREAM` 和 `CluStream` 流数据算法在整个维度空间对数据点进行聚类，不能很好地处理高维度的情况。此外 `PROCLUS` 投影聚类算法由于需要多数据进行多次扫描，不适合应用于流数据的场景。

2. 探索了投影聚类的线性更新方法，达到了高可缩放性和高质量的聚类。线性更新的方法最初在 `BIRCH` 聚类方法中使用，`CluStream` 将该方法引入到流聚类算法中。然而，CluStream 方法对高维数据的聚类质量并不是很好。然而，真实世界的数据很大一部分都是高维度的，并且只在一部分维度上可以形成很紧凑的聚类簇。`HPStream` 算法由于其对高维数据的适应性，对真实世界的数据可以达到很好的聚类质量。

除了上面的优点，HPStream 提出并探索了其他创新点。例如，`衰退簇结构`（fading cluster structure）使用用户提供并且可调的衰退因子，很好地整合了历史数据和当前数据。此外，HPStream 引入了位矢量（bit-vector）来注册并动态更新簇的聚类相关维度，使用最小半径来提高聚类的质量，等等。

# 衰退簇结构（The Fading Cluster Structure）

假设流数据 $X_1, X_2, \cdots, X_k, \cdots$ 是一系列到达时间为 $T_1, T_2, \cdots, T_k, \cdots$ 的 $d$ 维数据 $X_i = (x_i^1, x_i^2, \cdots, x_i^d)$。由于流数据聚类要给最新的点相对较高的重要性，所以引入了`衰退数据结构`（fading data structure）的概念，可以灵活地调整最新数据的权重。衰退数据结构假定每一个数据都有一个依赖于时间的权重因子 $f(t)$，$f(t) \in (0, 1)$ 又称为`衰退函数`（fading function）。

**定义 1：** 一个数据点的半衰期定义为权重 $f(t_0) = (1/2)f(0)$ 的时间。

半衰期定义了一个数据点权重的衰减速率。相应地，一个数据点的衰减速率定义为半衰期的倒数 $\lambda = 1/t_0$。因此，为了让半衰期的性质成立，这里定义一个流数据的权重因子为 $f(t) = 2^{-\lambda \cdot t}$。这里 $\lambda$ 控制了数据点权重的衰减速度，数据的半衰期为 $1/\lambda$，越大的 $\lambda$ 值意味着数据点的权重衰减越快。

**定义 2：** 一个 $d$ 维数据的集合 $\mathcal{C} = \\{ X_{i_1}, \cdots, X_{i_n}\\}$，时间戳为 $T_{i_1} \cdots, T_{i_n}$， 在 $t$ 时刻的衰退簇结构定义为一个 $(2\cdot d + 1)$ 的元组 $\mathcal{FC}(\mathcal{C},t) = (\overline{FC2^x(\mathcal{C},t)}, \overline{FC1^x(\mathcal{C},t)}, W(t))$。矢量 $\overline{FC2^x(\mathcal{C},t)}$ 和 $\overline{FC1^x(\mathcal{C},t)}$ 均包含 $d$ 个元素：

1. $\overline{FC2^x(\mathcal{C},t)}$ 的第 $j$ 个元素为集合 $\mathcal{C}$ 中所包含的数据在第 $j$ 维的值的加权平方和 $\sum_{k-1}^n f(t-T_{i_k}) \cdot (x_{i_k}^j)^2$。

2. $\overline{FC1^x(\mathcal{C},t)}$ 的第 $j$ 个元素为为集合 $\mathcal{C}$ 中所包含的数据在第 $j$ 维的值的加权和 $\sum_{k-1}^n f(t-T_{i_k}) \cdot (x_{i_k}^j)$。

3. $W(t)$  是一个标量，为集合 $\mathcal{C}$ 中所包含数据点的权重之和 $\sum_{k-1}^n f(t-T_{i_k})$。

可以证明，通过上述方法定义的衰退簇结构满足`可加性`（additivity）和`时间可乘性`（temporal multiplicity）：

**可加性：** 令 $\mathcal{FC}(\mathcal{C}_1,t)$ 和$\mathcal{FC}(\mathcal{C}_2,t)$ 分别为 $\mathcal{C}_1$ 和 $\mathcal{C}_2$ 的簇结构。则 $\mathcal{C}_1 \cup \mathcal{C}_2$ 的簇结构可以表示为 $\mathcal{FC}(\mathcal{C}_1 \cup \mathcal{C}_2,t) = \mathcal{FC}(\mathcal{C}_1, t) + \mathcal{FC}(\mathcal{C}_2, t)$。

**时间可乘性：** 考虑一个在 $t$ 时刻的簇结构 $\mathcal{FC}(\mathcal{C}, t)$，如果在时间间隔 $(t, t + \delta t)$ 内该结构没有吸收任何数据点，那么有 $\mathcal{FC}(\mathcal{C}, t + \delta t) = e^{-\lambda \delta t} \cdot \mathcal{FC}(\mathcal{C}, t)$。

由于投影聚类中每一个簇都是在一个数据子维度空间内的聚类，所以对每一个簇 $\mathcal{C}$ 还有一个对应的位矢量 $\mathcal{B}(\mathcal{C})$，用来记录该簇使用了哪些维度。$\mathcal{B}(\mathcal{C})$ 是一个长度为 $d$ 的矢量，如果簇 $\mathcal{C}$ 使用了某一个维度，则矢量对应位置的元素的值为 1， 否则为 0。

# 高维度数据的投影聚类算法

流数据的投影聚类算法使用迭代方法通过优化每个簇包含的子维度来持续更新新簇的结构。由于数据在每个维度的取值范围大小不一样（例如工资，年龄等），所以在聚类开始阶段需要对数据进行正则化（normalization）以方便对不同的数据进行比较。算法首先使用初始化时候的数据，计算每个维度 $i$ 的标准差 $\sigma_i$，然后将每个维度的值分别除以对应维度的 $\sigma$，得到正则化的数据。需要注意的是，由于数据流在不断的演化，所以数据的标准差 $\sigma_i$ 也一直在变。因此每隔一定的时间，需要对数据的标准差进行重新计算。每当 $\sigma_i$ 更新的时候，相应的衰退簇统计量也需要更新。假设正则化的时候，簇在第 $i$ 个维度的标准差从 $\sigma_i$ 变成了 $\sigma'_i$，那么每个簇 $\mathcal{C}$ 的统计量 $\mathcal{FC}(\mathcal{C},t) = (\overline{FC2^x(\mathcal{C},t)}, \overline{FC1^x(\mathcal{C},t)}, W(t))$ 需要进行相应的修改。特别地，$\overline{FC2^x(\mathcal{C},t)}$ 的第 $i$ 个维度的值需要乘以 $\sigma_i^2 / \sigma_i^{\prime2}$，$\overline{FC1^x(\mathcal{C},t)}$ 的第 $i$ 个维度的值需要乘以 $\sigma_i/\sigma'_i$。


## HPStream 算法

![5-1 HPStream 聚类算法](/assets/blog-images/Streaming-Data-Analysis-5.1.png)
**Fig. 5-1.** `HPStream` 聚类算法。

**Fig. 5-1** 显示了如何添加一个新的流数据点 $\overline{X}$ 到聚类簇的增量算法。算法的输入包括了当前的聚类簇结构 $\mathcal{FCS}$，每个簇对应的最优投影子空间的维度（记录在位矢量 $\mathcal{BS}$ 中）,最大簇个数 $k$，簇的 *平均* 投影子空间的维度大小 $l$。$\mathcal{FCS}$ 和 $\mathcal{BS}$ 随着数据流需要动态地更新，$\mathcal{FCS}$ 中每个簇 $\mathcal{C}_i$ 使用的子维度记录在位矢量 $\mathcal{B}(\mathcal{C}_i)$ 中。

流数据聚类算法使用迭代的方式在每一步将一个新的数据点分配给最近邻的聚类簇结构。聚类簇的最近邻性由投影距离度量决定，即对每一个簇，计算距离的时候只使用该聚类簇对应的子维度。同时，我们需要持续地优化每个簇对应的子维度。优化的目标是保证每个簇在对应投影子维空间的的半径尽可能地小。因此，聚类过程<u>在维护簇的同时还需要维护每个簇对应的子维度</u>。

1. 算法首先使用 `ComputeDimensions` 模块找到聚类使用的最优子空间的维度，

2. 然后对每个簇，计算数据点 $\overline{X}$ 在该簇对应的投影子空间内的到簇中心的平均投影距离。

3. 选取离 $\overline{X}$ 最近的簇 $\mathcal{C}_{index}$，计算该簇在其最优投影子空间内的自然极限半径。
   > 注：计算自然极限半径的时候，只考虑 $\mathcal{C}_{index}$ 中已经包含的数据点，新的数据 $\overline{X}$ 并没有用来计算极限半径。 
   
4. 如果 $\overline{X}$ 位于 $\mathcal{C}\_{index}$ 的自然极限半径内，则将 $\overline{X}$ 添加到簇 $\mathcal{C}_{index}$ 内。

5. 如果 $\overline{X}$ 位于 $\mathcal{C}_{index}$ 的自然极限半径外，则使用 $\overline{X}$ 创建一个只包含一个孤立点的新簇。

6. 删除投影子空间维度为 0 的簇。
   > 最优投影子空间的为什么可以为 0， 请参见 ComputeDimensions 部分。

7. 如果簇的个数大于  $k$ ，则删除最久没有吸收新数据的簇。

如果一个簇 $\mathcal{C}\_{index}$ 吸收了一个新的数据点，则需要对该簇的衰退簇结构 $\mathcal{FC}(\mathcal{C}\_{index},t)$ 进行更新。 假定 $\overline{X}$ 在时刻 $t$ 被簇 $\mathcal{C}\_{index}$ 吸收，簇 $\mathcal{C}\_{index}$ 上次更新的时间为 $t^{up}$，则首先将 $\mathcal{FC}(\mathcal{C}\_{index},t)$ 内的每个元素乘以 $e^{- \lambda ( t - t^{up})}$，然后将新数据的统计量添加到更新过的 $\mathcal{FC}(\mathcal{C}\_{index},t)$ 中。

## ComputeDimensions 模块

![5-2 ComputeDimensions 计算每个簇的最优投影子空间维度](/assets/blog-images/Streaming-Data-Analysis-5.2.png)
**Fig. 5-2.** `ComputeDimensions` 计算每个簇的最优投影子空间维度。

每个簇使用的最优子维度通过 ComputeDimensions 模块进行更新。该模块通过保证数据在选定维度的分散尽可能地小来决定该使用哪些维度。值得注意的是，有些簇包含的数据点非常少，对于这样的簇，维度的计算可能缺乏统计鲁棒性。在极端情况下，一个簇可能只包含一个数据点。对于这样的非简并情况，没办法使用统计的方法对不同维度进行区分。因此需要新添加的数据点 $\overline{X}$ 来决定每个簇的子维度空间。特别地，将数据点 $\overline{X}$ 临时性地添加到每个簇中，然后计算每个簇在不同维度下的分散度，选取最小的 $\vert \mathcal{FCS} \vert \cdot l$ 个维度作为最优投影子空间，每个簇相应的子空间维度被记录在 $\mathcal{BS}$ 中。将数据点 $\overline{X}$ 包含到最优子空间的计算中可以保证对包含数据较少的簇能得到一个稳定的结果。

1. 使用新的数据点 $\overline{X}$ 创建 $\vert \mathcal{FCS} \vert$ 个实验性的簇，

2. 对这 $\vert \mathcal{FCS} \vert$ 个簇的  $d$ 个维度，计算 $\vert \mathcal{FCS} \vert \cdot d$ 个半径。

3. 将第 2 步得到的 $\vert \mathcal{FCS} \vert \cdot d$ 个半径进行升序排列，选取前 $\vert \mathcal{FCS} \vert \cdot l$ 个维度作为最优投影子空间。

4. 对每个簇 $\mathcal{C}_r$, 创建一个位矢量 $\mathcal{B}(\mathcal{C}_r)$ 用来记录其最优投影子空间所用的维度。

> 这里值得一提的是，并不是每个簇都会被投影到一个 $l$ 维的子空间。有些簇的最优投影子空间可能是大于 $l$ 维的，有些簇的最优投影子空间可能是小于 $l$ 的。极端的情况下，如果簇内的数据分的比较散，则该簇的投影子空间可能是 0 维的，对于这样的簇，算法会把簇删除。


## FindProjectedDist 模块

![5-3 FindProjectedDist 计算簇的投影距离](/assets/blog-images/Streaming-Data-Analysis-5.3.png)
**Fig. 5-3.** `FindProjectedDist` 计算簇的投影距离。

接下来是计算新数据点 $\overline{X}$ 的最近邻簇。特别地，对每个簇 $\mathcal{C}_r$, 找到 ComputeDimensions 模块得到的其最优投影子空间，然后计算并返回 $\overline{X}$ 到簇 $\mathcal{C}_r$ 的中心在该子空间内的平均投影距离（每个簇的子空间维度被被保存在 $\mathcal{BS}$ 中）。该平均投影距离被称为`曼哈顿分段距离`（Manhattan Segmental Distance）。这部分的伪码 FindProjectDist 显示在 **Fig. 5-3** 中。需要提的是，这一阶段并不需要对数据点的正则化进行更新。


## FindLimitingRadius 模块

![5-4 FindLimitingRadius 计算簇的极限半径](/assets/blog-images/Streaming-Data-Analysis-5.4.png)
**Fig. 5-4.** `FindLimitingRadius` 计算簇的极限半径。

一旦确定了数据点 $\overline{X}$ 准备分配给哪个簇，就需要计算相应簇的自然极限半径（limiting radius）。这个半径被认为是该簇的自然边界。位于自然边界之外的数据点不应该包含在该簇内，而应该使用这些数据点创建一个新的簇。这部分的伪码 FindingLimitingRadius 显示在 **Fig. 5-4** 中。算法使用衰减簇结构内维护的数据计算簇在每个维度的半径，然后找到其最优投影子空间内的平均簇半径。一个簇的自然簇边界定义为该簇的平均半径乘以一个`边界因子`（boundary factor） $\tau$ 。


一旦确认 $\overline{X}$ 处于簇的自然边界内，就可以将新的数据点添加到簇内，否则，需要对 $\overline{X}$ 创建一个新的只包含一个孤立点的簇。如果该新的数据点是噪声，那么将来只会有有限的几个数据点会被添加到 $\overline{X}$ 的簇中。这样的簇最终会被删除。


## 初始化

在流数据聚类开始阶段，首先对最初的 $InitNumber$ 个数据点使用 k-means 算法在全空间进行聚类，得到 $k$ 个最初的簇。然后对每个簇，分别使用 ComputeDimensions 模块找到该簇对应的最优投影子空间。接着在每个簇的投影子空间内，计算每个数据点到簇中心的距离，根据该距离对数据点重新分配簇标签。需要注意的是，这一步分配的标签可能与在全空间聚类时数据点的簇标签不一样。然后算法使用新的簇标签，计算得到 $k$ 个新的簇中心，然后继续迭代计算每个簇的最优子空间，以及重新分配簇标签。不断迭代这个过程直到收敛。收敛的聚类簇被用来创建流数据开始时的衰退簇结构。 

# Reference

[^HPStream]: C. C. Aggarwal, J. Han, J. Wang, and P. S. Yu, "[A framework for projected clustering of high dimensional data streams,](https://dl.acm.org/citation.cfm?id=1316763)" in *Proceedings of the Thirtieth international conference on Very large data bases*, Vol. 30, Toronto, Canada, **2004,** pp. 852-863.




