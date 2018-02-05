---
layout: post
title: 异常检测（1）：Isolated Forest 算法
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

大部分基于模型的异常检测算法（包括基于统计的，基于分类的，以及基于聚类的方法）都是通过使用正常数据构建一个模型，不能被这个模型刻画的数据被认为是异常的。这样做的缺点有两个：

1. 异常检测器是基于正常值，而不是基于检测异常而优化，因此，可能会造成很多的假告警（即正常数据被当成了异常值），或者很少的异常被探测到。

2. 由于算法的计算复杂度一般都比较高，所以只能应用于低维数据，或者小样本。


`Isolated Forest` (iForest) 算法[^iForest1] [^iForest2]也是一种基于模型的方法，不同的是 iForest 着重于孤立异常值，而不是刻画正常值。该方法使用了异常值的两个定量性质：

1. 异常值是只有少量实例的少数类。

2. 异常实例的属性值非常不同于正常实例。

简言之，异常值具有“少而不同”的性质，导致他们比正常值更容易被孤立（isolation）。iForest 通过建立树结构来孤立数据集中的每一个实例。由于异常值更容易被孤立，所以他们更靠近树的根部，而正常值更靠近树的枝叶部分，这种树的孤立特征奠定了使用 iForest 来探测异常值的理论基础。这种树被称作 `Isolation Tree` 或者 `iTree`。iForest 对一个给定的数据集，建立一个 iTrees 系综。这样，异常值就是那些在系综内的 ITrees 中具有相对较短的平均路径长度的实例。iForest 具有两个变量：系综内 iTrees 的个数以及子采样（sub-sampling）的大小。iForest 只使用少量的 iTrees 就可以达到很快的性能收敛，并且它只要求一个很小的子采样就可以达到很高的探测性能。

除了孤立方法和刻画方法的关键不同之外，iForest 与基于模型的，基于距离的，和基于密度的方法相比，还具有以下优点：

1. iTrees 的孤立特征使得只建立部分模型成为了可能，并且使用子采样技术可以有效地降低计算量。由于数据中一大部分都是正常值，而我们的目的只是区分异常值，所以只建立树的上半部分就可以了，而不需要建立整个树结构。因为降低了错检（swamping）和漏检（masking），所以使用一个很小的子采样来构建 iTrees 可以达到更好的效果。

2. iForest 不需要计算距离和密度，因此可以极大地降低计算开销。

3. iForest 具有低常数因子线性时间复杂度，低内存需求。在 iForest 之前，最好的方法虽然能达到近似的线性时间复杂度，但是内存开销很大。

4. iForest 具有纵向扩展（scale up）的能力，可以处理很大的具有很多不相关属性的高维度数据集。


# Isolation 与 Isolation Trees

在 iForest 或者 iTree 中，`孤立`（isolation）的意思就是*将一个实例与其他所有实例分离开来*。由于异常值具有”少而不同“的性质，所以他们更容易被孤立。如果使用一个树结构对数据进行随机划分（random partitioning），重复迭代这个过程知道所有的数据都被孤立。那么在树中，我们会看到，异常值在随机划分下的树中具有非常短的*平均*路径（即平均来说，通过有限的几次划分就可以将它们孤立起来，见 **Fig. 1-1**）。

![1-1 异常值 $x_o$ 比正常值 $x_i$ 更容易被孤立](/assets/blog-images/Anomaly-Detection-1.1.png)
**Fig. 1-1.** 异常值 $x_o$ 比正常值 $x_i$ 更容易被孤立。 

在 **Fig. 1-1** 中，划分通过随机选择数据的一个属性，然后对选择的属性在其最大和最小值间随机选择一个分割点完成。由于该递归过程可以用一个树结构来表示，所以孤立一个数据点所需的划分次数等于从树的根节点到叶子节点的路径长度。

由于每次划分都是随机产生的，因此在 iForest 的 iTrees 系综中，单个的树都是通过不同的随机划分方式得到。接着需要对某个特定的数据点（例如图中的 $x_i$ 和 $x_o$），计算它在这些树中的平均路径长度，图 **Fig. 2-1** 中显示了 $x_i$ 和 $x_o$ 的平均路径长度随着树的个数的收敛性。使用 1000 个 iTrees, 异常值 $x_o$ 的平均路径长度收敛到 4.02，正常值 $x_i$ 的平均路径长度收敛到 12.82 。异常值的平均路径长度小于正常值。

![1-2 $x_o$ 和 $x_i$ 的平均路径长度随着树的个数的收敛性](/assets/blog-images/Anomaly-Detection-1.2.png)
**Fig. 1-2.** $x_o$ 和 $x_i$ 的平均路径长度随着树的个数的收敛性。


## Isolation Tree

**定义：** 令 $T$ 为 Isolation Tree (iTree) 的一个节点。$T$ 或者是没有任何子节点的外部节点，或者是包含一个测试条件以及两个子节点（$T_l$, $T_r$）的内部节点。测试条件包含一个特征 $q$ 以及一个分裂值 $p$, $p < q$ 将数据分裂到 $T_l$ 和 $T_r$。


给定一个包含 $n$ 个实例的样本集 $X = \\{x_1, \cdots, x_n \\}$，其中每个实例 $x_i = (x_i^1, \cdots, x_i^d)$ 是包含 $d$ 个属性的变量。建立一个 iTree 的过程就是随机地选取一个属性 $q$ 以及一个分裂点 $p$, 将一个 $X$ 递归地分裂成两个 $X$ 的过程，直到：

1. 树达到了最高限制，

2. 或者 $\vert X \vert = 1 $， 

3. 或者 $X$ 中的数据点都具有相同的值。

iTree 是一个`真二叉树`（proper binary tree）。它的每一个节点要么不包含任何子节点，要么包含两个子节点。如果假定所有的实例都具有不同的值，那么最终树会分裂成包含 $n$ 个叶子节点的二叉树，其中每个叶子节点只包含一个实例。这时树的内部节点个数为 $n-1$。树包含的总的节点个数为 $2n-1$。因此 iTrees 的内存需求随着 $n$ 线性增长。

为了度量一个数据点的异常度，可以将数据根据它们的平均路径长度升序排列。异常点就是那些排在前面的数据点。下面给出路径长度和异常度的定义

## 路径长度与异常度

**路径长度：定义** 数据点 $x$ 的路径长度 $h(x)$ 由从树的根节点到 $x$ 的叶子节点遍历所经过的边的个数度量。

由于 iTrees 具有与二叉搜索树（Binary Search Tree, BST）具有等价的结构，所以平均  $h(x)$ 的估计等价于 BST 中搜索不成功的情况。给定一个包含 $n$ 个实例的数据集，BST 中非成功搜索的平均长度可以估计为：

$$
c(n) = 2H(n-1) - (2(n-1)/n)
$$

其中 H(i) 是一个谐波数，可以估计为 $\ln(i) + 0.5772156649$ （欧拉常数，Euler's constant）。由于 $c(n)$ 是给定 $n$ 的平均 $h(x)$, 所以可以用来归一化 $h(x)$。实例 $x$ 的异常度 $s$ 可以定义为：

$$
s(x,n) = 2^{-\frac{E(h(x))}{c(n)}}
\label{eq:iforest1}
$$

其中 $E(h(x))$ 为 $x$ 在一个 iTrees 集合中的 $h(x)$ 的平均值。我们有：

$$
S \to \left\{
  \begin{array}{ll}
    0.5   \qquad       & 当 E(h(X)) \to c(n)，\\ 
    1     \qquad       & 当 E(h(X)) \to 0，\\
    0     \qquad       & 当 E(h(X)) \to n-1。
  \end{array}
\right.
$$

$s$ 是 $h(x)$ 的单调函数。下图显示了 $E(h(x))$ 与 $s$ 的关系。

![1-3 一个数据点 $x$ 的平均路径长度 $E(h(x))$  与其异常度 $s$ 的关系](/assets/blog-images/Anomaly-Detection-1.3.png)
**Fig. 1-3.** 一个数据点 $x$ 的平均路径长度 $E(h(x))$  与其异常度 $s$ 的关系。

对于 $0 < h(x) \leq n - 1$, 有 $ 0 < s \leq 1$。使用异常度 $s$，可以有如下论断：

1. 如果一个实例的异常度 $s$ 趋向于 1， 那么它确定无疑是一个异常值。

2. 如果一个实例的 $s \ll 0.5$，那么可以安全地认为它为正常值。

3. 如果所有的实例都有 $s \approx 0.5$, 这意味着整个数据集不包含明显的异常值。


# iTrees 的特征


iForest 使用一个 iTrees 系综 （1）来确认具有平均较短路径长度的数据为异常点，（2）使用多个树来检测不同的异常点。由于 iForest 并不需要孤立所有的正常值，而正常值又占据了数据集中的大部分，所以 iForest 不用建立完整的 iTrees，只建立可以检测异常的部分模型就足够了。并且 iForest 使用少量的样本效果反而更好。而使用大样本的数据，一些异常值由于数据稠密，反而可能会被当作正常值，从而影响探测器的性能。对于大样本数据，可以使用子采样（sub-sampling）技术压缩样本量。


错检（swamping）和漏检（masking）是异常检测的两个重要问题。错检意味着将一个正常值当成了异常值，漏检意味着由于异常值的数目庞大，掩盖了它们自身的存在。当异常值形成的簇大且密集的时候，也增加了孤立每个异常点需要的分区数。iForest 可以使用子采样来建立非完全模型来缓解漏检和错检。

![1-4 使用子采样技术缓解漏检和错检](/assets/blog-images/Anomaly-Detection-1.4.png)
**Fig. 1-4.** 使用子采样技术缓解漏检和错检。



# 使用 iForest 进行异常检测

使用 iForest 进行异常检测分两个阶段 （1）对训练数据集进行子采样，并建立 iTrees，（2）将数据点喂给 iTrees, 得到其异常度。

## 训练阶段

训练阶段，递归地划分给定的训练集建立 iTrees，直到所有的实例都被孤立，或者使用非完全模型，直到树达到一定的高度。树的高度 $l$ 有子采样的大小 $\psi$ 自动给定 $l = \text{ceiling}(\log _2 \psi)$（近似为树的平均高度）。**Fig. 1-5** 和 **Fig. 1-6** 分别给出了 iForest 和 iTree 算法的伪代码。iForest 算法的输入包括两个参数，子采样的大小 $\psi$ 以及 iTree 的个数 $t$。

![1-5 iForest 算法](/assets/blog-images/Anomaly-Detection-1.5.png)
**Fig. 1-5.** iForest 算法。

![1-6 iTree 算法](/assets/blog-images/Anomaly-Detection-1.6.png)
**Fig. 1-6.** iTree 算法。



当子采样的大小 $\psi$ 增加到一定程度的时候，iForest 就可以可靠地用来探测异常值。进一步的增加 $\psi$ 并不能进一步的提高探测的性能，反而会增加计算量和内存的消耗。作者发现，对大多数数据集而言，使用 $\psi = 2 ^8$ （即 256） 就可以使探测器达到很好的性能。 对于树的个数 $t$, 作者发现，路径长度在 $t < 100$ 的时候就可以达到很好的收敛。IForest 的训练复杂度为  $O(t \psi \log \psi)$

## 测试阶段

计算一个数据的异常度需要计算它在一个 iTrees 系综里的平均路径长度 $E(h(x))$。下面的算法 $LathLenght$ 给出了计算一个数据 $x$ 在单个 $iTree$ 内的路径长度的计算方法。

当一个数据 $x$ 从根节点出发，到一个叶子节点结束时候，假设走过的边树为 $e$。如果该叶子结点包含的数据实例个数 $Size > 1$ (即树为一个非完全模型)，那么对 $x$ 的路径长度可以估计为 $e + c(Size)$。其中 $c(Size)$ 是树没有被完全建立的部分的调整参数。得到数据点 $x$ 在一群 iTrees 内的平均路径长度后，就可以使用公式 $\ref{eq:iforest1}$ 来计算该数据点的异常度。对数据点异常度估计过程的计算复杂度为 $O(nt \log \psi)$，其中 $n$ 为测试数据集的大小。



![1-7 PathLength 算法](/assets/blog-images/Anomaly-Detection-1.7.png)
**Fig. 1-7.** PathLength 算法。


# Reference

[^iForest1]: F. T. Liu, K. M. Ting, and Z. H. Zhou, "[Isolated Forest,](https://doi.org/10.1109/ICDM.2008.17)" in *2008 Eighth IEEE International Conference on Data Mining,* **2008,** pp. 413-422.

[^iForest2]: F. T. Liu, K. M. Ting, and Z.-H. Zhou, "[Isolation-Based Anomaly Detection,](https://doi.org/10.1145/2133360.2133363)" *ACM Trans. Knowl. Discov. Data,* vol. 6, pp. 1-39, **2012.**

[^Mass1]: K. M. Ting, G.-T. Zhou, F. T. Liu, and J. S. C. Tan, "[Mass estimation and its applications,](https://doi.org/10.1145/1835804.1835929)" in *Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and data mining,* Washington, DC, USA, **2010,** pp. 989-998.

[^Mass2]: K. M. Ting, G.-T. Zhou, F. T. Liu, and S. C. Tan, "[Mass estimation,](https://link.springer.com/article/10.1007/s10994-012-5303-x)" *Machine Learning,* vol. 90, pp. 127-160, **2013.**

[^RS-Trees]: S. C. Tan, K. M. Ting, and T. F. Liu, "[Fast anomaly detection for streaming data,](https://doi.org/10.5591/978-1-57735-516-8/IJCAI11-254)" in *Proceedings of the Twenty-Second international joint conference on Artificial Intelligence* - Volume 2, Barcelona, Catalonia, Spain, **2011,** pp. 1511-1516.




