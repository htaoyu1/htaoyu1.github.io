---
layout: post
title: 异常检测（3）：HS-Trees 
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


# 高维数据的质量估算

在论文[^Mass1]中作者给出了一维数据的质量估算方法，并且将算法初步拓展到了高维数据的情况。在论文[^mMass] 中，作者又提出了一种新的质量估算的方法。该新方法主要有两个创新点：

1. 将一维数据的质量估算拓展到了高维的情况。

2. 将算法的时间复杂度从 $O(\psi^h)$ 降低到了 $O(\psi h)$，其中 $\psi$ 是样本的大小，$h$ 是质量估算的阶数。

在一维数据质量估算的原始算法中[^Mass1]，需要计算分裂点 $s_i$ 的概率 $p(s_i)$（见 [异常检测（2）：Mass Estimation](https://htaoyu1.github.io/machine%20learning/2018/02/11/Anomaly-Detection-2/)）。新的质量估算方法提出了一种随机化的版本来对数据点的质量进行近似的估计，消除了计算 $p(s_i)$ 的必要性[^mMass] [^Mass2]。

新版本的质量估算非常简单，对数据在其特征空间进行随机分割，完成分割后，数据点 $x$ 的质量就等于其所处随机空间内数据点的个数。对数据进行 $c$ 次随机分割，可以得到 $x$ 的 $c$ 个质量估计，$x$ 的质量可以近似估计为这 $c$ 个质量的平均值。 对于一阶质量估算，使用一个2元分割将数据点划分到两个不相交的空间。对于 $h$ 阶质量估算，需要将数据划分到 $2^h$ 个不相交的空间。 

令 $\mathbf{x} \in \mathcal{R}^d$，$T^h(\mathbf{x})$ 为 $2^h$ 个空间中数据点 $\mathbf{x}$ 所在的空间。$T^h(\cdot)$ 表示计算使用的是整个数据集 $D$；$T^h(\cdot \vert \mathcal{D})$ 表示计算使用的是 $D$ 的一个随机子集 $\mathcal{D} \subset D$。$m$ 表示数据点 $\mathcal{x}$ 所在空间中实例的个数。则广义的质量基函数可以定义为：

$$
\mathbf{m}\bigl(T^h(\mathbf{x})\bigr)= \left \{ \begin{array}{l@{\quad}l} m & \mbox{if } \mathbf{x} \mbox{ is in a region of $T^{h}$ having $m$ instances} \\ 0 & \mbox{otherwise}\\ \end{array} \right .
$$  

对于一维数据的情况，数据质量的近似估计可以表示为：

$$
\begin{eqnarray}
mass(x,1) = \sum\limits_{i=1}^{n-1} m_i(x)p(s_i) & \approx & \frac{1}{c} \sum\limits_{k=1}^c \mathbf{m}\bigl(T^1_k(x)\bigr) \\

mass(x,h) & \approx & \frac{1}{c} \sum\limits_{k=1}^c \mathbf{m}\bigl(T^h_k(x)\bigr) \\

\overline{mass}(x,h) & \approx & \frac{1}{c} \sum\limits_{k=1}^c \mathbf{m}\bigl(T^h_k(x \vert \mathcal{D}_k)\bigr) 
\label{eq:mass1}

\end{eqnarray}
$$

这里每个 $T^h_k$ 都是以等概率随机产生的。

因为 $T^h$ 是在多维空间中定义的，所以公式 $\ref{eq:mass1}$ 可以直接拓展到多维的情况

$$
\overline{mass}(x,h)  \approx  \frac{1}{c} \sum\limits_{k=1}^c \mathbf{m}\bigl(T^h_k(\mathbf{x} \vert \mathcal{D}_k)\bigr) 
\label{eq:massd}
$$

与一维数据类似，多维数据的质量估计也规定了一个从核心点到边缘点的序。下图显示了二维数据质量分布的等高线图。从图可以看出，对于高维数据，可以使用高阶的质量估算来刻画多峰分布。

![3-1 二维数据质量分布的等高线图](/assets/blog-images/Anomaly-Detection-3.1.png)
**Fig. 3-1.** 二维数据质量分布的等高线图。 

# 使用 Half-Space Trees 进行质量估算

## Half-Space Tree

Half-Space Tree 的概念是基于以下的观测事实：对于均匀分布的数据，相同大小的区域包含的数据质量是一样的，数据质量的大小与区域的形状无关。

![3-2 使用 Half-Space 分割（a）均匀质量分布（b）非均匀质量分布 ](/assets/blog-images/Anomaly-Detection-3.2.png)
**Fig. 3-2.** 使用 Half-Space 分割（a）均匀质量分布（b）非均匀质量分布。 

**Fig. 3-2(a)** 显示了通过递归性地三层平均划分将数据空间分为 $8=2^3$ 个等大小 half-spaces 的例子。注意到每次划分都是随机地选取一个属性对数据进行分割，所以这些区域的形状是不一样的。这种二元 half-space 划分保证每次分割都是生成两个等大小的子区域。在均匀分布下，划分后每个子区域数据的质量都是未分割前母区域数据质量的一半。这种特征可以使我们很容易地比较任意两个区域的关系。例如在 **Fig. 3-2(b)** 中，区域的数据质量的序提供了区域内数据异常度的序。


**定义 6：** Half-Space Tree 是一个二叉树，其中每个内部节点将空间拆分成两个等大小的区域，每个外部节点终止了对数据空间进行进一步拆分。 所有节点都记录了其区域的训练数据的质量。


令 $T^h[i]$ 是深度为 $i$ 的 Half-Space Tree；$\mathbf{m}(T^h[i])$（简记为 $\mathbf{m} [i]$）为第 $i$ 级的其中一个区域的质量。任何两个区域之间的关系可以使用相对于 Half-Space Tree 深度 = 0（根节点）处的 $\mathbf{m}[0]$ 来表示。例如在均匀分布下，有：

$$
\mathbf{m}[0] = \mathbf{m}[i]\times 2^i
$$

对任意两个  $i$ 和 $j$ 级的区域（$\forall x \neq y$），有如下关系：

$$
\mathbf{m}[i]\times 2^i = \mathbf{m}[j]\times 2^j
$$

在非均匀分布下，下面的关系建立了不同级别的区域之间的一个序：

$$
\mathbf{m}[i]\times 2^i < \mathbf{m}[j]\times 2^j
$$

使用上述性质对实例排序，并可以为 Half-Space Tree 定义 `增广（augmented）质量`：

$$
S(\mathbf{x}) = \mathbf{m}[\ell]\times 2^\ell
\label{eq:augmass}
$$

其中 $\ell$ 是一个外部结点的深度，该外部结点包含测试实例 $x$ ，并且总共包含 $m[\ell]$ 个实例。


仅当 Half-Space Tree 的所有外部节点具有相同的深度时，才使用 $m[\ell]$ 来估计质量。 如果外部节点具有不同的深度，则估计是基于增广质量 $m[\ell]×2^\ell$ 的。 下面是这两种情况下的 Half-Space Tree 的表述：


![3-3 HS-Tree 的两种变体](/assets/blog-images/Anomaly-Detection-3.3.png)
**Fig. 3-3.** HS-Tree 的两种变体。

- **HS-Tree：** 仅基于质量。 该 HS-Tree 构建了一个平衡二叉树结构，它在每个内部节点进行 half-space 空间分割，并且所有外部节点具有相同的深度。 记录落入每个外部节点的训练实例的数量，并将其用于质量估计（**Fig. 3-3(a)**）。 

- **HS$^{\ast}$-Tree：** 基于增广质量。与 HS-Tree 不同，第二个变种 HS$^{\ast}$-Tree 的外部节点具有不同的深度。方程 $\ref{eq:augmass}$ 定义了 HS$^{\ast}$-Tree 树的质量估计。


在 HS$^{\ast}$-Tree 的特殊情况下，当一个外部结点包含的训练数据个数为 1 时才停止树的生长。这时质量估计仅取决于深度，即 $2^{\ell}$ (或简记为 $\ell$）。换句话说，当外部节点的大小限制设置为 1时，深度级别成了 HS$^{\ast}$-Tree 中的质量的代理（**Fig. 3-3(b)**）。

由于这两个变体具有相似的性能，所以接下来我们只关注 HS$^{\ast}$-Tree，因为它构建的树比较小，可以节省训练时间和内存空间。而 HS-Tree 则可能会生成许多零质量的分支。

## Half-Space Trees 算法

### 训练阶段

![3-4 Algorithm 1](/assets/blog-images/Anomaly-Detection-3.4.png)

**Algorithm 1** 显示了如何生成一个 HS-Tree：首先为每个维定义一个（随机）范围，以形成覆盖所有训练数据的`工作空间`（work space）。**Algorithm 1** 中的 InitialiseWorkSpace（·）函数执行如下。对于每个属性 $q$ ，在范围 $[\mathcal{D}min_q，\mathcal{D}max_q]$ (即子样本集 $\mathcal{D}$ 中属性 $q$ 的最小和最大值) 内**随机**选择一个分割值 $z_q$，然后，工作空间的属性 $q$ 被定义为具有范围 $[min_q，max_q] = [z_q-r，z_q + r]$，其中 $r = 2 \cdot \max(z_q-\mathcal{D}min_q，\mathcal{D}max_q-z_q)$。所有维度的范围定义了用于生成 HS-Tree 的工作空间。然后将 $[min_q，max_q]$ 定义的工作空间传递给 **Algorithm 2** 以构造 HS-Tree。

![3-5 Algorithm 2](/assets/blog-images/Anomaly-Detection-3.5.png)

除了每个节点不需要选择拆分标准外，构造 HS-Tree 的过程与构造普通决策树几乎相同。
给定一个工作空间，随机选择一个属性 $q$ 形成一个 HS-Tree 的内部节点（ **Algirhtm 2**第4行）。该内部节点的分裂点是由工作空间（第5行）定义的属性 $q$ 的最小和最大值（即，$min_q$ 和 $max_q$ ）之间的中点。数据通过两个分支之一进行过滤，具体取决于数据位于分支的哪一侧（第6-7行）。对于每个分支（**Algorithm 2** 第9-12行）重复节点的构建过程，直到达到大小或深度限制，形成外部节点（**Algorithm 2** 的1-2行）。在深度为 $\ell$ 的外部节点处的训练实例个数形成了在 $\mathbf{x}$ 的测试过程中要使用的质量 $\mathbf{m}(T^h(\mathbf{x} \vert \mathcal{D}))$。

上述方法使用一个随机子样本 $\mathcal{D}$ 来构建一个 HS-TRee（即，$T^h(\cdot \vert \mathcal{D})$）。使用不同的随机子样本（无替换采样）可以构建多个 HS-Tree 以形成系综。

### 测试阶段

在测试阶段，一个测试实例 $\mathbf{x}$ 将从根节点到外部节点遍历系综中的每个HS-Tree，外部节点记录的质量被用来计算 $\mathbf{x}$ 的增广质量：

$$
s(\mathbf{x},h) = \mathbf{m} \bigl(T^h(\mathbf{x}|\mathcal{D})\bigr) \times 2^{\ell}
$$

对系综中的所有 HS-Tree 进行的操作，取其平均值作为对 $\mathbf{x}$ 的增广质量的最终估计：

$$
\overline{\mathit{mass}}( \mathbf{x},h) \approx\frac{1}{c} \sum_{k=1}^c s_k(\mathbf{x},h)
$$

### 时间和空间复杂度
由于不涉及评估或搜索，HS-Tree 可以快速生成。 另外一个优点是，对于一个给定大小为 $n$ 的数据集， HS-Tree 可以使用一个更小的子样本（大小为 $\psi$）生成（$\psi \ll n$）。 对于具有固定子样本大小 $\psi$，树最大深度为 $h$，树的个数为 $c$ 的系综而言， HS-Trees 系综具有训练时间复杂度 $O(c h \psi)$。 测试过程的时间复杂度为 $O(chn)$。 HS-Trees 系综的空间复杂度为 $O(ch\psi)$。

# 流数据的异常检测

论文[^HS-Trees] 给出了一个使用流式 HS-Trees 进行流数据异常检测的算法。与现有的异常检测算法相比，该算法具有以下特点：

- 首先，该算法只需要对数据进行一次处理，所需的内存是常数级的。

- 其次，流式 HS-Trees 是一个单类别的异常检测器，当数据流包含大量正常数据时非常有用。

- 第三，它模型更新速度快，以便在处理随时间变化的数据分布时可以保持高的检测精度。 它的模型更新简单快捷是因为它在处理流数据时不需要修改树结构。

与其他决策树（例如随机森林）不同，流式 HS-Trees 不会从实际的训练实例中归纳（induce）其树结构。 相反，仅使用数据空间维度来构造树结构。 因此树可以被快速构建，并且可以在流数据到达之前就完成好部署。 实验研究表明，实用流式 HS-Trees 系综可以生成一个强大而准确的异常检测器，该检测器对参数设置不敏感。

## 算法概述

为了对流式数据进行异常检测，需要将流划分为相同大小的窗口（每个窗口包含固定数目的数据点）。系统需要维护两个连续的窗口，即参考窗口和最新的窗口。在异常检测初始阶段，算法在参考窗口学习数据的质量分布。然后，通过学习到的数据质量画像来推断随后到达的最新窗口中数据的异常分数 - 落在高质量子区域中的新数据被认为是正常值，而落在低质量或空子区域中的数据为异常值。随着最新窗口中数据的不断到来，新数据的质量画像也被记录下来。当最新的窗口已满时，新数据的质量画像将覆盖参考窗口中旧数据的质量画像；因此参考窗口将始终存储最新的数据质量画像，以对下一批数据进行打分。完成参考窗口数据质量画像的更新后，最新窗口会清除其存储的数据质量画像，并准备捕获下一批即将到达的数据。


## 在 HS-Trees 中记录数据的质量画像

为了进行异常检测，我们需要首先使用正常值构建一个 HS-Trees 系综（值得一提的是，在之前的算法中我们使用数据在每个维度的最大和最小值来初始化工作空间，在流数据的 HS-Trees 工作空间初始化中，我们可以将数据的所有属性都归一化到范围 [0,1]， 这样构建 HS-Trees 的时候就可以只使用数据的维度信息来建立模型了，细节见论文[^HS-Trees]中 p.1513 的算法1），建立数据集的质量画像（mass profile）。下面的 **Algorithm 3** 显示了这样一个过程。该过程涉及使用窗口中的每个实例遍历系综模型中的每个 HS-Tree。 参考窗口中的实例将更新质量 $r$; 最新窗口中的实例更新质量$l$。 

![3-6 Algorithm 3](/assets/blog-images/Anomaly-Detection-3.6.png)


## 异常度打分

HS-Tree 每个分区中的质量来描述数据的特征。假设 $m[i]$ 是 HS-Tree 的深度为 $i$ 的 half-space 分区中的质量。在均匀质量分布下，两个深度分别为 $i$ 和 $j$ 的分区之间的质量值有关系 $m [i] \times 2^i = m [j] \times 2^j$。当数据分布不均匀时，下面的不等式建立了不同深度的分区之间的一个序：$m [i] \times 2^i < m [j] \times 2^j$。我们将使用这个属性来排序异常度。


假设 $Score(x，T)$ 是将一个测试实例 $x$ 从 HS-Tree（T）的根节点遍历直到某个叶子节点的函数。该函数将通过估计 $Node^{\ast}.r \times 2^{Node^{\ast}.k}$ 的值来返回 $x$ 的异常分，其中$Node^{\ast}.k$ 是包含 $Node^{\ast}.r$ 实例的叶子节点的深度。这里，叶子节点（$Node^{\ast}$）是达到 HS-Tree 最大深度的节点，或者包含 $sizeLimit$ 个甚至更少实例的节点。

$x$ 的最终得分是系综中每个 HS-Tree 打分的总和：

$$
\sum\limits_{T \in HS-Trees} Score(x, T)
$$

## 流式 HS-Trees

![3-7 Algorithm 7](/assets/blog-images/Anomaly-Detection-3.7.png)

**Algorithm 4** 显示了流式 HS-Tree 的操作过程。第1行首先建立了一个 HS-Trees 的系综。第2行使用流中的前 $\psi$  个实例作为参考窗口，在 HS-Trees 中建立数据的初始参考质量画像。由于这些实例来自初始参考窗口，因此只有被遍历结点的 $r$ 值会更新。完成这两个步骤之后，就可以使用该模型为接下来到达的流数据进行异常打分了。

质量 $r$ 用于计算每个流数据点的异常分数（第8行）。最新窗口中到达的流数据的质量被记录在 $l$ 里上（第9行）。在每个窗口的末尾，需要进行模型更新。更新过程很简单：在下一个窗口开始之前，只需将非零质量的 $l$ 传递给  $r$ 就可以了（第14行）。这个过程很快，因为它不涉及模型的结构变化。在此之后，将所有 $l$ 的值重置为零（第15行）。

## 时间和空间复杂度

**Algorithm 4** 主循环中的四个关键操作是：打分（第8行），更新质量（第9行），更新模型（第14行）和重置模型（第15行）。对于前两个操作，每个实例都从树的根部遍历至叶子节点（即，$O(h)$）；最后两个操作，每个最多访问 $\psi$ 节点，但在整个流上只发生 $n/\psi$次。因此,对 $n$ 个流数据的**平均分摊时间复杂度**为 $O(t(h + 1))$；当在流数据之间执行模型更新和重置时（最坏的情况）复杂度为 $O(t(h + \psi))$。当 HS-Tree 的最大深度 $h$，系综中树的个数$t$，和窗口中数据点的个数 $\psi$ 固定时，这些时间复杂度都是常数。

在流式 HS-Trees 中，每个到达的实例处理后就被丢弃，然后再处理下一个实例。因此是一种使用有限内存来处理无限数据流的单遍扫描算法。 HS-Trees 的空间复杂度是 $O(t2^h)$，当 $t$ 和 $h$ 固定时，也是一个常数。

# Reference

[^Mass1]: K. M. Ting, G.-T. Zhou, F. T. Liu, and J. S. C. Tan, "[Mass estimation and its applications,](https://doi.org/10.1145/1835804.1835929)" in *Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and data mining,* Washington, DC, USA, **2010,** pp. 989-998.

[^mMass]: K. M. Ting and J. R. Wells, "[Multi-dimensional Mass Estimation and Mass-based Clustering,](https://doi.org/10.1109/ICDM.2010.49)" in *2010 IEEE International Conference on Data Mining,* **2010,** pp. 511-520.

[^Mass2]: K. M. Ting, G.-T. Zhou, F. T. Liu, and S. C. Tan, "[Mass estimation,](https://link.springer.com/article/10.1007/s10994-012-5303-x)" *Machine Learning,* vol. 90, pp. 127-160, **2013.**


[^HS-Trees]: S. C. Tan, K. M. Ting, and T. F. Liu, "[Fast anomaly detection for streaming data,](https://doi.org/10.5591/978-1-57735-516-8/IJCAI11-254)" in *Proceedings of the Twenty-Second international joint conference on Artificial Intelligence* - Volume 2, Barcelona, Catalonia, Spain, **2011,** pp. 1511-1516.




