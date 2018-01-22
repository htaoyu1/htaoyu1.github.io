---
layout: post
title: 流数据聚类分析(4)：EDMStream 算法
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

Alex Rodriguez 和 Alessandro Laio 于 2014 年在 Science 上发表了`基于密度山峰的聚类算法`(Density Peak-based Clustering, `DPCluster`)[^DPCluster]。算法主要解决了如何决定聚类簇中心的问题。DPCluster 算法认为，聚类簇的中心应该是簇中数据最密集的地方（即密度山峰）。算法定义了两个关键变量：（1）数据点 $i$ 的`局部密度`（local density） $\rho_i$；（2）：$i$ 与比其局部密度高的其它数据点的最短距离 $\delta_i$。算法根据这两个变量产生一个`决策图`（decision graph），该决策图让用户对数据集的分布有了一个整体的印象。用户可以根据该决策图手动选择聚类簇中心。选择的标准是使用 $\rho$ 和 $\delta$ 值都非常大的数据点作为簇中心。该聚类方法的论文已经被翻译成中文[^DPClusterCN]。DPCluster 算法的优点是比较稳定，数据集的微小变化不影响聚类的结果。我之前的项目需要对分子模拟产生的环肽构象进行聚类。使用一些传统的算法，比如 k-means, 每次添加新的数据到数据集就会导致聚类结果的剧烈变化（当然，这个也跟我们选取的距离度量有一定的关系）。

> 基于密度的聚类算法, 比如 DBSCAN， 也可以有比较好的稳定性。但是请考虑这样一个场景，两个数据密集的区域被一个低密度区连在一起。如果 DBSCAN 的参数选取的不合理，就可能导致这两个区域被划分为一个簇。但是直观上我们可能希望将这两个区域分为两个不同的簇。当然，我们可以增加 DBSCAN 的 MinPts 参数将这两个区域分开。但是如果数据集的其他地方又有一个簇，簇的最大密度都比 MinPts 要低的话，DBSCAN 可能会遗漏这个簇。 DPCluster 在数据点的局部密度之外引入了一个距离参数 $\delta$，簇的中心由这两个参数共同决定。一个 $\rho$ 值低的数据点如果 $\delta$ 值够大，也可以考虑做为一个簇中心。虽然 DPCluster 算法每次都要在决策图上手动选取簇的中心，不够自动化，但是这个过程同时也给了我们很大的灵活性，用户可以根据自己数据的特点灵活地决定如何聚类。

DPCluster 算法可以算是我最熟悉的一种聚类方法。在我之前的论文中大量使用了该聚类算法。`EDMStream` (Evolution of Density Mountain Stream) 算法[^EDMStream] 将 DPCluster 算法改进为可以应用于流数据场景的聚类算法。DPCluster 算法需要计算所有数据点之间的距离，对于大数据的场景，需要维护一个非常大的矩阵。在我们的一篇论文中[^CPPrediction]，我们通过首先使用 PCA 技术对数据降维，然后在一个二维或者三维的空间将数据网格化，将对原始数据点的聚类问题改为对带密度属性的网格的聚类，有效地降低了 DPCluster 算法的计算量。EDMStream 也是首先对数据点进行粗粒化，类似于 DenStream 算法，将一大批数据点综合为一个细胞，称为簇胞（cluster cell），对数据的聚类改为对簇胞的聚类。一个簇胞具有三个性质：（1）簇胞的种子，定义了簇胞的中心位置；（2）簇胞的密度，综合刻画了簇胞包含的数据点的个数以及他们的新鲜度；（3）依赖距离，刻画了一个簇胞与其他簇胞的依赖关系。

EDMStream 基于簇胞的三个性质，构建了一个树状结构来维护这些簇胞，以及他们之间的依赖关系。流聚类问题就成了在这个树中发现最大强依赖子树的问题。随着时间的变化，由于流中不断有数据进来，所以这个树是动态变化的，EDMStream 提出了一系列的方法来维护这个树结构，包括如何更新簇胞的密度，簇胞间的依赖关系，如何使用过滤策略来筛选需要更新依赖关系的树节点，以降低计算量，什么时候可以抛弃一些过时的数据和离群点，以及如何自适应地调节参数，以反映流中数据分布的变化等。


# DP 聚类与 DP 树

## DP聚类算法

DP 算法是基于两个观测事实：（1）一个簇的中心经常被包围在一群局部密度低的数据点中间；（2）簇的中心距离其他具有高局部密度的数据点相对较远。基于这两个观测，DP 算法定义了两个度量：（1）一个数据点的局部密度（local density）$\rho$; (2) 一个数据点离另一个比它密度高的数据点的距离 $\delta$。DP 算法使用这两个度量来决定簇的中心（即密度山峰）。

**局部密度（local density）：** 一个数据点 $p_i$ 的局部密度 $\rho_i$ 定义为其 $d_c$ 邻域内的数据点个数：

$$
\rho_i = \left\vert \{ p_j ~~ \middle| ~~ \vert p_i, p_j \vert < d_c  \} \right\vert
$$

其中 $\vert p_i, p_j \vert$  表示数据点 $p_i$ 和 $p_j$ 之间的距离，$d_c$ 为界段距离（cutoff distance）。$\rho_i \leq \xi $ 的数据点被当作离群点，其中 $\xi$ 为一个预定义的参数。


**依赖距离（dependent distance）：** 一个数据点 $p_i$ 的依赖距离 $\delta_i$ 定义为 

$$
\delta_i = \min\limits_{j:p_j > p_i} (\vert p_i, p_j\vert)
$$ 

即所有其他局部密度比 $p_i$ 高的数据点离 $p_i$ 的最短距离。假设 $p_j$ 是离 $p_i$ 最近的局部密度高于 $p_i$ 的数据点，那么我们就称 $p_i$ 依赖于 $p_j$。

![4-1 DPCluster 聚类算法](/assets/blog-images/Streaming-Data-Analysis-4.1.png)
**Fig. 4-1.** DPCluster 聚类算法。 


考虑 **Fig 4-1** 所示的二维数据。数据点 1 和 10 具有很高的局部密度和依赖距离，所以可以当作簇的中心。数据点 8 虽然也具有很高的局部密度，但是因为它离局部密度更高的点（点 7）很近，所以导致 8 的依赖距离很小。而点 7 距离比它密度更高的数据点 1 也很近，所以 7 的依赖距离也很小。图中 1 的局部密度最大，数据集中找不到比它密度更大的点，那么怎么定义它的依赖距离呢？DPCluster 算法将该点的依赖距离定义为它离数据集中其他所有点的最大距离。

同理，我们可以知道图 **Fig 4-1** 中红色数据点的 13 虽然局部密度比较大，但是依赖距离较小。而点 10 在红色簇中的局部密度最大，所以它的依赖点应该位于蓝色的簇中。所以 10 的依赖距离也比较大。黑色的点 26， 27， 28 虽然具有很大的依赖距离，但是局部密度很低，所以被当作离群点。

> 注：定义局部密度最大的数据点的依赖距离为该点离数据集中其他点的最大距离有一个缺陷。考虑数据集中两个数据点 $p_k$ 和 $p_l$, 如果 $\rho_k = \rho_l$ 且均为数据集中局部密度最大的数据点，那么它们的依赖距离 $\delta_k$ 和 $\delta_l$ 均为各自离数据集中最远点的距离。这样就会导致 $k$ 和 $l$ 的 $\rho$ 和 $\delta$ 都很大。按照 DPCluster 算法，这两个点都会被选为簇中心。如果这两个点碰巧属于同一个簇的话，DPCluster 算法会将这两个簇分成两个。Peghoty[^DPClusterCN] 详细讨论了这个缺陷，并提出了一种解决方案。即，首先将数据点按照其局部密度排序，这样就算 $k$ 和 $l$ 密度一样，总要排一个先后顺序。假设 $l$ 排在 $k$ 的后面，那么计算 $k$ 的依赖距离的时候，依照 DPCluster 算法找数据集中离 $k$ 最远的点；计算 $l$ 的依赖距离的时候，因为 $k$ 排在它的前面，所以要计算 $k$ 和 $l$ 之间的距离当作 $l$ 的依赖距离。这样如果 $k$ 和 $l$ 离得很近的话，$\delta_l$ 的值会很小，这样 $l$ 就不会再被单独作为一个簇中心考虑。


如果一个数据点 $p_i$ 离它的依赖点 $p_j$ 的距离（即 $p_i$ 的依赖距离） $\delta_i \leq \tau$, 那么我们称  $p_i$ `强依赖`（strongly dependent）于 $p_j$；反之，称 $p_i$ `弱依赖`（weakly dependent）于 $p_j$。

假设存在一个数据点序列 $\\{p_1, p_2, \cdots, p_n\\}$，且 $p_i$ 强依赖于  $p_{i+1}$ （$1 \leq i \leq n-1$）。最终的数据点 $p_n$ 不强依赖于数据集中的其他任何点，那么我们称 $p_n$ 为这些数据点的 `强依赖根`（strongly dependent root）。点 $p_i$ $(1 \leq i \leq j-1)$ `依赖可达`（dependency-reachable）点 $p_j$ $(i+1 \leq j \leq n)$。


根据之前的描述，DPCluster 算法中的簇可以定义如下：

**定义 1：簇**：令 $P$ 为一个数据点集合，簇 $C$ 为 $P$ 的一个非空子集，满足

- **最大性**（maximality）：如果点 $p \in C$, 那么任意一个依赖可达 $p$ 的非离群点 $q$ 也属于簇 $C$。

- **可追溯性**（Traceability）：对任意 $p_1, p_2, \cdots \in C$，它们具有相同的强依赖根，即 $C$ 中的密度山峰。   



## 依赖树（Dependency Tree, DP-Tree）

上面的讨论我们可以看到，DP 的聚类过程可以通过追踪依赖链（dependency chain）达到。为了使用 DPCluster 算法实现流数据聚类，我们需要一个有效的数据结构来维护数据点之间的依赖关系。由于每个数据点只依赖于其他一个数据点（依赖根除外），因此可以使用树结构来维护依赖关系。这个树就被称为依赖树（DP-Tree）。下图显示了 **Fig 4-1** 中数据点的依赖树。

![4-2 依赖树](/assets/blog-images/Streaming-Data-Analysis-4.2.png)
**Fig. 4-2.** 依赖树。 

**Fig. 4-2** 可以看到，树的下方局部密度低于 $\xi$ 的部分为离群点。数据点 10 弱依赖于 5。我们定义，如果一个子树 $T_i$ 内所有的数据点都是强依赖的，那么我们称 $T_i$ 为一个`强依赖子树`（strongly dependent subtree）。如果 $T_i$ 不是其他任何强依赖子树的子树，那么我们称 $T_i$ 为一个`最大强依赖子树`（Maximal, Strongly Dependent SubTree, `MSDSubTree`）。例如 **Fig. 4-2** 中包含所有蓝色数据点的子树就是一个最大强依赖子树。所有包含红色数据点的子树是另一个最大强依赖子树。

**定义 2：基于 DP-Tree 的聚类算法** 基于 DP-Tree 的聚类算法是找到数据集中所有的最大强依赖子树，每个最大强依赖子树相应于一个簇。强依赖子树的根相应于该簇的聚类中心。

## DP 聚类 vs. DBSCAN

DBSCAN 定义的簇满足两个标准：*最大性*和*连通性*。DP 聚类定义的簇满足*最大性*和*可追溯性*。这两类算法都描述了两个数据点之间的可达（reachable）性质，数据点的可达性质依赖于数据点的密度信息。然而 DBSCAN 算法的连通性描述的是密度连通的性质，是*对称的*；而 DP 算法的可追溯性描述的是密度依赖的性质，是*非对称的*。因此 DBSCAN 的密度连通关系可以抽象成一个无向图，图的每一个连通部分可以看成一个基本的簇；而 DP 的密度依赖关系可以抽象成一个树状结构（DP-Tree），每一个 MSDSubTree 可以看作一个簇。DP 聚类算法就是在 DB-Tree 中找 MSDSubTree 的问题。**Fig. 4-3** 显示了 DBSCAN 与 DP 算法的不同点。

![4-3 DBSCAN vs. DP](/assets/blog-images/Streaming-Data-Analysis-4.3.png)
**Fig. 4-3.** DBSCAN vs DP。

# 问题陈述

## 基本概念

### 数据流

数据流  $S$ 是一个包含时间戳信息的数据序列 $S^N = \\{p_i^{t_i}\\}_{i=1}^N$, 其中 $N \to \infty$, $t_i$ 为数据点  $p_i \in \mathbb{R}^d$ 的到达时间的时间戳信息。

### 衰减模型

与 DenStream 类似，DP 算法也使用的衰减窗口模型。一个数据点的权重随着时间指数衰减。在 $t$ 时刻，数据点 $p_i$ 的权重为

$$
f_i^t = \alpha^{\lambda (t-t_i)}
$$

其中 $t_i$ 维数据点 $p_i$ 的到达时间，$\alpha$ 和 $\lambda$ 为衰减控制参数。$\lambda$ 的值越大，数据衰减的越快。论文作者选取的衰减参数为 $\alpha = 0.998, \lambda = 1$，因此 $f_i^t \in (0, 1]$。


假定 $\\{p_j \vert t_j < t, \vert p_i, p_j \vert < d_c \\}$ 为在点 $p_i$ 的 $d_c$ 邻域内的数据点集合， 则 $p_i$ 的局部密度定义为这些点的权重因子之和：

$$
\rho_i^t = \sum\limits_{p_j : t_j < t, \vert p_i, p_j \vert < d_c } f_j^t
$$

衰减模型意味着，（1）如果没有新的数据到达一个数据点的周围，则该数据的局部密度随着时间持续衰减；（2）所有的数据点以相同的速率衰减，即衰减函数 $\mathcal{D}^t()$ 在任意时刻 $t$ 被作用到当前流数据集 $S^n$中的所有数据点

$$
\mathcal{D}^t(S^n) = \{f_1^t, \cdots, f_n^t\}
$$ 

### 流聚类

**定义 3：流聚类：** 在衰减窗口模型下，给定流数据 $S^N$ 以及它们的衰减性 $\mathcal{D}^t(S^n)$ 流聚类 $\mathcal{C}^t()$ 在任意时刻 $t_1, \cdots, t_N$ 返回一系列不相交的簇。也就是，对任意 $n \in [1, N]$, 有 $\mathcal{C}^{t_n} (S^n, \mathcal{D}^{t_n}(S^n)) = \\{C_1^{t_n}, C_2^{t_n}, \cdots, C_{k^{t_n}}^{t_n}, C_o^{t_n}\\}$，其中 $C_i^{t_n} (1 \leq i \leq k^{t_n})$ 是 $t_n$ 时刻 $S^n$ 的一个子集，其中 $C_o^{t_n}$ 是 $t_n$ 时刻 $S^n$ 中的离群点集合，$k^{t_n}$ 是 $t_n$ 时刻簇的个数，$S^n = C_1^{t_n} \cup C_2^{t_n} \cup \cdots \cup C_{k^{t_n}}^{t_n} \cup C_{o}^{t_n} $，对任意 $i$ 和 $j$ 有 $C_{i}^{t_n} \cap C_{j}^{t_n} = \emptyset$, $C_{i}^{t_n} \cap C_{o}^{t_n} = \emptyset$。

### 簇演化 

簇是连续演化的，即 $\mathcal{C}^{t_n} (S^n, \mathcal{D}^{t_n}(S^n)) \neq \mathcal{C}^{t_{n+1}} (S^{n+1}, \mathcal{D}^{t_{n+1}}(S^{n+1}))$。即簇的个数，每个数据点的簇标签都有可能变化，下图显示了簇的五种演化类型。

![4-4 簇的演化类型](/assets/blog-images/Streaming-Data-Analysis-4.4.png)
**Fig. 4-4.** 簇的演化类型。

emerge 意味着新簇的出生，disapper 代表着旧簇的消亡，split 表示簇的分裂，merge 意味着簇的合并，adjust 表示簇的调整，意味着（1）一些数据点从一个簇迁移到另一个簇；（2）一些离群点变成可达点并被一些簇吸收；（3）簇内的一些边缘点演化成离群点。 

## 流聚类的概要数据结构


**定义 4：簇胞（cluste-cell）** 在 $t$ 时刻，一个簇胞 $c$ 是对一系列距离相近的数据点的一个概要数据，它通过一个三元组来描述 $\\{ s_c, \rho_c^t, \delta_c^t \\}$，其中

- $s_c$ 为簇胞 $c$ 的种子点（seed point）。以 $s_c$ 为种子点的簇胞 $c$ 概要了一个数据点的集合，该集合中的数据点离 $s_c$ 的距离小于它们离其他种子点的距离。并且该集合中的数据点到 $s_c$ 的距离不大于一个预定义的半径 $r$。即 $P_c = \\{ p_i : s_c = \arg \min\limits_{s_k \in S_{seed}} (\vert p_i, s_k \vert), \vert p_i, s_c\vert \leq r \\}$。其中 $S_{seed}$ 为数据集中的所有种子点的集合。

- $\rho_c^t$ 为簇胞内所有数据点在 $t$ 时刻的即时权重和， 定义为

  $$
  \rho_c^t = \sum\limits_{p_i \in P_c} f_i^t
  \label{eq:EDMStream6}
  $$
  
- $\delta^t_c$ 为簇包 $c$ 的依赖距离。即 $c$ 的种子点 $s_c$ 离其他所有密度高于 $c$ 的簇胞的种子点的最短距离。
  
  $$
  \delta_c^t = \min\limits_{c':\rho_{c'}^t > \rho_c^t} (\vert s_c, s_{c'} \vert)
  \label{eq:EDMStream7}
  $$
  
  在 DP 流聚类算法中，簇胞作为基本的聚类单元取代了原始的数据。即在 DP-Tree 中，一个树节点不再是一个数据点，而是一个簇胞。如果使用基于数据点的 DP-Tree，每当有新的数据点到来，或者旧的数据点消亡，都要增加或者删除树的节点。这样会造成很大的计算开销。而使用基于簇胞的 DP-Tree，一个新的数据点到来可能会导致增加一个簇包，也可能只是增加一个旧簇胞的密度；相反地，一个旧数据点的消亡会导致一个簇胞被删除或者只是降低一个簇胞的密度。

## 基本思想

### 使用 DP-Tree 进行流聚类

基于 DP-Tree 的流聚类就是在一个动态的 DP-Tree 中寻找所有最大强依赖子树（MSDSubTree）的问题。

### 使用 DP-Tree 进行簇的演化追溯

簇的演化可以通过监测 DP-Tree 的变化进行追溯。

- 簇的出生／消亡可以通过发现新产生／消亡的 MSDSubTree 进行追溯。

- 簇的分裂可以通过发现一个 MSDSubTree 是否分裂为两个或多个 MSDSubTree 进行追溯（即一个活着多个 MSDSubTree 的依赖距离超过了 $\tau$）。

- 簇的合并可以通过发现两个或多个 MSDSubTree 合并为一个 MSDSubTree 进行追溯。

- 簇的调整可以通过以下三种方式追溯：（1）来自于同一个 MSDSubTree 的多个簇胞被链接到其它 MSDSubTree；（2）多个簇胞的密度超过 $\xi$，并且被链接到包含它们的依赖簇胞的 MSDSubTree 内；（3）多个簇胞的密度衰减到小于 $\xi$，并被从他们原本属于的 MSDSubTree 中移除。

# EDMStream 

## 算法概述

![4-5 EDMStream 算法概述](/assets/blog-images/Streaming-Data-Analysis-4.5.png)
**Fig. 4-5.** EDMStream 算法概述。

### 存储结构

EDMStream 算法设计了两个关键的存储结构（**Fig. 4-5**）:

1. DP-Tree。 DP-Tree 为抽象密度山峰的数据结构。树的每一个节点为一个簇胞。

2. 离群数据池（outlier reservoir）。离群池保存了即时密度相对较低的簇胞，这些簇胞暂时不属于任何任何簇，不参与集类。但是如果有新的数据点被这些簇胞吸收，则它们可能被移动到 DP-Tree 中。另一方面，DP-Tree 中的簇胞密度随着时间的衰减，也可能会被移动到离群池中。离群池中的簇胞包括两种类型：（1）簇胞只包含有限的数据点，（2）簇胞包含很多的数据点，但是随着时间，这些数据点已经过时了。

### 关键操作

EDMStream 依赖于 **Fig. 4-5** 中的四种关键操作。

1. 新数据点分配。一个新的数据点或者被分配给一个现有的簇胞（可以是 DP-Tree 的簇胞，也可以是离群池中的簇胞），或者产生一个新的簇胞。满足下面两个条件的一个数据点 $p_i$ 被分配给一个现有的簇胞 $c$：

   1）. $p_i$ 离簇胞 $c$ 的种子 $s_c$ 的距离不超过 $r$, 即 $\vert p_i, s_c \vert \leq r$;

   2）. $s_c$ 是离 $c$ 最近的种子。即 $s_c = \arg \min\limits_{s_k \in S_{seed}} (\vert p_i, s_k\vert)$。其中 $S_{seed}$ 是所有簇胞的种子的集合。
   
   如果找不到这样的簇胞 $c$, 则以 $p_i$ 为种子，创建一个新簇胞，并放入离群池。
   
   
2. 依赖更新。由于一些簇胞的密度随着时间会退化，另外一些簇胞可能会吸收一些新的数据点导致密度的增加，因此需要更新 DP-Tree 来反映这些变化。

3. 簇胞出生（DP-Tree 节点插入）。离群池的簇胞可能会吸收一些新的数据点导致其密度的增加。对于这样的簇包，当其密度大于临界值的时候，需要将其插入到 DP-Tree 中。

4. 簇包消亡（DP-Tree 节点删除）。一个簇胞的即时密度随着时间会衰减，当一个簇胞的密度低于临界值的时候，需要从 DP-Tree 中删除，并临时放入离群池。

### 簇的演化追溯

上面的 2， 3， 4 操作涉及到簇的演化。我们可以通过监测 DP-Tree 的结构来追溯这种演化（包括依赖距离的变化 - 将导致 MSDSubTRee 的分裂或合并，簇胞的插入和删除，以及簇胞在不同 MSDSubTree 之间的迁移）。DP-Tree 的更新操作以及操作的时间需要被记录下来以供将来查询。

### 初始化

开始阶段，一部分吸收数据点的簇胞首先被放入缓存。一旦被缓存的簇胞的密度超过了临界值，则分别使用公式 \ref{eq:EDMStream6} 和 \ref{eq:EDMStream7} 计算每个被缓存簇胞的密度和依赖距离。同时，检索每个簇胞的依赖簇胞，并构建 DP-Tree 的初始化结构。进一步地，给定 $\tau$, 可以得到一个初步的聚类结果。

## 依赖更新（DP-Tree 更新）

### 密度更新

如果一个簇胞在时间 $t_j$ 和 $t_{j+1}$ 之间吸收一个新的数据点，则通过下式更新其密度：

$$
\rho^{t_j + 1}_c = \alpha^{\lambda(t_{j+1} - t_j)} \rho_c^{t_j}  + 1
\label{eq:EDMStream8}
$$

### 依赖更新

假设一个簇胞 $c$ 的密度 $\rho_c$ 增加并超过了它原本的依赖簇胞的密度，这就需要根据公式 \ref{eq:EDMStream7} 更新 $c$ 的依赖簇胞以及依赖距离 $\delta_c$。同时，$c$ 也有可能成为其他簇胞的新依赖。因此，每次 $c$  吸收了新的数据点，就需要更新一大批簇胞的依赖，这就会造成很大的计算开销。

令 $\rho_{c}^{t_j}$ 为 $c$ 在 $t_j$ 时刻的密度，$F_c^{t_j} = \\{ c' \vert \rho_c^{t_j} < \rho_{c'}^{t_j} \\}$ 为 $t_j$ 时刻密度大于 $c$的密度的簇胞集合。从 DP-Tree 的观点来看，$F_c^{t_j}$ 为树中级别比 $c$ 高的节点的集合。我们定义 $D_c^{t_j}$ 为 $c$ 在 $t_j$ 时刻的依赖簇胞（dependent cluster-cell）：

$$
D_c^{t_j} = \arg \min\limits_{c':c' \in F_c^{t_j}} \vert s_c, s_{c'}\vert
\label{eq:EDMStream9}
$$

因为所有的簇包都以相同的速度衰减，所以在 $t_j$ 到 $t_{j+1}$ 时刻，除了吸收数据的簇包 $c'$，其他簇胞密度的顺序并不会改变。对于每个簇胞 $c$，我们只需要判断更新的 $c'$ 是否新出现在 $F_c^{t_{j+1}}$ 中。如果 $c'$ 不在 $F_c^{t_{j}}$ 内 但是出现在了 $F_c^{t_{j+1}}$ 中，则需要更新  $c$ 的依赖。这是因为根据公式 \ref{eq:EDMStream9}，只要 $F_c$ 不变，$c$ 的依赖 $D_c$ 就不会变。从 $c'$ 的角度来看，只有之前密度大于或者等于 $\rho_{c'}$ （ $\rho_{c}^{t_j} \geq \rho_{c'}^{t_j}$ ）但是现在密度小于 $\rho_{c'}$ ($\rho_{c}^{t_{j+1}} < \rho_{c'}^{t_{j+1}}$) 的簇胞需要更新依赖。从 DP-Tree 的角度来看，我们需要找到树中之前级别比 $c'$ 高，但是现在级别低的节点。并更新他们的依赖。因此为了减小计算的开销，需要进行密度过滤。下面定理给出了密度过滤策略。

**定理 1：密度过滤器** 假定另一个簇胞 $c'$ 在 $t_{j+1}$ 时刻吸收了一个数据点，对于另一个簇胞 $c$, 如果

$$
\rho_c^{t_j} < \rho_{c'}^{t_j} \qquad 或者 \qquad \rho_c^{t_{j+1}} \geq \rho_{c'}^{t_{j+1}}
$$

则有 
 
$$
D_c^{t_j} = D_{c}^{t_{j+1}}
$$

即对于这样的 $c$, 不需要更新它的依赖。

> 证明见论文 p. 7

![4-6 使用密度过滤器选择需要更新依赖的簇胞](/assets/blog-images/Streaming-Data-Analysis-4.6.png)
**Fig. 4-6.** 使用密度过滤器选择需要更新依赖的簇胞。**定理 1** 一个直观的理解是：只有被簇胞 $c'$ 密度反超的簇胞需要更新依赖。考虑图中的簇包 $c'$, 密度从 $t_j$ 时刻（橙色虚圆圈）到 $t_{j+1}$ 时刻（橙色实圆圈）升高了。那么所有被 $c'$ 反超的红色簇胞的依赖都需要考虑更新。而那些黑色的簇胞不需要更新。即所有密度落在两个灰色虚线之间的簇胞都要考虑更新依赖。注意：从 $t_j$ 到 $t_{j+1}$ 时刻，所有其他簇胞的密度都在衰减，为了清晰起见，示意图并没有显示这种变化。





**定理 2： 三角不等式过滤器** 假定另一个簇胞 $c'$ 在 $t_{j+1}$ 时刻吸收了一个数据点 $p$，如果

$$
\left\vert \vert p, s_c \vert - \vert p, s_{c'}\vert \right\vert > \delta_c^{t_j}
$$

则有

$$
D_c^{t_j} = D_{c}^{t_{j+1}}
$$

即对于这样的 $c$, 不需要更新它的依赖。

> 证明见论文 p. 7

![4-7 使用三角不等式过滤器选择需要更新依赖的簇胞](/assets/blog-images/Streaming-Data-Analysis-4.7.png)
**Fig. 4-7.** 使用三角不等式过滤器选择需要更新依赖的簇胞。**定理 2** 一个直观的理解是：根据三角不等式，点 $p$ 到 $c$ 和  $c'$ 的距离之差肯定小于 $c$ 到 $c'$ 之间的距离（灰色虚线）。如果 $\left\vert \vert p, s_c \vert - \vert p, s_{c'}\vert \right\vert > \delta_c^{t_j}$, 那么肯定也有 $\vert s_c, s_c' \vert > \delta_c^{t_j}$，所以不需要更新 $c$ 的依赖。


## 簇胞出生于消亡

DP-Tree 中的簇胞被称为`活跃的`（active）簇胞， 离群池中的簇胞被称为`非活跃的`（inactive）簇胞。一个非活跃的簇胞可能吸收数据点而进入活跃状态。

假定数据点到达的时间是匀速的。即 $t_{i+1} - t_i $ 对任意 $i$ 都相等，数据流速率为 $v = \frac{1}{t_{i+1} - t_i}$。对于这里使用的衰减模型，我们有

$$
\lim_{n\to \infty}\sum\limits_{i=1}^n (\alpha^{\lambda (t_n - t_i)}) = \frac{v}{1 - \alpha^{\lambda}}
$$

即流中所有数据权重的和不会超过一个常数。所以我们定义一个簇胞 $c$ 在 $t$ 时刻是活跃的，如果他的密度 $\rho_c^t \geq \frac{\beta \cdot v}{1 - \alpha^{\lambda}}$。这里 $\beta  < 1$ 是一个可调参数，用来控制活跃簇胞的临界阈值。一个活跃簇胞的密度满足 $\frac{\beta \cdot v}{1 - \alpha^{\lambda}} \leq \rho_c^t \leq \frac{ v}{1 - \alpha^{\lambda}}$。因为一个新到达的数据点如果单独生成一个簇胞，我们认为这个簇胞是非活跃的，所以有 $\rho_c^t = 1 < \frac{\beta \cdot v}{1 - \alpha^{\lambda}}$。从而我们得到 $\beta$ 的取值范围 $\frac{1 - \alpha^{\lambda}}{v} < \beta < 1$。

如果一个簇胞 $c$ 因为密度 $\rho_c^t < \frac{\beta \cdot v}{1 - \alpha^{\lambda}}$，而被移入离群池，则所有依赖于 $c$ 的密度也肯定低于这个阈值，所以不用再对他们的密度进行判断，或者更新他们的依赖距离。如果一个离群池内的簇胞吸收数据变成活跃簇胞，则需要将该簇胞插入 DP-Tree 并更新依赖。可以使用**定理 1** 和**定理 2** 首先进行簇胞过滤。

## 内存回收

在实践中，我们需要删除那些长期不吸收数据的非活跃簇胞以减小内存的消耗。如果一个非活跃簇胞没有吸收数据的时间超过了一个新簇胞从开始形成到变成活跃簇胞的时间，则可以安全地删除该非活跃簇胞。我们称这种簇胞为`过时簇胞` (outdated cluster-cells)。

**定理 3：** 假定数据流的速率 $v$ 为常数，我们可以安全地删除一个非活跃簇胞而不会造成任何负面的影响，如果该非活跃簇胞没有吸收数据的时间超过了 $\Delta T_{del}$：

$$
\Delta T_{del} > \frac{\log_a(1-\alpha^{\lambda}) - \log_a(\beta \cdot v)}{\lambda \cdot v}
$$

> 证明见论文 p. 7


离群池需要维护的簇胞的个数不会超过 $\Delta T_{del} \cdot v + \frac{1}{\beta}$。

> 在 $\Delta T_{del}$ 时间内, 数据流共产生了 $\Delta T_{del} \cdot v$ 个数据点。假设每个数据点都形成一个单独的簇胞，则共形成了 $\Delta T_{del} \cdot v$ 个簇胞。过了  $\Delta T_{del}$ 时间，一些旧的非活跃簇胞开始被删除。所以离群池维护这样的簇胞不会超过 $\Delta T_{del} \cdot v$ 个。另外，一些活跃的簇包也可能因为密度衰减而变称非活跃的。假设所有活跃簇胞的密度和为 $\frac{v}{1 - \alpha^{\lambda}}$，而一个活跃簇包的密度至少为 $\frac{\beta \cdot v}{1 - \alpha^{\lambda}}$, 所以活跃簇胞的总个数不会超过 $\frac{v}{1 - \alpha^{\lambda}} \big/ \frac{\beta \cdot v}{1 - \alpha^{\lambda}} = \frac{1}{\beta}$。综合起来考虑，离群池需要维护的簇胞的个数不会超过 $\Delta T_{del} \cdot v + \frac{1}{\beta}$。


# $\tau$ 的自适应调节

参数 $\tau$ 控制着簇的分离度和颗粒度。$\tau$ 值越大，EDMStream 产生的簇的个数越少。流中数据分布随着时间在变化，所以 $\tau$ 需要设置为一个可以动态调整的参数，以适应数据的演化。当数据流中的数据点比较松散的时候，需要使用较大的 $\tau$，反之亦然。DPCluster 算法使用决策图来帮助用户选取合适的 $\tau$ 值，而这样的策略并不适合流数据算法。因此需要设计一套可以自动调整 $\tau$ 的方法。

聚类中一个常用的优化策略是最小化簇内（intra-cluster）距离同时最大化簇间（inter-cluster）距离。相似地，这里我们的策略是最小化平均`相对内依赖距离`（relative intra-dependent-distance） $\frac{\sum_{c:\delta_c \leq \tau} \delta_c /\overline{\delta} }{m}$ 同时最大化平均`相对间依赖距离`（relative inter-dependent-distance）$\frac{\sum_{c:\delta_c > \tau} \delta_c /\overline{\delta} }{n}$。这里 $m$ 为`内簇包`（intra-cluster-cells）的个数 $m = \vert \\{ c : \delta_c \leq \tau \\}\vert$，$n$ 为`间簇包`（inter-cluster-cells）的个数 $n = \vert \\{ c : \delta_c > \tau \\}\vert$，$\overline{\delta}$ 为平均依赖就 $\overline{\delta} = \sum_c \frac{\delta_c}{m + n}$。考虑到时间因素，作者提出的目标是对流聚类，最小化下面的估值函数：

$$
\mathcal{F}(\tau^t) = \alpha \cdot \frac{\sum\limits_{c:\delta_c^t > \tau^t} \delta_c^t}{n \cdot \overline{\delta}} + （1 - \alpha） \cdot \frac{m \cdot \overline{\delta}}{\sum\limits_{c:\delta_c^t \leq \tau^t} \delta_c^t}
$$

其中  $0 < \alpha < 1$ 为平衡参数， 反映了优化的偏好是更注重最小化平均相对内依赖距离还是最大化平均相对间依赖距离。$\alpha$ 意味着用户对聚类颗粒度的偏好。

如果 $\tau^t$ 是一个较大的值，那么平均内依赖距离和间依赖距离也是相对较大的值，这时 $\frac{m \cdot \overline{\delta}}{\sum_{c:\delta_c^t \leq \tau^t} \delta_c^t}$ 会变得非常大，将会得到少量的很大的簇。当 $\tau^t$ 是一个较小的值，那么平均内依赖距离和间依赖距离也是相对较小的值，这时 $\frac{\sum_{c:\delta_c^t > \tau^t} \delta_c^t}{n \cdot \overline{\delta}}$ 会变得非常大，将会得到大量的很小的簇。因此，一个合适的 $\tau$ 趋向于最小化 $\mathcal{F}(\tau^t)$。

另一个问题是如何选取合适的 $\alpha$。$\alpha$ 通过学习用户在初始化时如何从决策图中选取簇中心的偏好得到。这里使用下面的启发式方法来估计 $\alpha$。在初始化阶段，等到缓存了一定数量的簇胞形成了一个初始的 DP-Tree，首先根据这些簇胞的 $\rho$ 和 $\delta$ 画一个决策图，让用户选择簇中心。假设用户选择了一些依赖距离至少为 $\tau^0$ 的密度山峰。给定了 $\tau^0$，我们就可以找到 $\alpha = \widehat{a}$ 使得对任意 $\delta \neq \tau^0$，有 $\mathcal{F}(\widehat{a}, \tau^0) < \mathcal{F}(\widehat{a}, \delta)$。给定了反应用户偏好的 $\alpha$ 值，在任意时刻 $t$，$\tau^t$ 的值可以通过优化 $\mathcal{F}(\tau^t)$ 自动得到。


# Reference

[^DPCluster]: A. Rodriguez and A. Laio, ["Clustering by Fast Search and Find of Density Peaks,"](http://science.sciencemag.org/content/344/6191/1492) *Science*, vol. 344, pp. 1492-1496, **2014.**

[^DPClusterCN]: Peghoty, ["发表在 Science 上的一种新聚类算法。"](http://www.cnblogs.com/peghoty/p/3945653.html)

[^EDMStream]: S. Gong, Y. Zhang, and G. Yu, ["Clustering Stream Data by Exploring the Evolution of Density Mountain,"](https://arxiv.org/abs/1710.00867) *arXiv:*1710.00867, **2017.**

[^CPPrediction]: H. Yu, and Y.-S. Lin ["Toward Structure Prediciton of Cyclic Peptides,"](http://pubs.rsc.org/-/content/articlelanding/2015/cp/c4cp04580g/unauth#!divAbstract) *Physical Chemistry Chemical Physics,* vol. 17, pp. 4210-4219, **2015.**




