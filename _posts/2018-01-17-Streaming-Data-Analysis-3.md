---
layout: post
title: 流数据聚类分析(3)：D-Stream 算法
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

与 [`DenStream算法`](https://htaoyu1.github.io/machine%20learning/2018/01/01/Streaming-Data-Analysis-1/)类似，`D-Stream 算法`也是一种基于密度的聚类方法。不同的是 DenStream 算法使用微簇来维护数据流中数据点的空间密度分布，而 D-Stream 算法则是基于网格结构来表征数据点的密度。D-Stream 的中心思想是将数据的每一个维度离散化成一个一个的小窗口。对于一个 $d$ 维的数据流，每个数据根据其每个维度的值最终被分配到一个 $d$ 维的网格内。每个网格包含的数据密度为其中包含的数据点个数的加权和。与 DenStream 类似，每个数据点的权重随着时间指数衰减。最后，D-Stream 基于网格包含的数据密度对这些网格聚类，得到最终的聚类结果。

D-Stream 算法也使用了 [`CluStream 算法`](https://htaoyu1.github.io/machine%20learning/2018/01/06/Streaming-Data-Analysis-2/)的两阶段时间框架，将聚类过程分为在线的`概要数据结构`生成过程和离线的根据数据概要生成聚类的过程。D-Stream 和 DenStream 都是为了克服 CluStream 的以下缺点：

1. CluStream 使用的是基于 k-means 的聚类算法，只能生成球形的聚类，而基于密度的聚类算法则可以生成任意形状的簇。

2. k-means 算法不能发现数据里面的噪声和离群点。

3. k-means 算法需要预先指定聚类的个数 $k$，因此聚类过程需要对数据的特征具有一定的领域知识。

# D-Stream 算法

![3-1 D-Stream 算法流程图](/assets/blog-images/Streaming-Data-Analysis-3.1.png)
**Fig. 3-1.** D-Stream 算法流程图。关于init\_clustering 和 adjust\_clustering 的算法请参考**聚类算法**部分。 

**Fig. 3-1** 显示了 D-Stream 算法的流程图。算法首先读入一个数据 $x$，然后寻找 $x$ 对应的密度网格（density grid） $g$，并更新 $g$ 的特征矢量（characteristic vector）。算法的离线部分在第一个 $gap$ 时间初始化聚类簇，然后每隔 $gap$ 时间移除 $grid\\_list$ 内的松散网格（sporadic grid），并动态地调整聚类簇。


# 密度网格（Density Grids）

![3-2 密度网格](/assets/blog-images/Streaming-Data-Analysis-3.2.png)
**Fig. 3-2.** 密度网格。

## 基本定义

假设流数据是 $d$ 维的 $x = (x_1, x_2, \cdots, x_d)$。将每个维度平均分成 $p_i$ 份，则整个数据空间可以被划分为 $N = \prod_{i=1}^d p_i$ 个高维的`密度网格`（density grids）。网格 $g$ 可以用它每一维的位置来表示：

$$
g = (j_1, j_2, \cdots, j_d)。
$$

例如考虑 **Fig. 3-2** 中的二维数据，红色网格就可以用 2 维矢量 $g = (4,4)$ 表示。

D-Stream 使用了衰减窗口模型。在时刻 $t$, 到达时间为 $T(x) = t_c$ 的数据 $x$ 的密度系数（density coefficient）为

$$
D(x, t) = \lambda^{t - T(x)} = \lambda^{t-tc}, \quad t \geq t_c
$$

其中 $\lambda \in (0,1)$ 为`衰减系数`（decay factor）。

**网格密度（Grid Density）：** 假设 $E(g, t)$ 为在 $t$ 时刻网格 $g$包含的所有数据。则定义网格 $g$ 在 $t$ 时刻的密度为：

$$
D(g, t) = \sum\limits_{x \in E(g, t)} D(x,t) = \sum\limits_{x \in E(g, t)} \lambda^{t - T(x)}
$$ 

在实践中我们不需要每个时刻都更新网格的密度，而只需要在有新数据到达一个网格 $g$ 时更新  $g$ 的密度。假设网格 $g$ 在 $t_n$ 时候吸收一个新数据，而 $g$ 上一次吸收数据的时间是 $t_l$, 则有

$$
D(g, t_n) = \lambda^{t_n - t_l} D(g, t_l) + 1
$$

**特征矢量（Characteristic Vector）：**网格 $g$ 的特征矢量定义为一个五元组 $(t_g, t_m, D, label, status)$。其中 $t_g$ 为网格最后一次更新的时间；$t_m$ 为网格 $g$ 最后一次被作为松散网格从 grid_list 内被移除的时间；$D$ 为网格在 $t_g$ 时刻的密度；$label$ 为网格所属的簇标签；$status = \\{SPORADIC, NORMAL\\}$ 是移除松散网格使用的标签。

## 基于密度的网格簇

根据前面网格密度的定义，我们可以知道，在有限的时间内，数据空间所有网格的密度不会超过 $1/(1-\lambda)$ 即

$$
\sum D(x,t) \leq \frac{1}{1-\lambda}, \qquad t = 1, 2, \cdots
$$

$$
\sum\limits_{t \to \infty} D(x,t) = \frac{1}{1-\lambda}
$$

由于整个空间被划分为 $N = \prod_{i=1}^d p_i$ 个网格，所以网格的平均密度 $\leq \frac{1}{N(1-\lambda)}$。



- **稠密网格（Dense Grid）：** 在时刻 $t$, 如果一个网格的密度满足下式，则我们称该网格为稠密网格：

  $$
  D(g, t) \geq \frac{C_m}{N(1-\lambda)} = D_m
  $$

  其中 $C_m > 1$ 为控制稠密网格临界值的参数。我们要求 $N > C_m$ 因为 $D(g,t)$ 的值不会超过$1/(1-\lambda)$。 

- **稀疏网格（Sparse Grid）：** 如果一个网格在 $t$ 时刻的密度满足

  $$
  D(g, t) \leq \frac{C_l}{N(1-\lambda)} = D_l
  $$

  则我们称该网格为稀疏网格。其中 $C_l \in (0, 1)$。


- **过渡网格（Transitional Grid）：** 在 $t$ 时刻，如果一个网格的密度满足

  $$
  \frac{C_l}{N(1-\lambda)} \leq D(g, t) \leq \frac{C_m}{N(1-\lambda)} 
  $$

  则我们称该网格为过渡网格。
  
  
- **松散网格（Sporadic Grid）：** 对于一个稀疏网格，如果网格内包含的数据点非常少，则称该网格为一个松散网格。
  
  > 怎么判断“少”这个量，会在后文解释。这里把定义放在这里是为了强调松散网格和稀疏网格不一样。一个稀疏网格不一定是松散的。因为一个稀疏网格内可能包含很多的数据点，但是由于该网格很久没有吸收新的数据，旧数据的权值不断衰减导致网格的密度低于临界值 $D_l$。这样的稀疏网格不是松散网格。但是反过来，一个松散网格一定是稀疏的。



- **邻接网格（Neighboring Grids）：** 如果网格 $g_1 = (j_1^{(1)}, j_2^{(1)}, \cdots, j_d^{(1)} )$ 和 $g_2 = (j_1^{(2)}, j_2^{(2)}, \cdots, j_d^{(2)} )$ 满足

  $$
  j_i^{(1)} = j_i^{(2)}, \quad i=1, \cdots, k-1, k+1, \cdots, d 
  $$

  $$
  \vert j_k^{(1)} = j_k^{(2)} \vert = 1
  $$

  其中  $1< k < d$, 则称 $g_1$ 和 $g_2$ 为在第 $k$ 维的邻接网格，记为 $g_1 \sim g_2$


- **网格组（Grid Group）：** 一个密度网格集合 $G = \\{g_1, \cdots, g_m \\}$，如果对于任意两个网格 $g_i, g_j \in G$, 存在一个网格序列 $g_{k_1}, \cdots, g_{k_l}$ 满足 

  $$
  g_{k_1} \sim g_{k_2}, g_{k_2} \sim g_{k_3}, \cdots, g_{k_{l-1}} \sim g_{k_l}
  $$

  其中 $g_{k_1} = g_i, g_{k_l} = g_j$, 则称 $G$ 为一个网格组。


- **内部网格与外部网格（Inside and Outside Grids）：** 考虑一个网格组 $G$ 以及 $G$ 内的一个网格 $g$，如果 $g$ 在所有维度上均有邻接网格，则称 $g$ 为 $G$ 的一个内部网格，否则 $g$ 为 $G$ 的一个外部网格。


- **网格簇（Grid Cluster）：** 令 $G = \\{g_1, \cdots, g_m \\}$ 为一个网格组，如果 $G$ 的每一个内部网格都是稠密网格，而 $G$ 的每一个外部网格要么是一个稠密网格，要么是一个过渡网格，则称 $G$ 为一个网格簇。


# D-Stream 算法的组成

在 D-Stream 算法中（**Fig. 3-1**），我们每隔 $gap$ 时间移除 $grid\\_list$ 内的松散网格（sporadic grid），并动态地调整聚类簇。 这里有三个问题需要解决：（1）如何选取 $gap$ 的值；（2）如何维护活跃的网格（active grids）列表；（3）如何产生簇。

## 网格检测的时间间隔

因为网格的密度是随着时间动态变化的。一个网格如果长期不吸收新的数据，随着时间的衰减，网格可能会退化为一个过渡网格或者稀疏网格。而一个稀疏网格由于吸收了一部分数据，则可能升级为一个过渡网格或者稠密网格。因此每隔一段时间，我们需要检查网格的密度并相应地调整网格簇。

检查网格密度的时间间隔 $gap$ 既不能太大，也不能太小。如果 $gap$ 太大，则不能很好地识别数据流的动态变化，如果 $gap$ 太小，则频繁的离线计算会增加机器的工作负载。这里有两个关键时间，第一个是一个稠密网格退化成稀疏网格所需的最短时间，另一个是一个稀疏网格成长为稠密网格所需的最短时间。$gap$ 的最优选择应该是这两者中值较小的一个。

**命题 1：** 一个稠密网格退化为稀疏网格所需的最短时间为

$$
\delta_o = \left\lfloor  \log_{\lambda} \left(\frac{C_l}{C_m}\right) \right\rfloor
$$ 

> 证明见原论文 p. 136


**命题 2：** 一个稀疏网格成长为稠密网格所需的最短时间为

$$
\delta_o = \left\lfloor  \log_{\lambda} \left(\frac{N -C_m}{N - C_l}\right) \right\rfloor
$$ 

> 证明见原论文 p. 137

所以 $gap$ 的最佳选择是

$$
\begin{eqnarray}
gap & = & \min \left\{ \delta_0, \delta_1 \right\} \\
    & = & \min \left\{ \left\lfloor  \log_{\lambda} \left(\frac{C_l}{C_m}\right) \right\rfloor, \left\lfloor  \log_{\lambda} \left(\frac{N -C_m}{N - C_l}\right) \right\rfloor \right\} \\
    & = & \left\lfloor \log_{\lambda} \left( \max \left\{  \frac{C_l}{C_m}, \frac{N-C_m}{N-C_l} \right\} \right) \right\rfloor
\end{eqnarray}
$$


## 检测并移除松散网格

使用密度网格方法的最大挑战在于数据维度很高时，需要的网格数量呈指数级增加。例如对于一个 $d$ 维的数据，如果没个维度被划分为 20 个窗口，那么就需要 $20^d$ 个网格。在实际应用中，并不是每个网格都能吸收数据。因此我们可以只为非空的网格分配内存空间。然而由于离群点和噪声的存在，有的网格仅仅包含有限的几个数据。对于这样仅仅包含有限个数据点的网格我们称之为`松散网格`（sporadic grids）。为了节省内存空间，我们需要周期性地检测并移除这些松散的网格。


一个密度 $D \leq D_l$ 的稀疏网格来源主要有两个方面：第一个是由于网格内的数据点本来就非常少导致网格密度很低，第二个是一个本来包含很多个数据点的网格由于长时间没有吸收数据，原有数据的权重因子不断衰减导致的网格密度降低。只有前者可以被称为松散网格，需要移除；而后者产生的稀疏网格则应该被保留。因为后则将来可能还有新的数据被吸收，有潜力成长为一个过渡网格或者稠密网格。D-Stream 算法定义了一个密度阈值函数来区分这两种网格。

**密度阈值函数（Density Threshold Function）：** 假定网格 $g$ 上一次更新的时间是 $t_g$, 则在时间 $t > t_g$, 密度阈值函数定义为

$$
\pi(t_g, t) = \frac{C_l}{N}\sum\limits_{i=0}^{t-t_g} \lambda^i = \frac{C_l (1 - \lambda^{t - t_g + 1})}{N(1-\lambda)}
$$

**命题 3：** 密度阈值函数 $\pi(t_g, t)$ 满足下列性质：

1. 如果 $t_1 \leq t_2 \leq t_3$, 那么有

   $$
   \lambda^{t_3 - t_2} \pi(t_1, t_2) + \pi(t_2 + 1, t_3) = \pi(t_1, t_3)
   $$

2. 如果 $t_1 \leq t_2$ 那么对任意 $t > t_1, t_2$ 有 $\pi(t_1, t) \geq \pi(t_2, t)$。



在 $t$ 时刻，我们说一个稀疏网格 $g$ 是松散的，如果

- **(S1)：** $D(g, t) < \pi(t_g, t)$；

- **(S2)：** $ t \geq (1 + \beta) t_m $ 如果 $g$ 在 $t_m$ 时刻曾经被删除过（$t_g$ 和 $t_m$ 的值被维护在 $g$ 的特征矢量里）。其中 $\beta > 0 $ 为一个常数。 

所有用来聚类的网格被维护在 $grid\\_list$ 中。$grid\\_list$ 是一个哈希表，使用双向链表来解决 hash 冲突。哈希表的键为网格的坐标，值为网格的特征矢量。


算法使用下面的规则从 $grid\\_list$ 中删除松散网格：

- **(D1)：** 在检测 $grid\\_list$ 中的网格时，所有满足条件 **(S1)** 和 **(S2)** 的网格被标记为 SPORADIC。但是此时并不删除这些网格，而是保留到下一个检查周期。

- **(D2)：**  在下一个检测周期，如果一个网格 $g$ 的标记为 SPORADIC，并且自从上一次检测后 $g$ 没有吸收任何数据，则将该网格从 $grid\\_list$ 删除。否则，重新检查 $g$ 是否满足 **(S1)** 和 **(S2)**， 如果满足，则保持 $g$ 的标记仍然为 $SPORADIC$ 不变，并且不做删除操作。如果不满足，则将 $g$ 的标签设置为 NORMAL。


一个值得思考的问题是，一旦一个松散网格被删除，就意味着这个网格的密度被重置为 0。如果后面又有数据被投影到这个网格，我们需要重新开始计数。虽然网格的删除机制可以有效地降低算法需要的内存空间，并节省聚类时间。但是这样的删除是否非影响到聚类的效果？换句话说，一个松散网格后面可能又吸收数据成长为一个过渡网格或者稠密网格，但是算法的删除机制是否会影响到我们对这个过程的判断？万幸的是，前面我们设置了一个临界阈值函数 $\pi(t_g, t)$ 并定义了删除规则，按照这个规则，一个过渡网格或者稠密网格并不会因为松散网格的移除而被错误地删除。


为了证明这个问题，我们需要定一个**完全密度函数（complete density function）** $G_a(g, t)$。这个函数表示一个网格 $g$ 的精确密度，即 $g$ 当前维护的密度与之前所有被删除密度的和。使用这个定义，我们可以证明下面这个重要结论：*一个松散网格的删除操作并不会影响后面它吸收数据成长为一个过渡网格或者稠密网格。*




**命题 4：** 假设 $g$ 最后一次被当作松散网格删除的时间为 $t_m$， $g$ 最后一次吸收数据的时间为 $t_g, (t_g \geq t_m + 1)$，如果在当前时刻 $t$ 有 $G(g, t) < \pi (t_g, t)$，那么我们有 $D_a(g, t) < \pi (0, t) < D_l$。

> 证明见原论文 p. 138

这个命题的涵义是，如果一个网格 $g$ 曾经因为密度低于密度阈值函数被删除过（密度清零）。但是后来  $g$ 又吸收了新的数据重新出现在 $grid\\_list$ 中。在当前时间 $t$，如果 $g$ 的密度又一次小于了临界密度阈值函数，此时就算把之前被删除的数据全部考虑进来，$g$ 的密度也不会超过 $D_l$。即此时删除$g$ 不会导致一个过渡网格或者稠密网格被错误地删除。


**命题 5：** 假定网格 $g$ 在 $t$ 时刻的密度为 $D(g, t)$，并且 $g$ 在 $t+1$ 和 $t+gap$ 之间没有吸收任何数据，那么存在 $t_0 > 0$ 和 $t_1 > 0$ 满足：

1. 对 $t > t_0$, 如果 $D(g,t) < D_l$, 则 $D_a(g, t+gap) < D_l$。

2. 对 $t > t_1$, 如果 $D(g,t) < D_m$, 则 $D_a(g, t+gap) < D_m$。

> 证明见原论文 p. 139


这个命题的涵义是，只要系统运行的时间足够长，则一定会满足条件 $t >t_0$ 和 $t > t_1$。再考虑我们删除一个网格的策略 **(D1)** 和 **(D2)**：一个被标记为 SPORADIC 的网格并不会被立即删除，而是等待一个 $gap$ 周期，在这个周期内没有新的数据被吸收，我们才对其删除。该命题就是策略 **(D1)** 和 **(D2)**的理论依据。

综合考虑 **(S1)**、**(S2)**、**(D1)** 和 **(D2)**。根据**密度阈值函数**的定义，松散网格的阈值随着时间的增加逐渐靠近 $D_l$。而一个松散的网格 $g$ 之前可能被删除过，这些被删除的数据点权重不断地衰减，假定它们从来没有被删除，也只是给 $g$ 的权重增加一个很小的量。就算考虑到这个小量的贡献， 根据**命题 4**，$g$ 也是一个稀疏网格。一个松散的网格定义为密度低于**密度阈值函数**给定阈值的网格，因此它的密度也一定小于 $D_l$, 根据规则**(D1)** 和 **(D2)**，该网格被删除时，其完全密度也一定小于 $D_l$。因此**(S1)**、**(S2)**、**(D1)** 和 **(D2)**一块保证了被删除的网格肯定是稀疏的。总之，上面的结果表明，在一个初始化阶段之后，删除松散的网格不会影响聚类的结果。

## 聚类算法

![3-3 init\_clustering 算法的流程图](/assets/blog-images/Streaming-Data-Analysis-3.3.png)
**Fig. 3-3.** init\_clustering 算法的流程图。


![3-4 adjust\_clustering 算法的伪代码](/assets/blog-images/Streaming-Data-Analysis-3.4.png)
**Fig. 3-4.** adjust\_clustering 算法的伪代码。


# Reference

1. Y. Chen and L. Tu, "[Density-based Clustering for Real-time Stream Data,](https://doi.org/10.1145/1281192.1281210)" in *Proceedings of the 13th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining,* San Jose, California, USA, **2007,** pp. 133-142.


