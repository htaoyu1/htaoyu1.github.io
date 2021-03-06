---
layout: post
title: 流数据聚类分析(1)：DenStream 算法
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

# 流数据

流数据是一种按照时间顺序不间断地大量到达的、快速变化的、潜在无限的数据。流数据有一下几个特点：

1. 数据量十分庞大，数据量随着时间急剧增加；

2. 数据按照时间顺序连续到达；

3. 由于内存的有限性，不可能存储整个数据集，只能存储数据的汇总信息；

4. 大部分流数据本质上是多维度的，多层的数据，需要多维多层次的处理。

日常生活中有很多流数据的例子，比如股票价格信息，环境监测数据，手机通话记录，网站点击数据，以及传感器网络数据等。

流数据的收集和挖掘过程是同时进行的，因此必须以最快的速度，从不断到来的数据中挖掘出感兴趣的模式，以响应实时查询的需求。由于收集时间和处理速度的有限性，流数据的处理在精确度上要求比较低，因此**结果的近似性**是流数据处理不同于传统数据挖掘的一个特点。

流数据挖掘的特点决定了它比传统的数据挖掘要复杂。流数据挖掘算法必须以增量的方式跟踪数据流的变化，同时不能占用过多的计算和内存资源。在流数据的挖掘中，一般要考虑以下几个问题：

1. 由于数据量的无限增长，对数据的处理仅限单次扫描（single scan）。除非刻意保存，每个数据只能被处理一次。

2. 算法对数据的处理速度不能低于数据流动的速度。

3. 由于数据量的无限性以及内存的有限性，无法把所有的数据都存放在内存中处理，所以算法要考虑怎么充分利用有限的内存资源处理更多的数据。算法的空间复杂度通常要求在 $O(\text{poly}(\log N))$ 的范围内。

4. 数据量的无限性导致算法无法保留原始数据信息，只能在内存中维护原始数据的概要，最终的分析结果通过概要产生。

5. 由于数据的产生环境是动态变化的，数据底层的分布模型也会随着时间不断改变，因此通常需要多个概念模型来处理流数据。




# 流数据模型

流数据可以看作是一个不断增长的据集合 {$ \\{\mathbf{X}\_1, T_1 \\}, \cdots, \\{\mathbf{X}\_i, T_i \\}, \cdots$}。其中 $\mathbf{X}\_i = (x_i^1, x_i^2, \cdots, x_i^d)$  是一个 $d$ 维的数据。$T_i$ 为数据对应的时间戳, 且对任意 $i < j$ 有 $T_i < T_j$

对流数据的挖掘通常基于某种特定的时间区间（又叫窗口）进行。按照数据流里数据的重要程度，可以将流数据划分为三种不同的子模型，分别为：界标窗口模型（landmark window model）、滑动窗口模型（sliding window model）、以及衰减窗口模型（damped window model）。

![1-1 常用的三种窗口模型](/assets/blog-images/Streaming-Data-Analysis-1.1.png)
**Fig. 1-1.** 常用的三种窗口模型。 


## 界标窗口模型
`界标窗口模型`（landmark window model）中，被处理的数据为从某个界标时间（$t_s$）开始到当前时间（$t_c$）之间的所有数据。窗口大小随着数据流不断地增加。

## 滑动窗口模型
`滑动窗口模型`（sliding window model）中，被处理的数据为到当前时间为止，最近到达的 $N$ 个数据。这里 $N$ 为滑动窗口的大小。窗口位置在时间轴上随着数据流不断地滑动。

## 衰减窗口模型
`衰减窗口模型`（damped window model）中，被处理的数据为从数据流开始，到当前时间之间包含的所有数据。各个数据根据其时间戳被赋予不同的权重。最新的数据具有最高的权重，并且数据的权重随着时间的流逝不停地衰减（例如时间戳为 $T_i$ 的数据在 $t_c$ 时刻，以指数方式衰减的权重为 $f(t) = 2^{-\lambda (t_c - T_i)} $, $\lambda > 0$）。

实际中选用那种模型根据要研究问题的性质而定。无论采用哪种模型，流数据的挖掘都采用相同的框架。在框架中，流数据挖掘算法在内存中维护一个`概要数据结构`（synopsis data structure）。算法不断从数据流中接受数据，当一个新数据到达时，算法通过`增量计算`（incremental computation）更新数据概要，当收到挖掘请求时，算法从概要中提取信息。一般而言，直方图，抽样，小波，哈希等都是构造概要数据结构的常用基本算法。


# DenStream 算法

由于流数据是动态变化的，就算对同一个系统，在不同的时间段内，由于产生数据流的底层模型的改变，采集到的数据会具有不同的特征。因此流数据对聚类算法提出了如下要求：

1. 算法不能对自然簇的个数做任何假设。

2. 算法可以产生任意形状的簇。

3. 算法具有处理离群点的能力。

由于数据流的动态性，新簇会不断地出现，旧簇会不断地消亡。一个理想的流聚类算法应该能区分代表新簇的数据点和噪声点。

## 基本概念

我们知道，数据流应用有三种常用的窗口模型：界标窗口模型，滑动窗口模型，和衰减窗口模型。`DenStream 算法`
使用了衰减窗口模型。在该模型下，数据点的权重随着时间指数衰减 $f(t) = 2^{-\lambda t}$，$\lambda > 0$。$\lambda$ 的取值越大，历史数据对当前数据的影响越小。数据流的整体权重为一个常数 $W = v(\sum_{t=0}^{t_c} 2^{-\lambda t}) = \frac{v}{1-2^{-\lambda}}$。其中 $t_c$ $(t_c \to \infty)$ 为当前时间，$v$ 为数据的流速（单位时间内到达的数据点个数）。 

## 核心微簇

`核心对象`(core object)：如果一个对象 $\epsilon$ 邻域中包含的数据点的权重和不小于某个正整数 $\mu$, 那么该对象就被称为一个核心对象。

`密集区域`（density-area）：各核心对象 $\epsilon$ 邻域的并集。

`核心微簇`（core-micro-cluster，简记为 c-micro-cluster）：对一组时间戳为 $T_{i_1}, T_{i_2}, \cdots, T_{i_n}$ 的相互临近的数据点 $p_{i_1}, p_{i_2}, \cdots, p_{i_n}$, 在 $t$ 时刻的定义为 $CMC(w, c, r)$。其中 $w=\sum_{j=1}^{n}f(t-T_{i_j})$，$w \geq \mu$ 为核心微簇的权重；$c = \frac{\sum_{j=1}^{n}f(t-T_{i_j}) p_{i_j}}{w}$ 为核心微簇的中心；$r = \frac{\sum_{j=1}^{n}f(t-T_{i_j}) dist(p_{i_j}, c)}{w}$, $r \leq \epsilon$ 为核心微簇的半径；$dist(p_{i_j}, c)$ 为点 $p_{i_j}$ 到中心 $c$ 的欧几里德距离。

上述定义要求一个核心微簇的权重至少为 $\mu$, 半径不能超过 $\epsilon$。因此核心微簇是一种“密集”微簇。由于半径的限制，核心微簇的个数 $N_c$ 必然远大于自然簇的个数；由于权重的限制，$N_c$ 又必然远小于流中数据点的个数。数据流中任意形状的簇可被一组无冗余的核心微簇描述。


## 潜在核心微簇与离群微簇

在流数据中，簇和离群点常常会相互转化，并且任何一个核心微簇都是随着数据流的演化逐渐形成。因此，算法引入了潜在核心微簇和离群微簇的概念来进行增量式计算。这两个概念比较相似，不同之处在于他们的权重限制。对于潜在核心微簇，$w \geq \beta \mu$; 对于离群微簇，$w < \beta \mu$

`潜在核心微簇`（potential c-micro-cluster，简记为 p-micro-cluster）：对一组时间戳为 $T_{i_1}, T_{i_2}, \cdots, T_{i_n}$ 的相互临近的数据点 $p_{i_1}, p_{i_2}, \cdots, p_{i_n}$, 在 $t$ 时刻的潜在核心微簇定义为 { $\overline{CF^1}, \overline{CF^2}, w$ }。其中 $w=\sum_{j=1}^{n}f(t-T_{i_j})$，$w \geq \beta\mu$ 为潜在核心微簇的权重；$ 0< \beta \leq 1$ 为用于决定离群点相对于核心微簇的阈值的参数。$\overline{CF^1} = \sum_{j=1}^{n}f(t-T_{i_j}) p_{i_j}$ 为数据点加权的线性和；$\overline{CF^1} = \sum_{j=1}^{n}f(t-T_{i_j}) p_{i_j}^2$ 为数据点加权的平方和。潜在核心微簇的中心为 $c = \frac{\overline{CF^1}}{w}$，半径为 $r = \sqrt{\frac{\vert \overline{CF^2}\vert}{w} - (\frac{\vert \overline{CF^1}\vert}{w})^2}$，$r \leq \epsilon$。

`离群微簇`（outlier micro-cluster，简记为 o-micro-cluster）：对一组时间戳为 $T_{i_1}, T_{i_2}, \cdots, T_{i_n}$ 的相互临近的数据点 $p_{i_1}, p_{i_2}, \cdots, p_{i_n}$, 在 $t$ 时刻的定义为 { $\overline{CF^1}, \overline{CF^2}, w, t_o$ }。其中 $w$，$\overline{CF^1}$，$\overline{CF^2}$，$c$，$r$ 的定义都与潜在核心微簇相同。$t_o = T_{i_1}$ 为离群微簇的创建时间，用来确定离群簇的生命期。离群微簇的权重限制为 $w < \beta \mu$。


潜在核心微簇和离群微簇可以被增量维护。例如，假设某潜在核心微簇为 $c_p = (\overline{CF^2}, \overline{CF^1}, w)$，如果在 $\delta t$ 时间内没有数据被 $c_p$ 吸收，那么有 $c_p = (2^{-\lambda \delta t} \cdot \overline{CF^2}, 2^{-\lambda \delta t} \cdot  \overline{CF^1}, 2^{-\lambda \delta t} \cdot w)$。若数据点 $p$ 被 $c_p$ 吸收，则 $c_p = (\overline{CF^2} + p^2, \overline{CF^1} + p, w + 1)$。

> 注：有一点不是很理解，如果一个数据点在 $\delta t$ 时刻被吸收，之前数据的权重已经衰减，那么不应该是 $c_p = (2^{-\lambda \delta t} \cdot \overline{CF^2} + p^2, 2^{-\lambda \delta t} \cdot  \overline{CF^1} + p, 2^{-\lambda \delta t} \cdot w + 1)$ 么？为什么这里没有权重衰减的过程？


## DenStream 算法

DenStream 算法被分为两部分：（1）在线的微簇维护和 （2）离线的用户请求所触发的最终簇生成过程。

### 微簇维护

为了及时发现进化数据流中的簇，算法需要在线维护一组潜在核心微簇 {$cmc_p^1, cmc_p^2, \cdots, cmc_p^{N_p}$} 和一组离群微簇 {$cmc_o^1, cmc_o^2, \cdots, cmc_o^{N_o}$}。所有的离群点微簇被维护在一个独立的内存空间，被称为`离群点缓存`（outlier-buffer）。当一个新的数据点到达时，可以用下面算法进行合并

![1-2 合并新到达数据点的算法](/assets/blog-images/Streaming-Data-Analysis-1.2.png)
**Fig. 1-2.** 合并新到达数据点的算法。 

1. 算法首先尝试将新的数据点 $p$ 并入距离它最近的潜在核心微簇 $c_p$。如果并入后 $c_p$ 的新半径 $r_p \leq \epsilon$，则可以将 $p$ 真正的并入 $c_p$，并入操作依据增量维护的性质来实现。

2. 如果 $p$ 不能并入 $c_p$， 则尝试将 $p$ 并入离它最近的离群微簇 $c_o$。如果并入后 $c_o$ 的新半径 $r_o \leq \epsilon$，则可以将 $p$ 真正并入 $c_o$。紧接着我们检查并入后 $c_o$ 的新权值 $w$。 如果 $w \geq \beta \mu$, 则表明 $c_o$ 已经成长为一个潜在核心微簇，因此将 $c_o$ 从离群点缓存中删除，并根据 $c_o$ 创建一个新的潜在核心微簇。

3. 如果 $p$ 不能被现有的任何潜在核心微簇和离群微簇吸收，则根据 $p$ 创建一个新的离群微簇，并插入到离群点缓存中。


一个潜在核心微簇 $c_p$ 的权重随着时间不停的衰减。如果一直没有新的数据点被吸收，则最终 $c_p$ 的权值将会小于 $\beta \mu$，当 $c_p$ 的 $w$ 小于 $\beta \mu$ 时，证明该潜在核心微簇已经退休，需要将它删除并释放内存。


在计算中，并不需要频繁地检查微簇的权值。考虑一个潜在核心微簇，其权重已经退化到临界值 $\beta \mu$, 如果在接下来的时间 $T_p$ 内没有新的数据点加入，则该数据点将最终退化并退休，然后如果在 $T_p$ 时候有新的数据加入，导致该核心微簇的权值重新反弹到 $\beta \mu$, 则我们需要仍然保留该微簇。因此我们检查微簇的最短时间间隔可以通过 $2^{-\lambda T_p} \beta \mu + 1 = \beta \mu$ 计算，最终得到检查的最短时间间隔为

$$
T_p = \lceil \frac{1}{\lambda} \log \left( \frac{\beta \mu}{\beta \mu - 1}\right) \rceil
$$ 



离群微簇的数量可能随着数据不断的增长。我们一方面要保留那些可能成长为潜在核心微簇的离群微簇，一方面又要及时删除真正的离群微簇以释放内存。理想的情况下，我们需要无限长的时间才能最终判定一个离群微簇能否成长为一个潜在核心微簇。在实际中这是不可能的。因此需要引入一个近似方法来判断一个离群微簇能否成长为一个潜在核心微簇。DenStream 算法定义了一个权值下限 $\xi$，如果某个离群微簇的权值低于 $xi$，则意味着该离群微簇不太可能成长为一个潜在核心微簇，因此可以安全地从离群缓存中删除。权值下限定义为

$$
\xi(t_c, t_o) = \frac{2^{-\lambda (t_c - t_o + T_p)} - 1}{2^{-\lambda T_p} - 1}
$$

其中 $t_o$ 为该离群微簇的创建时间, $t_c$ 为当前时间。随着时间的流逝 $xi$ 逐渐增加，且有 $\lim\limits_{t_c \to \infty} \xi(t_c) = \frac{1}{1 - 2^{-\lambda T_p}} = \beta \mu$

![1-3 $\xi$ 随时间的变化](/assets/blog-images/Streaming-Data-Analysis-1.3.png)
**Fig. 1-3.** $\xi$ 随时间的变化。 

> 这里的 $\xi$ 为什么这么定义应该只是一个近似。

下图为 DenStream 的算法流程图

![1-4 DenStream 算法](/assets/blog-images/Streaming-Data-Analysis-1.4.png)
**Fig. 1-4.** DenStream 算法。 



#### 初始化

对数据流中最初的包含 $N$ 个数据的集合 $P$ 使用 `DBSCAN 算法`，首先扫描数据集获得最初的一组潜在核心微簇。对于 $\forall p \in P$， 若其 $\epsilon$ 邻域内的数据点总权值之和大于或等于 $\beta \mu$，则依据 $p$ 以及其邻居数据点建立一个潜在核心微簇，并将他们从 $P$ 中删除。


### 理论分析

显然，离群微簇的删除将会对离群微簇以及潜在核心微簇的权值产生影响。考虑一个离群微簇 $c_o$, 如果在一定时间内其权值低于我们定义的下限 $\xi$, 那么 $c_o$ 将会被删除。但是如果过了一段时间，在 $c_o$ 的位置又有一个新的离群微簇 $c'_o$ 生长出来。由于我们已经失去了 $c_o$ 的信息，那么我们估计 $c'_o$ 的权值时将不会包含 $c_o$ 的权重。

幸运的是，如果对于任意一个潜在核心微簇 $c_p$ 而言，如果 $c_p$ 的当前的精确权值大于 $\beta \mu$ 的（即在 $c_p$ 的半径范围内所有数据点的权值都被包含了进来，没有因为删除导致的误差）。那么其必定出现在离群缓存或者潜在核心微簇内。若 $c_p$ 当前的精确权值大于 $2\beta\mu$，则其必出现在潜在核心微簇集内。

令 $w_e$ 为 $c_o$ 或者 $c_p$ 当前的精确权值，$w$ 表示 $c_o$ 或者 $c_p$ 所维护的权值, $t_o$ 为 $c_o$ 或者 $c_p$ 的创建时间。

**引理 1：** 任意时刻某离群微簇 $c_o$ 被删除，必定有 $w_e \leq \beta \mu$。

**证明：** 根据 $c_o$ 创建前所处位置有无微簇出现，分下面两种情况：

1. 若 $c_o$ 创建前，其所在位置没有微簇，直接有 $w_e = w$。 根据删除规则 

   $$
   w_e < \frac{2^{-\lambda (t_c - t_o + T_p)} - 1}{2^{-\lambda T_p} - 1} < \beta \mu
   $$
   
2. 若 $c_o$ 创建前，其所在位置有微簇 $c_x$ 出现，根据删除规则， $c_x$ 被删除时权值最大为 $\beta \mu$，因而有

   $$
   \begin{eqnarray}
   w_e  & < & w + 2^{-\lambda (t_c - t_o + T_p)} \beta \mu \\
        & \leq & \frac{2^{-\lambda (t_c - t_o + T_p)} - 1}{2^{-\lambda T_p} - 1} + 2^{-\lambda (t_c - t_o + T_p)} \beta \mu \\
        & \leq & \beta \mu
   \end{eqnarray}
   $$ 

即不论哪种情况，我们均有 $w_e \leq \beta \mu$。


**引理 2：** 对任意一个潜在核心微簇 $c_p$，其权重满足 $w \leq w_e \leq w + 2^{-\lambda (t_c - t_o)} \beta \mu $。

**证明：** 若 $c_p$ 在建立前其所在位置没有任何微簇被删除，则有 $w = w_e$。否则，如果 $c_p$ 所在位置之前有一个或者多个微簇被删除，根据引理1, 我们可以推断 $c_p$ 所在位置之前的微簇被删除时，精确权值最多为 $\beta \mu$，并且该权值已经衰减为 $2^{-\lambda (t_c - t_o)} \beta \mu$。因此有 $w_e \leq w + 2^{-\lambda (t_c - t_o)} \beta \mu $。


### 最终簇的生成

在线维护的微簇可以看作对数据流密集区域的一个粗粒化，即用一个包含权重和半径信息的虚拟对象代表许多点的集合。如果只是做异常值检测，有离群微簇的信息就足够了。但是 DenStream 算法的最终的目的是对流数据做聚类。当聚类请求到达时，DenStream 算法对 DBSCAN 算法进行了拓展，应用到在线维护的潜在核心微簇 $c_p$ 上，并对 $c_p$ 进行聚类分析。

DenStream 对 DBSCAN 的扩展包含两个参数 $\epsilon$ 和 $\mu$，并借用 DBSCAN 的`密度连通`（density-connectivity）概念来生成最终的簇。

`直接密度可达`（directly density-reachable）： 某潜在核心微簇 $c_p$ 相对于另一个潜在核心微簇 $c_q$ 以参数 $\epsilon$ 和 $\mu$ 密度可达，如果 $c_q$ 的权值大于 $\mu$（即 $c_q$为一个核心微簇） 并且 $dist(c_p, c_q) \leq 2 \epsilon$。其中 $dist(c_p, c_q)$ 为 $c_p$ 和 $c_q$ 中心间的距离。

> 即使两个微簇 $c_p$ 和 $c_q$ 间的距离小于 $2\epsilon$, 由于微簇的半径可以小于 $\epsilon$，所以他们也有可能不相交（即密度直达）。这样的情况可以通过检查 $dist(c_p, c_q) \leq r_p + r_q$ 判断。最终会与上述算法得到类似的结果。

`密度可达`（density-reachable）： 某潜在核心微簇 $c_p$ 相对于另一个潜在核心微簇 $c_q$ 以参数 $\epsilon$ 和 $\mu$ 密度可达，当且仅当存在一条核心微簇链 $c_{p_1}, \cdots, c_{p_n}$，且 $c_{p_1} = c_p$，$c_{p_n} = c_q$，$c_{p_i}$ 直接可达 $c_{p_{i+1}}$。

`密度连通`（density-connected）：某个潜在核心微簇 $c_p$ 对于另一个潜在核心微簇 $c_q$ 以参数 $\epsilon$ 和 $\mu$ 密度连通，当且仅当存在一个潜在核心微 $c_m$, 使得 $c_p$ 和 $c_q$ 对于参数 $\epsilon$ 和 $\mu$ 都与 $c_m$ 密度可达。


# Reference

1. F. Cao, M. Estert, W. Qian, and A. Zhou, "[Density-Based Clustering over an Evolving Data Stream with Noise,](https://doi.org/10.1137/1.9781611972764.29)" in *Proceedings of the 2006 SIAM International Conference on Data Mining,* **2006,** pp. 328-339.

2. 曹锋, "[数据流聚类分析算法](http://cdmd.cnki.com.cn/Article/CDMD-10246-2007068543.htm)" 博士, 计算机科学与工程系, 复旦大学, 2006.

3. Paul Voigtlaender, "[DenStream](http://dme.rwth-aachen.de/en/system/files/file_upload/course/12/elementary-data-mining-techniques/denstreampresentation.pdf)"

