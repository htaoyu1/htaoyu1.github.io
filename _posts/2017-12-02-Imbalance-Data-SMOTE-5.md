---
layout: post
title: 非平衡数据集的处理：SMOTE 类算法（5）
category: Machine Learning 
author: Hongtao Yu
tags: 
  - machine-learning 
  - imbalanced-data
comments: true
use_math: true
lang: zh
---

- TOC
{:toc}




# G-SMOTE

SMOTE 算法以及其一些变种在两个少数类样本的连线上合成新的少数类样本，这可能会造成以下一些问题：（1）会产生入侵到多数类样本空间的噪声样本；（2）在少数类样本稠密区域产生的新样本并没有给数据集引入太多新的信息，可能会造成过拟合。`G-SMOTE`（Geometric SMOTE）首先选择一个少数类样本，以及其一个近邻样本（可以是多数类样本，也可以是少数类样本），然后在他们周围一个定义好的弹性几何空间内合成新的样本。该空间的几何边界弹性由预定义的参数控制。

## 算法

假定整个训练集为 $T$，其中少数类样本集记为 $P$（正例），多数类样本集记为 $N$ （负例），样本的维度为 $l$。

1. 定义截断（truncation）因子 $\alpha_{t} \in [-1, 1]$，形变（deformation）因子 $\alpha_{d} \in [-1, 1]$，以及一个近邻选择策略 $\alpha_s \in \\{ minority, majority, combined\\}$。

2. 从少数类里随机选取一个样本 $\mathbf{x}$。然后根据预定义的近邻选择策略 $\alpha_s$，进行以下三种操作里面的一种：

   1. 如果 $\alpha_s = minority$:  从少数类 $P$ 中搜索 $\mathbf{x}$ 的 $k$-近邻，并随机选取一个，记为 $\hat{\mathbf{x}}$，定义弹性空间的半径为 $\mathbf{x}$ 和 $\hat{\mathbf{x}}$ 之间的距离 $R = d(\mathbf{x}, \hat{\mathbf{x}})$。

   2. 如果 $\alpha_s = majority$:  从多数类 $N$ 中找到 $\mathbf{x}$ 的最近邻，记为 $\hat{\mathbf{x}}$，定义弹性空间的半径为 $\mathbf{x}$ 和 $\hat{\mathbf{x}}$ 之间的距离 $R = d(\mathbf{x}, \hat{\mathbf{x}})$。
   
   3. 如果 $\alpha_s = combined$:  从少数类 $P$ 中搜索 $\mathbf{x}$ 的 $k$-近邻，并随机选取一个，记为 $\hat{\mathbf{x}}\_+$；从多数类 $N$ 中找到 $\mathbf{x}$ 的最近邻，记为 $\hat{\mathbf{x}}\_-$。 定义 $\hat{\mathbf{x}}$ 为 $\hat{\mathbf{x}}\_-$ 和 $\hat{\mathbf{x}}\_+$ 两者中离 $\mathbf{x}$ 最近的样本，并定义弹性空间的半径 $R = d(\mathbf{x}, \hat{\mathbf{x}})$。
   
3. 使用 hyperball 函数产生一个随机矢量:

   $$
   \mathbf{x}_{\text{new}} = hyperball_{(\text{center=0, radius=1})}()
   $$

4. 对 hyperball 产生的随机矢量做截断：

   $$
   \mathbf{x}_{\text{new}} = truncate(\mathbf{x}_{\text{new}}, \mathbf{x}, \hat{\mathbf{x}}, \alpha_t)
   $$

5. 对截断后的随机矢量做形变处理：

   $$
   \mathbf{x}_{\text{new}} = deform(\mathbf{x}_{\text{new}}, \mathbf{x}, \hat{\mathbf{x}}, \alpha_d)
   $$
   
6. 对形变后的随机矢量做平移和缩放，产生新的少数类样本：

   $$
   \mathbf{x}_{\text{new}} = translate(\mathbf{x}_{\text{new}}, \mathbf{x}, R)
   $$

### 几何函数

1. 定义平行于 $\hat{\mathbf{x}} - \mathbf{x}$ 的单位矢量:

   $$
   \mathbf{e}_{//} = \frac{\hat{\mathbf{x}} - \mathbf{x}}{\vert \hat{\mathbf{x}} - \mathbf{x} \vert}
   $$
   
2. 计算 $\mathbf{x}\_{\text{new}}$ 在 $\mathbf{e}\_{//}$ 上的投影长度：

   $$
   x_{//} = \mathbf{x}_{\text{new}} \cdot \mathbf{e}_{//}
   $$
   
3. 则 $\mathbf{x}\_{\text{new}}$ 在平行和垂直于 $\mathbf{e}\_{//}$ 的方向上的投影矢量分别为:

   $$
   \mathbf{x}_{//} = x_{//} \mathbf{e}_{//}
   $$
   
   $$
   \mathbf{x}_{\perp} = \mathbf{x}_{\text{new}} - \mathbf{x}_{//}
   $$
   
### hyperball 函数

1. 产生具有正态分布 $N(0,1)$ 的 $l$ 个随机数，构成矢量 

   $$
   \mathbf{v}_{\text{norm}} = (v_1, \cdots, v_l)
   $$

2. 构造单位矢量:

   $$
    \mathbf{e}_{\text{sphere}} = \frac{\mathbf{v}_{\text{norm}}}{\vert \mathbf{v}_{\text{norm}} \vert}
   $$ 

3. 产生一个均匀分布 $U(0,1)$ 的随机数 $r$，构造矢量:

   $$
   \mathbf{x_{\text{new}}} = r^{1/l} \mathbf{e}_{\text{sphere}}
   $$
   
![5-1 hyperball 函数](/assets/blog-images/Imbalance-Data-SMOTE-5.1.png)
**Fig. 5-1:** hyperball 函数。

> 该步骤产生了一个单位超球体内方向为正态分布、半径为均匀分布的矢量（Fig. 5-1）。其中步骤1和2产生一个具有正态方向分布的随机单位矢量，步骤3将该单位矢量的半径收缩，转化为超球体内的均匀分布。


### truncate 截断函数

1. 如果 $\hat{\mathbf{x}} \neq \mathbf{x}$ 并且 $ \vert \alpha_t - x_{//} \vert > 1$ 则将 $\mathbf{x}_{\text{new}}$ 投影到允许空间：

    $$
    \mathbf{x}_{\text{new}} = \mathbf{x}_{\text{new}} - 2 \mathbf{x}_{//}
    $$
    
![5-2 truncate 函数](/assets/blog-images/Imbalance-Data-SMOTE-5.2.png)
**Fig. 5-2:** truncate 函数。

$\mathbf{x}$ 和其被选中的近邻样本 $\hat{\mathbf{x}}$ 定义了一个特殊的方向 $\mathbf{e}\_{//}$。而 $\mathbf{e}\_{//}$ 又可以定义一系列与其垂直的超平面（比如上图中 $P$ 为通过原点与 $\mathbf{e}\_{//}$ 垂直的超平面）。超平面与 $\mathbf{e}\_{//}$ 的交叉点和 $\alpha_t$ 之间可以形成一对一的映射。因此每一个超平面可以用 $\alpha_t$ 唯一地表示。当 $\alpha_t > 0$ 时，不包含 $\mathbf{e}\_{//}$ 矢量的一半超球体的部分区域被设置为禁止空间（空间大小取决于 $\alpha_t$）。如果 hyperball 产生的随机矢量不幸落在了禁止区域（上图中的蓝色区域），那么该函数将其镜像投影回允许空间。如果 $\alpha_t < 0$, 那么禁止空间将会落在包含矢量 $\mathbf{e}\_{//}$ 的半球内。

### deform 形变函数

1. 如果 $\hat{\mathbf{x}} \neq \mathbf{x}$, 则将 $\mathbf{x}\_{\text{new}}$ 沿着 $\mathbf{x}\_{\perp}$ 方向压缩：

   $$
   \mathbf{x}_{\text{new}} = \mathbf{x}_{\text{new}} - \alpha_d \mathbf{x}_{\perp}
   $$
   
![5-3 deform 函数](/assets/blog-images/Imbalance-Data-SMOTE-5.3.png)
**Fig. 5-3:** deform 函数。

> 该函数将经过 trancate 作用后的矢量做压缩。除了 $\mathbf{e}\_{//}$ 方向之外，矢量在其余所有方向上均被进行压缩，压缩的幅度由参数 $\alpha_d$ 控制。

### translate 平移函数

1. 将 $\mathbf{x}_{\text{new}}$ 的原点平移到 $\mathbf{x}$ ，并施加缩放因子 $R$:

   $$
   \mathbf{x}_{\text{new}} = \mathbf{x} + R \cdot \mathbf{x}_{\text{new}}
   $$ 
   
![5-4 translate 函数](/assets/blog-images/Imbalance-Data-SMOTE-5.4.png)
**Fig. 5-4:** translate 函数。

> 这一步很容易理解，将压缩后的随机矢量平移到以 $\mathbf{x}$ 为起点，然后进行缩放。所以最终合成的新样本会落在 $\mathbf{x}$ 附近的一个被截断（取决于 $\alpha_t$）的超椭球体内。椭球体的最长轴大小 $R$ 等于 $\hat{\mathbf{x}}$ 和 $\mathbf{x}$ 之间的距离，方向沿着 $\hat{\mathbf{x}} - \mathbf{x}$。

## 优缺点

- 在少数类样本周围定义了一个安全区域，在此区域内合成的新样本不会成为噪声。

- 通过扩展样本的生成空间来增加样本的多样性。

- 实现起来相对比较复杂。


# DBSM

 `DBSM`  同时使用了欠采样和过采样技术。首先使用 DBSCAN 算法将样本聚类后对多数类进行欠采样，然后使用 SMOTE 技术对少数类进行过采样。

## 算法
 
假定整个训练集为 $T$，其中少数类样本集记为 $P$（正例），多数类样本集记为 $N$ （负例）。

1. 使用 DBSCAN 聚类算法将整个训练集 $T$ 聚类。

2. 对于聚类后的第 $i$ 个集簇 $C_i$, 

   1. 如果 $C_i$ 中全是多数类样本, 则计算集簇的中心点，然后计算每个样本到中心点的距离。

   2. 如果 $C_i$ 中同时包含了少数类和多数类样本，则对每一个多数类样本，计算其离 $C_i$ 中少数类样本的最短距离。

3. 对步骤2中计算出来的每个多数类样本的距离进行排序，然后移除前 50% 距离最小的多数类样本（即如果是纯多数类集簇，移除离集簇中心最近的样本；如果是混合集簇，移除离少数类样本最近的多数类样本。后者其实有点类似 [Tomek Links](https://plushunter.github.io/2017/04/18/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95%E7%B3%BB%E5%88%97%EF%BC%8817%EF%BC%89%EF%BC%9A%E9%9D%9E%E5%B9%B3%E8%A1%A1%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/)）。

4. 对少数类样本使用 SMOTE 算法合成新的数据。

5. 将步骤3欠采样后的多数类样本和步骤4过采样后的少数类样本结合起来，得到平衡数据集。

# LLE-based SMOTE

SMOTE 算法假定两个少数类（正例）样本之间的样本仍为少数类。然而如果训练集不是线性可分的，那么这种假设就可能存在问题。然而如果先把训练集映射到一个线性可分的流形空间，然后使用 SMOTE 算法在线性可分流形内对少数类样本进行过采样，再将合成的新样本重新映射回原流形，就可以有效地规避这种缺陷。

`LLE-based SMOTE` 算法首先使用 LLE（Locally Linear Embedding）技术将高维样本映射到一个低维的易分流形，然后利用 SMOTE 技术对少数类样本进行过采样，最后再将合成的样本映射回原始的高维流形。


## LLE（Locally Linear Embedding）

给定一个在 $d$-维空间 $R^d$ 的包含 $N$ 个样本的数据集 $ X = \\{ \mathbf{x}\_1, \mathbf{x}\_2, \cdots, \mathbf{x}\_N \\} $，LLE 算法在一个低维的 $R^l$ 空间寻求数据集 $X'$，使其与 $X$ 在 $k$-近邻图（kNN）中具有相同的局部几何结构。换句话说，任意 $\mathbf{x}\_i \in X$ 被投影成 $\mathbf{x}'\_i = F(\mathbf{x}\_i) \in X'$，如果 $\mathbf{x}\_i$ 可通过其 $k$-近邻 $X_{kNN}(\mathbf{x}_i) = \\{ \mathbf{x}\_{ij} \vert 1 \leq j \leq k \\} $ 线性张成

$$
\mathbf{x}_i = \sum\limits_{j=1}^k w_{ij} \mathbf{x}_j
$$

那么有

$$
\mathbf{x}'_i = \sum\limits_{j=1}^k w_{ij} \mathbf{x'}_j
$$


LLE 算法首先使用原始样本构建一个 $k$-近邻图，然后通过最小化下面的损失函数来求解权重系数 $w_{ij}$:

$$
J(W) = \sum\limits_{i=1}^{N} \Vert \mathbf{x}_i - \sum\limits_{j=1}^k w_{ij}\mathbf{x}_j \Vert ^2
\label{eq:lle-w}
$$

其中 $\sum\limits_{j=1}^{k} w_{ij} = 1$ 并且如果样本 $\mathbf{x}\_j$ 不是 $\mathbf{x}\_i$ 的 $k$-近邻，有 $w_{ij}=0$。求解出权重系数 $W$ 后，LLE 通过最小化损失函数

$$
J(X') = \sum\limits_{i=1}^{N} \Vert \mathbf{x}'_i - \sum\limits_{j=1}^k w_{ij}\mathbf{x}'_j \Vert ^2
\label{eq:lle-xp}
$$

来构建低维流形内的嵌入数据集 $X'$。对 LLE 算法感兴趣的请参考[这篇帖子](http://www.cnblogs.com/pinard/p/6266408.html)和 [Saul 的论文](https://dl.acm.org/citation.cfm?id=945372)。

## LLE-based SMOTE

与 SMOTE 不同，LLE-based SMOTE 算法不是使用随机线性插值的方式, 而是使用一种确定性的方式生成新样本。新样本的生成规则是保证其离其他多数类样本集 $N'$ 中的样本的平均距离最远：

$$
\mathbf{x'}_{\text{new}} = \underset{\mathbf{x'}_{\text{new}} \in \overline{\mathbf{x'}\mathbf{x'}_j}}{\text{argmax}} \frac{1}{k}\sum\limits_{\mathbf{x'}_- \in N'} \Vert \mathbf{x'}_{\text{new}} - \mathbf{x'}_-\Vert
$$

这里 $\mathbf{x'}$ 为种子样本，$\mathbf{x'}\_j$ 为被随机选中做 SMOTE 合成的 $\mathbf{x'}$ 的第 $j$ 个 $k$-近邻，$\mathbf{x'}\_{\text{new}} \in \overline{\mathbf{x'}\mathbf{x'}\_j}$ 表示新合成的样本位于 $\mathbf{x'}$  和 $\mathbf{x'}\_j$ 的连线中间。

> 对这一步理解的不是很清楚。我个人感觉公式中的 $N'$ 其实应该指的是在多数类样本集中找到的 $k$-近邻。因为计算平均距离的时候是用的 $k$ 当除数，所以不可能是在整个样本集中找到的既包含多数类又包含少数类的 $k$ 近邻。另一个不明白的地方是，如果两个样本两次被同时选中做 SMOTE 合成，那么根据这个确定性公式，两个新合成的样本岂不是要重合了？
> 
> 

## 算法

假定整个训练集为 $T$，其中少数类样本集记为 $P$（正例），多数类样本集记为 $N$ （负例），样本的维度为 $d$。


### LLE

1. $\forall \mathbf{x}\_i \in T$ ，在整个训练集 $T$ 中寻找其 $k$-近邻，记为 $X^0\_{kNN}(\mathbf{x}\_i)$。我们称其为 $\mathbf{x}\_i$ 的初始 $k$-近邻集。

2. 对 $X^0\_{kNN}(\mathbf{x}\_i)$ 中的每个少数类样本 $\mathbf{v}$，如果 $\mathbf{v}$ 的初始 $k$-近邻集，$X^0\_{kNN}(\mathbf{v})$， 中包含的多数类样本大于某个值 $k_+$，那么将 $\mathbf{v}$ 加入到 $\mathbf{x}\_i$ 的修正 $k$-近邻集合 $X_{kNN}(\mathbf{x}\_i)$ 中。显然此时 $X_{kNN}(\mathbf{x}\_i)$ 中的样本个数 $\vert X_{kNN}(\mathbf{x}\_i) \vert \leq k$。接下来将 $\mathbf{x}\_i$ 的 $k-\vert X_{kNN}(\mathbf{x}\_i) \vert$ 个多数类近邻加入到集合 $X_{kNN}(\mathbf{x}\_i)$ 中。

3. 利用步骤2得出的修正 $k$-近邻集合，结合公式 \ref{eq:lle-w} 计算张成因子 $W$。

4. 使用公式 \ref{eq:lle-xp} 计算 $T$ 在低维流形的嵌入样本集 $T'$。


### LLE-based SMOTE

1. 首先使用 LLE 算法将训练集 $T$ 映射到低维流形 $T'$。假设少数类样本集 $P$ 的低维嵌入为 $P'$, 多数类样本集 $N$ 的低维嵌入为 $N'$。

2. 在低维流形内合成少数类样本 $\mathbf{x}'_{\text{new}}$。

3. 在 $T'$ 内寻找 $\mathbf{x}'\_{\text{new}}$ 的 $k$-近邻 $X_{kNN}(\mathbf{x}'\_{\text{new}})$。

4. 最小化损失函数

   $$
   J(W) = \sum\limits_{i=1}^{N} \Vert \mathbf{\mathbf{x}'_{\text{new}}} - \sum\limits_{j=1}^k w_{j}\mathbf{x'}_j \Vert ^2
   $$

   找到 $\mathbf{x}'\_{\text{new}}$ 在低维流形内被其 $k$-近邻线性张成的张成系数 $w_j$。其中 $\mathbf{x'}\_j \in X_{kNN}(\mathbf{x}'\_{\text{new}})$ 为 $\mathbf{x}'\_{\text{new}}$ 在低维流形的第 $j$ 个 $k$-近邻。

5. 使用公式
 
   $$
   \mathbf{x}_{\text{new}} = \sum\limits_{j=1}^k w_j \mathbf{x}_j
   $$

   将新合成的样本映射回原始 $d$-维空间。其中 $w_j$ 为步骤4计算出的张成系数，$\mathbf{x}\_j$ 为步骤4中的 $\mathbf{x}'\_j$ 在原始流形内对应的样本。



# MWMOTE

![5-5 当前 SMOTE 算法可能存在的一些缺陷。](/assets/blog-images/Imbalance-Data-SMOTE-5.5.png)
**Fig. 5-5:** 当前 SMOTE 算法可能存在的一些缺陷。

在 Borderline-SMOTE 算法中，一个样本的 $k$-近邻包含的多数类样本个数为 $m$, 如果 $\frac{k}{2} \leq m \lt k$, 那么该样本被归为边界样本。在某些情形下，该算法可能会遇到一些问题。如 Fig. 5-5(a) 所示，如果选取 $k=5$, 那么样本 $A$ 和 $B$ 的 5个近邻样本全属于少数类，因此这两个样本将不会被归于边界类样本，也就不会被选取用来合成新样本。

ADASYN 和 RAMOBoost 算法试图通过自适应地赋予少数类样本不同的权重来避免上述问题。他们使用一个少数类样本的 $k$-近邻里含有的多数类样本个数 $\delta$ 来定义权重因子。$\delta$ 值大的少数类样本被选中的概率大。然而这种定义存在以下几个问题：

   1. 参数 $\delta$ 不适合用来赋予少数类的边界样本权重。比如 Fig. 5-5(a) 中的样本 $A$ 和 $B$ 的权重都将会是0。

   2. 只使用参数 $\delta$ 来区分边界样本的重要性是不够的。比如 Fig. 5-5(a) 中的样本 $A$ 更靠近边界，因此需要赋予更大的权重。然而只使用 $\delta$ 并不能区分少数类样本 $A$ 和 $C$ 重要性的差异。就算使用较大的 $k$ 值使 $A$ 的 $k$-近邻包含一部分多数类样本，$\delta$ 的值仍然会比较低，导致 $A$ 不能拥有更大的权重。
 
   3. $\delta$ 会导致算法偏好噪声样本。比如 Fig. 5-5(a) 的样本 $D$。


SMOTE 算法通过在两个样本间线性插值合成新的样本。在某些情况下，SMOTE 算法的效果可能会不是很理想。例如在 Fig. 5-5(b) 中, 如果 $A$ 和 $B$ 被选中，那么就可能产生错误的，与多数类重叠的少数类样本 $P$；或者算法从集簇 $L1$ 和 $L2$ 中各选择一个种子合成新样本，结果产生了噪声样本 $R$；亦或集簇 $L3$ 中的样本 $B$ 被选中为种子，结果其所有 $k$-近邻都落在同一集簇 $L3$ 内（$k=5$的情形下），那么新合成的样本也会落在 $L3$ 内，与已有样本近乎重叠。

导致上述一些列问题的原因在于 SMOTE 算法盲目地使用样本的 $k$-近邻来合成新的样本，而没有考虑到周围其他少数类样本的位置和距离。此外，$k$ 值的选取也不能提前预知。`MWMOTE` 具有双重目标：（1）改进种子样本的选择机制；（2）提高新样本的合成机制。实践中 MWMOTE 主要分三步：（1）从原始样本集中找到难分类（边界）样本，（2）给每个难分类样本赋予一个权重，（3）使用难分类样本，依照其权重合成新的样本。




## 算法

![5-6 MWMOTE 算法。](/assets/blog-images/Imbalance-Data-SMOTE-5.6.png)
**Fig. 5-6:** MWMOTE 算法。

假定整个训练集为 $T$，其中少数类样本集记为 $P$（正例），多数类样本集记为 $N$ （负例）。

1. 首先过滤少数类样本，移除 $k_1$-近邻全是多数类样本的噪声，过滤后的少数类样本组成少数类样本集 $P_1$ （Fig. 5-6(a)）。

2. $\forall x_i \in P_1$, 寻找其在多数类样本集 $N$ 中的 $k_2$-近邻。将这些近邻样本组成边界多数类样本集 $N'$ (Fig. 5-6(b))。

3. $\forall x_i \in N'$, 寻找其少数类 $k_3$-近邻，将这些近邻样本组成边界少数类样本集 $P'$ (Fig. 5-6(c))。

4. $\forall x_i \in P', \forall y_i \in N'$，计算信息权重（information weight） $I_w(y_i, x_i)$。

5. $\forall x_i \in P'$, 根据下面公式计算其选择权重 （selection weight）

   $$
   S_w(x_i) \sum\limits_{y_i \in N'} I_w(y_i, x_i)
   $$

   并将这些权重转换为选择概率（selection probability）

   $$
   S_p(x_i) = \frac{S_w(x_i)} {\sum_{x_i \in P'} S_w(x_i)}
   $$


6. 使用聚类算法，将少数类样本集 $P$ 聚类为 $m$ 个集簇 $L_1, L_2, \cdots, L_m$。

7. 使用选择概率 $S_p$, 从 $P'$ 中随机选择一个少数类样本 $x$。假定 $x$ 属于集簇 $L_k$。

8. 随机从集簇 $L_k$ 中选择另一个少数类样本 $\hat{x}$，使用 SMOTE 算法插值合成新的样本。


## 信息权重

MWMOTE 算法的步骤4涉及到信息权重的计算。每一个 $y_i \in P'$ 赋予了少数类 $x_i \in N'$ 一个权重因子，称为信息权重 $I_w(y_i, x_i)$。在 MWMOTE 中，信息权重定义为近邻因子 (closeness factor) $C_f(y_i, x_i)$ 和密度因子（density factor）$D_f(y_i,x_i)$ 的乘积：

$$
I_w(y_i, x_i) = C_f(y_i, x_i) \times D_f(y_i, x_i)
$$

### 近邻因子

对于样本 $x_i \in P', y_i \in N'$， 首先计算两者的正则化距离

$$
d_n(y_i, x_i) = \frac{dist(y_i, x_i)}{l} 
$$

其中 $dist(y_i, x_i)$ 为 $y_i$ 和 $x_i$ 之间的 Euclidean 距离，$l$ 为样本的维度。


近邻因子定义为

$$
C_f(y_i, x_i) = \frac{f(\frac{1}{d_n(y_i, x_i)})}{C_f^{th}} \times CMAX.
$$

其中 $C_f^{th}$ 和 $CMAX$ 为预定义的参数。函数 $f$ 定义为：

$$
f(x) = \left\{
  \begin{array}{ll}
    x        & \text{if } x \leq C_f^{th}  
     \\
    C_f^{th}  & \text{otherwise}
    \end{array}
\right.
$$

上述定义保证了 $C_f(y_i, x_i)$ 落在 $[0, CMAX]$ 范围内，并且相距较远的两个样本之间的信息权重较低。


### 密度因子

密度因子保证了稀疏区域的样本能被用来合成更多的新样本，定义为

$$
D_f(y_i, x_i) = \frac{C_f(y_i, x_i)}{\sum_{x_i \in N'} C_f(y_i, x_i)}
$$

# References

1. `G-SMOTE` G. Douzas and F. Bacao, "[Geometric SMOTE: Effective oversampling for imbalanced learning through a geometric extension of SMOTE,](http://adsabs.harvard.edu/abs/2017arXiv170907377D
)" *ArXiv:* 1709.07377, **2017.** 

2. `DBSM` Y. Sanguanmak and A. Hanskunatai, "[DBSM: The combination of DBSCAN and SMOTE for imbalanced data classification,](https://doi.org/10.1109/JCSSE.2016.7748928)" in *2016 13th International Joint Conference on Computer Science and Software Engineering (JCSSE),* **2016,** pp. 1-5.

3. `LLE-based SMOTE` J. Wang, M. Xu, H. Wang, and J. Zhang, "[Classification of Imbalanced Data by Using the SMOTE Algorithm and Locally Linear Embedding,](https://doi.org/10.1109/ICOSP.2006.345752)" in *2006 8th international Conference on Signal Processing,* **2006.**

4. `LLE` L. K. Saul and S. T. Roweis, "[Think globally, fit locally: unsupervised learning of low dimensional manifolds,](https://dl.acm.org/citation.cfm?id=945372)" *J. Mach. Learn. Res.,* vol. 4, pp. 119-155, **2003.**

5. `MWMOTE` S. Barua, M. M. Islam, X. Yao, and K. Murase, "[MWMOTE--Majority Weighted Minority Oversampling Technique for Imbalanced Data Set Learning,](https://doi.org/10.1109/TKDE.2012.232)" *IEEE Transactions on Knowledge and Data Engineering,* vol. 26, pp. 405-425, **2014.**



