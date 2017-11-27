---
layout: post
title: 非平衡数据集的处理：SMOTE 类算法（2）
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


# ADASYN

`ADASYN` (Adaptive Synthetic Sampling Approach) 根据样本的分布自适应地合成少数类实例：根据样本的学习难易程度，使用难以学习的少数类样本合成更多的实例。ADASYN 算法不但可以降低学习偏倚(Learning Bias)，还能自适应地调整决策边界以专注于难学习的样本。

## 算法

假定整个训练集为 $T$, 其中少数类样本集记为 $P$（正例）, 多数类样本集记为 $N$ （负例）。

1. 计算样本集的类不平衡度:

   $$
   d = \frac{\vert P \vert}{\vert N \vert}
   $$

   其中 $\vert P \vert $ 和 $\vert N \vert$ 分别为少数类和多数类样本集内包含的样本个数。根据定义有 $d \in (0, 1]$。
   
   
2. 计算需要合成的少数类样本数:

   $$
   G = (\vert N \vert - \vert P \vert) \times \beta
   $$
   
   其中 $\beta \in [0, 1]$ 为期望合成少数类样本后达到的类别不平衡度。如果 $\beta = 1$, 则意味着产生一个多数类和少数类比例为 1:1 的样本集。


3. 对于 $x_i \in P$, 在整个训练样本集 $T$ 中基于 Euclidean 距离计算其 $k$-近邻，并计算比值

   $$
   r_i = \Delta_i/k, \quad i = 1, 2, \cdots, \vert P \vert
   $$
   
   其中, $\Delta_i$ 为 $x_i$ 的 $k$-近邻中包含的多数类样本个数。因此有 $r_i \in [0, 1]$。
   
4. 归一化 $r_i$

   $$
   \hat{r}_i = \frac{r_i}{\sum_{i=1}^{\vert P \vert} r_i}
   $$
   
5. 计算需要使用 $x_i$ 合成的样本个数：

   $$
   g_i = \hat{r}_i \times G
   $$

   并根据下面策略合成 $g_i$ 个新样本。

6. 从 $x_i$ 的 $k$-近邻里随机选取一个样本 $\hat{x}_i$, 使用下面的公式合成新样本：

   $$
   x_{\text{new}} = x + \text{rand}(0,1) \times (\hat{x} - x)
   $$

# AND-SMOTE

不论是 SMOTE，Borderline-SMOTE，Safe-Level SMOTE，LN-SMOTE，还是 ADASYN，在寻找 $k$-近邻时均使用一个单一的 $k$ 值。然而现实的数据中，数据集有不同的分布。即使在同一个数据集中，也可能存在类内的不平衡、噪声、以及小集簇。所以<u>对不同的样本点需要使用不同的 $k$ 值</u>。下图显示了 SMOTE 算法在实际使用中的一些局限性。

![2-1 SMOTE 算法的局限性](/assets/blog-images/Imbalance-Data-SMOTE-2.1.png)
**Fig. 2-1:** SMOTE 算法的局限性。

- 多数类样本中存在少数类样本的噪声实例。如上图(a)所示，有两个噪声实例落在了多数类中间。如果 SMOTE 算法不幸选中这两个实例合成数据，就有可能造成类别的重叠。类似地，在Borderline-SMOTE（$m=5$）和 ADASYN（$k=5$）算法中，这两个噪声数据也有可能被选中用来合成样本。理想的情况下，选择少数类样本合成新的数据实例时，这两个噪声样本需要被排除在外。

- 少数类样本形成分离的集簇或则复杂的形状。Fig. 2-1(b) 中，少数类样本形成了三个分离的小集簇。如果集簇的规模小于 $k$ 值，那么就有可能两个来自不同集簇的少数类样本被选中合成数据，从而造成样本的重叠。如果少数类的分布形成复杂的形状，也有可能造成同样的问题，比如 Fig. 2-1(c) 的情况。

为了克服 SMOTE 算法存在的问题，Yun 等人提出了 Automatic Neighborhood size Determination （`AND`）算法 以及基于 AND 的 `AND-SMOTE` 算法。

## 算法

### AND 算法

假定整个训练集为 $T$, 其中少数类样本集记为 $P$（正例）, 多数类样本集记为 $N$ （负例）。

1. 对每个 $x \in P$, 在 $P$ 中依次计算其 $i=1,2,\cdots,K$ 近邻。

2. 对于 $k$-近邻，计算所有的近邻点与 $x$ 包围的方形区域的并集 $R_{x,k}$。
  
3. 将所有落在 $R_{x,k}$ 内的样本都标记为少数类，然后计算在该近邻条件下的精度。（例如 Fig. 2-2, 点 $O_1$ 的 1-近邻包含样本 $N_1$, $R_{O_1,1} = Region_{11}$, 精度为 $2/2=1$; $O_1$ 的 2-近邻包含样本 $N_1$ 和 $N_2$，$R_{O_1,2} = Region_{11} \cup Region_{12}$ 包含了5个样本，其中2个属于多数类，所以预测精度是 $3/5=0.6$。）

2. 对于每一个样本 $x$，寻找导致预测精度突变的 $k$ 值，并将该值作为样本 $x$ 计算 $k$-近邻时需要使用的最优 $k$ 值。

![2-2 AND 算法中少数类样本的空间扩展](/assets/blog-images/Imbalance-Data-SMOTE-2.2.png)
**Fig. 2-2:** AND 算法中少数类样本的空间扩展。


### AND-SMOTE 算法

1. 对 $x_i \in P$, 在 $P$ 中计算其 $k_i$-近邻，其中 $k_i$ 为 AND 算法确定的 $x_i$ 的最优 $k$ 值。

2. 从 $x_i$ 的 $k_i$-近邻样本里随机选取一个样本 $\hat{x}_i$ 合成数据：

   $$
   \mathbf{x}_{\text{new}} = \mathbf{x}_i + \mathbf{\delta} \times (\hat{\mathbf{x}} - \mathbf{x}_i)
   $$

值得说明的是，AND-SMOTE 算法的 $\mathbf{\delta}$ 值是一个与 $x_i$ 同维度的矢量，其每一个维度都是一个 [0,1] 之间的随机数。所以与 SMOTE 在两个样本点之间插值不同，AND-SMOTE 在整个  $R_{x,k}$  区域内形成新的样本。

![2-3 AND-SMOTE 与 SMOTE 算法在少数类具有复杂分布形状的数据集上的性能比较](/assets/blog-images/Imbalance-Data-SMOTE-2.3.png)
**Fig. 2-3:** AND-SMOTE 与 SMOTE 算法在少数类具有复杂分布形状的数据集上的性能比较。


# SMOTE-D

SMOTE 算法是一种随机性算法，每次使用时产生的新样本集都不一样。因此，在多次使用 SMOTE 算法合成出几个新的数据集后，如何选择最好的一个成了问题。鉴于此，Torres 等人提出了一种确定性的 SMOTE 算法 —— `SMOTE-D` 算法。

## 算法

假定整个训练集为 $T$, 其中少数类样本集记为 $P$（正例）, 多数类样本集记为 $N$ （负例）。

1. 假定合成样本的比例为 $R\in [0,1]$，首先计算需要合成的新样本数量 $n = (\vert N \vert - \vert P \vert) \times R$。其中 $\vert N \vert$ 和 $\vert P \vert$ 分别为多数类和少数类包含的实例个数。


2. 对每个 $x_i \in P$, 在少数类样本集 $P$ 中计算其 $k$-近邻，以及每个近邻样本与 $x_i$ 的距离 $d_{ij}, \quad j = 1, 2, \cdots, k$。

3. 计算 $d_{ij}$ 的标准差 $\sigma_{i}$。

4. 计算少数类样本 $x_i$ 的标准差 $\sigma_i$ 在所有少数类样本的标准差里所占的比例：

   $$
   p_i = \frac{\sigma_i}{\sum_{j=1}^{\vert P \vert} \sigma_j}
   $$ 

4. 计算 $x_i$ 的与其第 $j$ 个近邻样本的距离在所有 $k$-近邻的距离里所占的比例：

   $$
   p_{ij} = \frac{d_{ij}}{\sum_{j=1}^{k} d_{ij}}
   $$ 

5. 则 $x_i$ 与其第 $j$ 个近邻样本之间需要合成的新样本个数为 $s_{ij} = p_i \times p_{ij} \times n$。

6. 在 $x_i$ 与其第 $j$ 个近邻样本之间，以 $d_{ij}/(s_{ij}+1)$ 为间隔，均匀地产生 $s_{ij}$ 个新样本。

![2-4 SMOTE-D 算法通过每个样本与其 $k$-近邻距离的方差占总体方差的比例决定每个样本合成新样本的数量。](/assets/blog-images/Imbalance-Data-SMOTE-2.4.png)
**Fig. 2-4:** SMOTE-D 算法通过每个样本与其 $k$-近邻距离的方差占总体方差的比例决定每个样本合成新样本的数量。

例如上图使用了3-近邻的 SMOTE-D 算法。假定样本的总体方差为 $\sum_{i=1}^{m} \sigma_i = 2$。所以 $\sigma_1$ 的 $p_1 = \sigma_1/\sum_{i=1}^{m} \sigma_i = 0.5$，因此有 50% 的样本通过 $\sigma_1$ 产生。相应地, $p_2 = 0.3, p_3 = 0.2$。如果需要产生 10 个样本，那么使用 SMOTE-D 算法，会有 5 个通过 $\sigma_1$ 产生，3 个通过 $\sigma_2$ 产生，2 个通过 $\sigma_3$ 产生。

# SMOTE-Out, SMOTE-Cosine, and Selected-SMOTE

SMOTE 和 Borderline-SMOTE 算法对一下三种情况的处理不够理想：

1. 少数类分布地非常密集，并且相互之间非常靠近，使用简单的线性插值已经不能有效地增加样本的多样性（variation）。

2. 传统的 SMOTE 算法在计算近邻样本时，只考虑两个样本之间的 Euclidean 距离，二个样本的相似度实际还可以用角度，或者方向向量度量。

3. 在产生新样本时，SMOTE 对每一个特征均进行合成。然而并不是样本的每一个特征都能代表多数和少数类的边界，只合成某些重要的特征有时候能产生更好的样本。


针对这三个问题，Kato 分别提出了 `SMOTE-Out`，`SMOTE-Cosine` 和 `Selected-SMOTE` 算法。

## SMOTE-Out 算法

![2-5 SMOTE-Out 算法](/assets/blog-images/Imbalance-Data-SMOTE-2.5.png)
**Fig. 2-5:** SMOTE-Out 算法。

假定整个训练集为 $T$, 其中少数类样本集记为 $P$（正例）, 多数类样本集记为 $N$ （负例）。

1. 对样本 $u \in P$, 分别计算其在 $P$ 和 $N$ 中的 $k$-近邻，记为 $K_P$ 和  $K_N$。

2. 从 $K_N$ 中随机选取一个多数类实例 $v$。

3. 在 $v$ 和 $u$ 的连线之外产生一个辅助实例 $u'$：

   $$
   u' = u + \text{rand}(0,0.3) \times (u - v)
   $$

4. 从 $K_P$ 中随机选取一个少数类实例 $x$。

5. 在 $x$ 和 $u'$ 之间合成一个少数类实例 $w$：

   $$
   w = x + \text{rand}(0,0.5) \times (u’ - x)
   $$

## SMOTE-Cosine 算法

假定整个训练集为 $T$, 其中少数类样本集记为 $P$（正例）, 多数类样本集记为 $N$ （负例）。

1. 对样本 $u \in P$, 分别计算其与 $v \in P$ 且 $v \neq u$ 的 Euclidean 距离和余弦相似度：

   $$
   \text{sim}(\vec{u},\vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\Vert \vec{u} \Vert \Vert \vec{v} \Vert}
   $$

2. 按照 Euclidean 距离，将所有 $v$ 增序排列为 $A$， 按照余弦相似度降序排列为 $B$。

3. 对 $A$ 和 $B$ 按照排序分别赋予一个权重，排名靠前的权重更大。然后将 $A$ 和 $B$ 的权重相加，根据组合后的权重排序决定 $u$ 的 $k$-近邻。


## Selected-SMOTE 算法

假定整个训练集为 $T$, 其中少数类样本集记为 $P$（正例）, 多数类样本集记为 $N$ （负例）。

1. 对样本集做特征选择，选出重要的特征。

2. 对样本 $x \in P$, 计算其在 $P$ 中的 $k$-近邻，并随机选取一个 $\hat{x}$ 合成新的样本。

3. 合成时，如果是重要的特征，按照标准的 SMOTE 算法进行插值合成，如果是非重要特征，直接复制 $x$ 的特征作为新样本的特征。


# References

1. `ADASYN` H. Haibo, B. Yang, E. A. Garcia, and L. Shutao, "[ADASYN: Adaptive synthetic sampling approach for imbalanced learning,](http://140.123.102.14:8080/reportSys/file/paper/manto/manto_6_paper.pdf)" in *2008 IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence),* **2008**, pp. 1322-1328.

2. `AND-SMOTE` J. Yun, J. Ha, and J.-S. Lee, "[Automatic Determination of Neighborhood Size in SMOTE,](https://dl.acm.org/citation.cfm?id=2857648)" presented at the *Proceedings of the 10th International Conference on Ubiquitous Information Management and Communication,* Danang, Viet Nam, **2016**.

3. `SMOTE-D` F. R. Torres, J. A. Carrasco-Ochoa, and J. F. Martínez-Trinidad, "[SMOTE-D a Deterministic Version of SMOTE,](https://link.springer.com/chapter/10.1007/978-3-319-39393-3_18)" in *Pattern Recognition: 8th Mexican Conference, MCPR 2016, Guanajuato, Mexico, June 22-25, 2016. Proceedings,* J. F. Martínez-Trinidad, J. A. Carrasco-Ochoa, V. Ayala Ramirez, J. A. Olvera-López, and X. Jiang, Eds., ed Cham: Springer International Publishing, **2016**, pp. 177-188. 

4. `SMOTE-Out, SMOTE-Cosine, Selected-SMOTE` F. Koto, "[SMOTE-Out, SMOTE-Cosine, and Selected-SMOTE: An enhancement strategy to handle imbalance in data level,](https://doi.org/10.1109/ICACSIS.2014.7065849)" in *2014 International Conference on Advanced Computer Science and Information System,* **2014,** pp. 280-284.




