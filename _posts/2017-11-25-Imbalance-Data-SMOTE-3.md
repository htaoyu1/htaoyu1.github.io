---
layout: post
title: 非平衡数据集的处理：SMOTE 类算法（3）
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

# Cluster-SMOTE

## 算法

假定整个训练集为 $T$, 其中少数类样本集记为 $P$（正例）, 多数类样本集记为 $N$ （负例）。

1. 首先使用 K-Means 算法聚类少数类样本 $P$。

2. 对每个聚类的集簇，使用 SMOTE 算法合成新的少数类样本点。

3. 将合成的数据插入到整个训练集 $T$。


# Cluster-Based Oversampling

Cluster-Based Oversampling (CBO) 试图同时解决类内和类间的不平衡。技术上来说，CBO 不属于 SMOTE 类算法，应该属于 ROS 类算法。

## 算法 

假定整个训练集为 $T$, 其中少数类样本集记为 $P$（正例）, 多数类样本集记为 $N$ （负例）。

1. 使用 K-Means 算法，对 $P$ 和 $N$ 内的样本分别聚类。

2. 对每个多数类集簇（除了最大的一个），都过采样到与最大集簇相等的规模。

3. 对每个少数类集簇也都过采样到相等的规模，规模的大小由合成后的多数类样本总数目与少数类集簇个数的比值决定。

例如，假设一个数据集包含60个样本，其中多数类样本54个， 少数类样本6个。经过聚类后，多数类集簇的规模分别为 10, 10, 10, 24, 少数类集簇的规模分别为2，3，2。那么经过 Cluster-Based Oversampling 之后，多数类集簇的规模应该为 24, 24, 24, 24, 少数类集簇的规模为 32, 32, 32，即多数类和少数类的样本个数都被过采样到96个。

# BDSK 

`BDSK` (Bi-Directional Sampling based on K-means) 是另一种试图同时解决类内和类间不平衡的算法。BDSK 同时使用了欠采样和 SMOTE 过采样技术，具体算法如下：

## 算法 

1. 假定数据集包含 $m$ 个类别 $\{ C_1, C_2, \cdots, C_m\}$, 每个类别的大小 $n_i = \vert C_i \vert$。

2. 计算类别的平均大小 $k = \sum_{i=1}^m n_i / m$。

2. 对于类别 $C_i$：

   1. 如果 $\vert C_i \vert > k$， 则 $C_i$ 为多数类。使用 K-means 算法将数据聚集为 $k$ 个集簇，选择每个集簇中心的最近邻点集合成新的 $C_i$ 类别集。

   2. 如果 $\vert C_i \vert < k$， 则 $C_i$ 为少数类。使用 K-means 算法将数据聚集为2个集簇，选择较小的集簇，使用 SMOTE 技术合成数据，直到 $\vert C_i \vert \approx k$ 。


# SNOCC 

基于 SMOTE 算法合成的样本并不能真正地反映原样本的分布。SMOTE 算法通过在两个样本 $x$ 和 $\hat{x}$ 之间插值合成新的样本，所以新合成的样本只能分布在两个样本之间的连线上，并不能覆盖样本分布的整个区域（见下图）。

![3-1 基于 SMOTE 的算法不能覆盖整个样本的分布空间](/assets/blog-images/Imbalance-Data-SMOTE-3.1.png)
**Fig. 3-1:** 基于 SMOTE 的算法不能覆盖整个样本的分布空间。

鉴于此，Zheng 等人提出了基于凸组合(Convex Combination)的过采样技术 `SNOCC` (Sigma Nearest Oversampling based on Convex Combination)。凸组合是一类线性组合，其所有组合系数均非负且归一。即给定有限个实空间的矢量对象 $x_1, x_2, \cdots, x_n$, 这些点构成的凸组合为：

$$
x = \sum\limits_{i=1}^n \alpha_i x_i
$$ 

其中 $\alpha_i$ 为实数，且满足 $ \alpha_i > 0$ 和 $\sum\limits_{i=1}^n \alpha_i = 1 $


## 算法 

假定整个训练集为 $T$, 其中少数类样本集记为 $P$（正例）, 多数类样本集记为 $N$ （负例）。

1. 对每个 $x_i \in P$, 在少数类样本集 $P$ 中计算其 $k$-近邻。

2. 计算 $x_i$ 到其所有 $k$-近邻样本的平均距离 $m_i$。

3. 计算所有 $m_i$ 的标准差为 $s$，记 $\sigma_i = m_i + s$。

4. 在 $x_i$ 的 $k$-近邻样本中寻找所有与 $x_i$ 的距离小于 $\sigma_i$ 的近邻。这些近邻被称为 $x_i$ 的 $\sigma$-近邻。

5. 随机从 $x_i$ 的 $\sigma$-近邻里选取两个近邻点 $\hat{x}_1$ 和 $\hat{x}_2$。

6. 生成一个三维非负随机矢量 $\alpha = (\alpha_1, \alpha_2, \alpha_3)$，并归一化使 $\alpha_1 + \alpha_2 + \alpha_3 = 1$。


## 优缺点

- SNOCC 与 SMOTE 相比，拓展了生成样本的空间。SMOTE 只能在两个种子样本之间的连线上生成新的样本，而 SNOCC 可以在种子样本围成的凸包(Convex Hull)内生成样本。

- SMOTE 还有可能造成类别的重叠。SNOCC 使用 $\sigma$-近邻代替了 SMOTE 的 $k-$近邻。这种替代可以有效提高过采样的有效性。

- SNOCC 只能处理连续（Continuous）特征或有序类别（Ordinal）特征, 不能处理无序类别（Categorical）特征和布尔（Boolean）特征。



# Random-SMOTE

`Random-SMOTE` 是另外一种试图拓展 SMOTE 样本生成空间的算法。SMOTE 只能在两个少数类样本间的连线上生成新样本（见 Fig. 3-1），合成的数据保留了样本密集或者稀疏的特征，所以并不能很好地预测稀疏区域的样本位置。

## 算法

假定整个训练集为 $T$, 其中少数类样本集记为 $P$（正例）, 多数类样本集记为 $N$ （负例）。

1. 对每个 $x \in P$, 在少数类样本集 $P$ 中计算其 $k$-近邻，并随机选取其中两个近邻点 $y_1$ 和 $y_2$。

2. 在 $y_1$ 和 $y_2$ 的连线上产生临时样本 $t_i$:

   $$
   t_i = y_1 + \text{rand}(0,1) \times (y_2-y_1)
   $$

3. 使用 $x$ 和临时样本 $t_i$ 合成新的少数类样本：

   $$
   x_{\text{new}} = x + \text{rand}(0,1) \times (t_i - x)
   $$

对于 $x$ 和 $y_1$， $y_2$ 的选取存在以下三种情况：

1. $x$ 和 $y_1$， $y_2$ 三个样本重合，此时 Random-SMOTE 退化为标准的 RUS 算法。

2. $y_1$ 和 $y_2$ 两个样本重合，此时 Random-SMOTE 退化为标准的 SMOTE 算法。

3. $x$ 和 $y_1$， $y_2$ 三个样本均不重合，此时 Random-SMOTE 合成的新样本落在这三个点围城的三角形区域内。


# References

1. `Cluster-SMOTE` D. A. Cieslak, N. V. Chawla, and A. Striegel, "[Combating imbalance in network intrusion datasets,](https://doi.org/10.1109/GRC.2006.1635905)" in *2006 IEEE International Conference on Granular Computing,* **2006,** pp. 732-737.

2. `Cluster-Based Oversampling` T. Jo and N. Japkowicz, "[Class imbalances versus small disjuncts,](https://dl.acm.org/citation.cfm?id=1007737)" *SIGKDD Explor. Newsl.,* vol. 6, pp. 40-49, **2004.**

3. `BDSK` J. Song, X. Huang, S. Qin, and Q. Song, "[A bi-directional sampling based on K-means method for imbalance text classification,](https://doi.org/10.1109/ICIS.2016.7550920)" in *2016 IEEE/ACIS 15th International Conference on Computer and Information Science (ICIS),*  **2016,** pp. 1-5.

4. `SNOCC` Z. Zheng, Y. Cai, and Y. Li, "[Oversampling Method for Imbalanced Classification,](http://www.cai.sk/ojs/index.php/cai/article/viewArticle/1277)" *Computing and Informatics*, vol. 34, pp. 1017-1037, **2016.**

5. `Random-SMOTE` Y. Dong and X. Wang, "[A New Over-Sampling Approach: Random-SMOTE for Learning from Imbalanced Data Sets,](https://link.springer.com/chapter/10.1007/978-3-642-25975-3_30)" in *Knowledge Science, Engineering and Management: 5th International Conference, KSEM 2011, Irvine, CA, USA, December 12-14, 2011. Proceedings,* H. Xiong and W. B. Lee, Eds., ed Berlin, Heidelberg: Springer Berlin Heidelberg, **2011**, pp. 343-352.




