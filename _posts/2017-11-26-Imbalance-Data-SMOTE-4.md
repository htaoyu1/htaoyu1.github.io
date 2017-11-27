---
layout: post
title: 非平衡数据集的处理：SMOTE 类算法（4）
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


# Cluster-Based Oversampling

## 算法 

假定整个训练集为 $T$, 其中少数类样本集记为 $P$（正例）, 多数类样本集记为 $N$ （负例）。

1. 使用 K-Means++ 算法，对整个训练集 $T$ 分为 $K$ 个聚类，记为 $C_1, C_2, \cdots, C_k$。

2. 计算集簇 $i$ 之内所有样本之间的距离之和 $D_i = \sum_{x_A, x_B \in C_i} d(x_A, x_B)$

3. 计算所有集簇的簇间方差 $W_K = \sum_{i=1}^{K}\frac{D_i}{2N_i}$, 其中 $N_i = \vert C_i \vert$ 为第 $i$ 个集簇包含的样本个数。选取导致 $W_K$ 突降的 $K$ 值作为最优集簇个数。

4. 选取第二大集作为参考集簇（这个标准是根据初步试验结果选定的），对每个规模小于参考集簇的集簇 SMOTE 合成新样本。

5. 在标准的 SMOTE 算法中，新的样本只使用少数类样本产生，因此合成的样本也都被标记为少数类。而这里因为每个集簇内即包含少数类样本，也包含多数类样本，因此样本的标签需要根据随机数 $\sigma$ 的大小来决定。具体做法为：

   1. 如果不要求对集簇内所有样本都进行 SMOTE 过采样，那么随机选取一个样本点 $x$。

   2. 在 $x$ 的 $k$-近邻里随机选取一个样本点 $\hat{x}$ 进行插值

      $$
      x_{\text{new}} = x + \delta \times (\hat{x}-x)
      $$
      
      其中 $\delta = \text{rand}(0,1)$ 为 0 到 1 间的随机数。如果 $\delta \leq 0.5$ 将 $x_{\text{new}}$ 标记为与 $x$ 相同的类，如果 $\delta > 0.5$，将 $x_{\text{new}}$ 标记为与 $\hat{x}$ 相同的类。


![4-1 过采样第一阶段：产生平衡数据集](/assets/blog-images/Imbalance-Data-SMOTE-4.1.png)
**Fig. 4-1:** 过采样第一阶段：产生平衡数据集。

6. 使用步骤 1-5， 使用不同的随机数种子，产生 $R$ 个候选集 $D_1^\ast, D_2^\ast, \cdots, D_R^\ast$ （Fig. 4-1）。

7. 从这 $R$ 个数据集中每个随机采样 20%, 合成新的代表性数据集 $M$（Fig. 4-2）。

![4-2 过采样第二阶段：构建代表性数据集](/assets/blog-images/Imbalance-Data-SMOTE-4.2.png)
**Fig. 4-2:** 过采样第二阶段：构建代表性数据集。



# K-Means SMOTE

SMOTE 算法以等概率随机选取少数类样本合成数据，可以处理类间不平衡，却<u>不能处理类内不平衡和离散的小集簇</u>。因为数据点密集的区域被选中的概率较大，而稀疏区域被选中产生新样本的概率相对较小。此外，SMOTE 还可能会放大数据集的噪声；并且因为所有点被等概率选中，不能增强边界点数据的合成。

`K-Means SMOTE` （1）通过在安全区合成数据避免噪声的产生；（2）同时考虑到类内和类间不平衡，通过在少数类稀疏区域合成更多的样本来克服离散集簇的问题；（3）算法简单易行，复杂度地，基于集簇密度来分配不同集簇内合成样本的树木。

![4-3 K-Means SMOTE 算法](/assets/blog-images/Imbalance-Data-SMOTE-4.3.png)
**Fig. 4-3:** K-Means SMOTE 算法。

## 算法

假定整个训练集为 $T$, 其中少数类样本集记为 $P$（正例）, 多数类样本集记为 $N$ （负例）。

1. 聚类：通过 K-Means 算法将整个训练集 $T$ 聚类。

2. 过滤：计算每个集簇中少数类样本的比例，选取少数类样本比例超过 50% 的集簇进行过采样。

3. 计算倍增率： 对每个在步骤2过滤出来进行过采样的集簇分配权重因子，对少数类密度低的集簇给予更高的权重因子，以合成更多的新样本。具体做法是：

   1. 计算每个集簇内少数类样本间的平均 Eculidean 距离 $r_f$。

   2. 假定集簇内包含 $m$ 个少数类样本，则定义该集簇的密度 

      $$
      \rho_f = \frac{m}{r_f^m}
      $$
      
      , 集簇内样本的稀疏性定义为密度的倒数
      
      $$
      s_f = \frac{1}{\rho_f}
      $$
      
   3. 计算集簇的权重因子，权重因子定义为集簇的稀疏性与所有被选中做过采样的集簇稀疏性之和的比值：

      $$
      w_f = \frac{s_f}{\sum s_f}
      $$

4. 数据合成：对集簇 $f$, 随机选取集簇内的两个少数类样本，在其中间插值合成一个新的少数类样本，重复该过程直到产生 $w_f \times n$ 个新样本。其中 $n$ 为需要合成的样本总数。

# Selective-SMOTE

Borderline-SMOTE 算法认为<u>在边界合成更多的数据能有效地提升分类器的性能</u>。Borderline-SMOTE 算法根据少数类和多数类的相对分布来定义边界样本。`Selective-SMOTE` 算法将<u>边界样本定义为学习器易分错的少数类样本</u>。对于平衡的数据集，预测概率为 0.5 的样本被认为是一个易错分样本。对于非平衡数据集，Selective-SMOTE 算法将易错分样本的临界阈值定义为 $k = \vert P \vert / \vert N \vert$, 即少数类样本个数和多数类样本个数的比值。
 
 
## 算法
 
假定整个训练集为 $T$, 其中少数类样本集记为 $P$（正例）, 多数类样本集记为 $N$ （负例）。

1. 使用训练集 $T$ 训练预测模型 (可以是决策树，神经网络，支持向量机等)。

2. 利用步骤1训练出来的模型预测每个样本，假设第 $i$ 个样本的预测概率为 $p_i$。

3. 从少数类样本里选取 $p\% \times \vert P \vert$ 个预测概率接近 $k = \vert P \vert / \vert N \vert$ 的样本作为边界样本。其中 $p\%$ 为预定义的边界样本在少数类样本里的占比。假定选出的边界样本集合为 $D$。

4. 对边界点集合 $D$ 中的数据，使用 SMOTE 算法合成新的样本，产生合成样本集 $D_1$。

5. 将 $D_1$ 插入到原始训练集 $T$, 得到平衡数据集 $B$。


# SMOTEBoost


在标准的 Boosting 算法里，每一个被错分的样本都被赋予相同的权重。由于在非平衡数据集中大部分样本都属于多数类，因此在随后的采样中，训练集仍然可能会偏好多数类。`SMOTEBoost` 算法的目的在于降低由于类别不平衡导致的学习过程中的内在偏倚。

SMOTEBoost 算法在第 $t$ 轮训练时，在样本分布 $D_t$ 里 SMOTE 合成少数类样本。这意味着隐式增加了少数类样本的权重。在训练完毕得到弱分类器 $h_t$ 后，合成的样本将会被从数据集中删除，即合成的样本不会被添加到原始训练集中，在每轮的迭代中，都会从分布 $D_t$ 中合成新的样本。经过每轮提升迭代后，分类器错误率估计只会基于原始的训练集。每轮迭代都进行 SMOTE 合成新的少数类样本增加了样本的多样性，同时也增加了弱分类器的多样性。

SMOTEBoost 算法基于 AdaBoost.M2 算法。具体算法如下：

## 算法

给定训练集 $S = \{(x_1, y_1), \cdots, (x_m, y_m)\}$, $x \in X$, $y \in Y = \{1, \cdots, C\}$。少数类样本集为 $C_p, (C_p < C)$。 定义 $B = \{(i,y): i=1,\cdots, m, y\neq y_i\}$ 为被错分的样本集。

1. 初始化样本整体分布 $D_1 = 1/m$。

2. 对于每轮迭代 $t = 1, 2, \cdots, T$， 进行下列操作：

   1. 通过对少数类 $C_p$ 使用 SMOTE 算法合成新的样本来改变样本整体分布 $D_t$。

   2. 使用 $D_t$ 训练一个弱分类器 $h_t$。

   3. 计算弱分类器的假设： $h_t: X \times Y \to [0, 1]$。

   4. 计算 $h_t$ 的伪损失（pseudo-loss）:

      $$
      \epsilon_t = \sum\limits_{(i,y) \in B} D_t (i,y)(1 - h_t(x_i, y_i) + h_t(x_i, y))
      $$
      
   5. 设 $\beta_t = \epsilon_t / (1-\epsilon_t)$, $w_t = \frac{1}{2} \cdot (1 - h_t(x_i, y_i) + h_t(x_i, y)) $ 。

   6. 更新样本分布 $D_t$:

      $$
      D_{t+1}(i, y) = \frac{D_t (i,y)}{Z_t} \cdot \beta_t^{w_t}
      $$
      
      其中 $Z_t$ 为归一化因子。
      
3. 输出最终假设:

   $$
   H = \mathrm{arg}\ \underset{y \in Y}{\mathrm{max}} \sum\limits_{t=1}^T (\log \frac{1}{\beta_t}) \cdot h_t (x, y)
   $$
   

# C_SMOTE 

SMOTE 算法主要存在两个方面的问题: （1）在 $k$-近邻选择时，$k$ 值的选取具有一定的盲目性。(2) 容易产生分布边缘化问题。如果一个少数类样本处在少数类样本集的分布边缘，则由此样本和相邻样本产生的“人造”样本也会处在这个边缘，且会越来越边缘化，从而模糊了少数类样本和多数类样本的边界，而且使边界变得越来越模糊。 

`C_SMOTE` 算法的基本假设是: 同一类的样本应有一个共同的重心，一个类用它的重心做代表比较合理，在“人造"样本的过程中，新产生的样本也应向类的重心靠拢。


## 算法
 
假定整个训练集为 $T$, 其中少数类样本集记为 $P$（正例）, 多数类样本集记为 $N$ （负例）。

1. 计算少数类样本集 $P$ 的重心:

   $$
   \vec{P}_{com} = \sum\limits_{x \in P} \frac{\vec{x}}{\vert P \vert}
   $$

2. 对于样本 $\vec{x} \in P$, 使用下面公式合成新的样本：

   $$
   \vec{x}_{\text{new}} = \vec{x} + \text{rand}(0,1) \times (\vec{P}_{com} - \vec{x})
   $$

3. 使用欠抽样技术删除一些离少数类样本重心远的样本。

![4-4 C_SMOTE 算法](/assets/blog-images/Imbalance-Data-SMOTE-4.4.png)
**Fig. 4-4:** C_SMOTE 算法。需要注意的是，少数类（即本文中的正类）在作者博士论文的流程图中标记为“负类”。

# References

1. `Cluster-Based Oversampling` M. S. Santos, P. H. Abreu, P. J. García-Laencina, A. Simão, and A. Carvalho, "[A new cluster-based oversampling method for improving survival prediction of hepatocellular carcinoma patients,]()https://doi.org/10.1016/j.jbi.2015.09.012" *Journal of Biomedical Informatics,* vol. 58, pp. 49-59, **2015.**

2. `K-Means SMOTE` F. Last, G. Douzas, and F. Bacao. "[Oversampling for Imbalanced Learning Based on K-Means and SMOTE](http://adsabs.harvard.edu/abs/2017arXiv171100837L)" 	*arXiv:1711.00837*, **2017**. 

3. `Selective-SMOTE` S. Nguyen, J. Quinn, and A. Olinsky, "[An Oversampling Technique for Classifying Imbalanced Datasets,](http://www.emeraldinsight.com/doi/abs/10.1108/S1477-407020170000012004)" in *Advances in Business and Management Forecasting*, **2018**, pp. 63-80.

4. `SMOTEBoost` N. V. Chawla, A. Lazarevic, L. O. Hall, and K. W. Bowyer, "[SMOTEBoost: Improving Prediction of the Minority Class in Boosting,](https://link.springer.com/chapter/10.1007/978-3-540-39804-2_12)" in *Knowledge Discovery in Databases: PKDD 2003: 7th European Conference on Principles and Practice of Knowledge Discovery in Databases, Cavtat-Dubrovnik, Croatia, September 22-26, 2003. Proceedings,* N. Lavrač, D. Gamberger, L. Todorovski, and H. Blockeel, Eds., ed Berlin, Heidelberg: Springer Berlin Heidelberg, **2003,** pp. 107-119.

5. `C_SMOTE` 曹正凤, ["随机森林算法优化研究,"](http://cdmd.cnki.com.cn/Article/CDMD-11912-1014220587.htm) 博士, 首都经济贸易大学, **2014.**






