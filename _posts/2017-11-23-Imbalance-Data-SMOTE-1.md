---
layout: post
title: 非平衡数据集的处理：SMOTE 类算法（1）
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

# 类不均衡问题

`类不均衡`（Class Imbalance） 是指：属于某一类别的观测样本的数量显著少于其它类别。比如说一个二分类（Binary Classification）问题，1000个训练样本，理想的情况是正、负类样本的数量相差不多。而如果负类样本有995个、正类样本仅5个，就意味着存在类不平衡。 **不均衡率**（Imbalance Ratio, IR）定义为多数类样本的数量与少数类样本数量的比值。类不均衡问题在异常检测是至关重要的场景中很明显，例如电力盗窃、欺诈交易、罕见疾病识别等。在这种情况下，利用传统机器学习算法开发出的预测模型可能会存在偏差和不准确。

发生这种情况的原因是机器学习算法通常被设计成通过减少**整体误差**来提高准确率。它们并没有考虑类别的分布/比例或者是类别的平衡。诸如决策树和 Logistic 回归这些标准的分类算法会偏向于数量多的类别。它们往往会仅预测占数据大多数的类别。在总量中占少数的类别的特征就会被视为噪声，并且通常会被忽略。因此，与多数类别相比，少数类别存在比较高的误判率。比如考虑一个公用事业欺诈检测数据集：1000个训练样本中，欺诈观测（正类）20个，非欺诈观测（负类）980个，欺诈事件比例 = 2%。训练过程中在某次迭代结束后，模型把所有的样本都分为非欺诈，虽然分错了正类，但是所带来的损失实在微不足道，Accuracy 已经是98.0%。

对类不均衡问题的处理目前主要有**算法层面**的方法和**数据层面**的方法。数据层面的`随机欠采样`（Random Undersampling，RUS）技术通过随机删除多数类别的实例来达到平衡样本的目的，由于 RUS 在平衡数据的过程中删除了多数类别的实例,<u>容易丢失有价值的重要信息</u>。另一方面，对于类别严重不平衡的数据集，比如上面提到的公用事业欺诈检测数据集，由于1000个训练样本中只有20个实例属于少数类别，为了达到平衡类别的目的，使用 RUS 技术<u>需要删除大量的的负类实例</u>，这造成了资源的极大浪费。

数据层面的另一种技术叫`随机过采样`（Random Oversampling，ROS）技术。ROS 通过随机复制少数类别的实例来增加少数类样本的数量。ROS 算法由于仅仅通过简单的复制来达到平衡类别的目的，加大了**过拟合**的风险，并且由于在复制数据的过程中并没有增加任何有效的信息，在提升分类器的性能方面效果有效。

# SMOTE

`合成少数类过采样技术`（Synthetic Monitory Oversampling TEchnique，SMOTE）可用来避免 ROS 带来的过拟合风险。SMOTE 算法的基本思想是对少数类进行分析，并根据少数类样本人工合成新样本添加到数据集中。


![1-1 SMOTE 算法](/assets/blog-images/Imbalance-Data-SMOTE-1.1.png)
**Fig. 1-1:** SMOTE 算法。


## 算法

假定整个训练集为 $T$, 其中少数类样本集记为 $P$（正例）, 多数类样本集记为 $N$ （负例）。

1. 对于少数类样本集中每一个样本 $x \in P$, 以 Euclidean 距离为标准计算其到少数类样本集 $P$ 中所有样本的距离，得到其 $k$-近邻。

2. 根据样本不平衡比例设置一个采样倍增率 $m$, 对每一个 $x \in P$, 从其 $k$-近邻随机选取若干个样本。假设选择的近邻实例为 $\hat{x}$。

3. 对每一个随机选出的近邻 $\hat{x}$，与原样本按照如下公式构建新样本：

   $$
   x_{\text{new}} = x + \delta \times (\hat{x} - x)
   $$
   
   其中 $\delta = \text{rand}(0,1)$ 为 $0$ 到 $1$ 间的随机数。
   
> 注：虽然在 SMOTE 论文的算法描述里，好像是针对每一个特征均产生一个随机数，但我更倾向于认为 SMOTE 算法对所有特征均使用同一个随机数，因为这样才能保证新合成的样本不会侵入到 $x$ 和 $\hat{x}$ 连线之外的空间，造成类别的重叠（请参考 AND-SMOTE 算法）。这也符合其他论文对 SMOTE 算法的评论，即在两个样本点之间线性插值。

值得说明的是，上述算法只涉及到对连续特征（Continuous Features）的处理。在 [SMOTEBoost 论文里](https://link.springer.com/chapter/10.1007/978-3-540-39804-2_12)，作者提到了对非连续的名义特征（Nominal Features）的处理。对于同时包含连续和名义特征的数据，可以混合使用 Euclidean 距离和 Value Distance Metric （VDM）来计算样本的 $k$-近邻。合成新样本时，新样本的名义特征设置为 $x$ 与其 $k$-近邻里该特征占多数的值。如果找不到多数值，则随机选取一个值。

> 注：本系列的第4篇帖子讲到了 SMOTEBoost 算法。

## 优缺点
- 通过随机采样合成新的实例而非复制实例的副本，可以缓解过拟合的问题。
- 不会像 RUS 一样损失有价值的信息。
- 当合成新的实例时，SMOTE 没有考虑到其他类别的分布。如果少数类别的数据存在噪声，新合成的数据很容易落在多数类别中间，导致类重叠的增加（即引入了额外的噪声，见 Fig. 1-2）。
- SMOTE 算法需要搜索少数类实例的 $k$-近邻，如果数据的维度很高，很容易碰到所谓的`高维诅咒`（The Curse of Dimensionality）的问题。

![1-2 SMOTE 算法引入了额外的噪声](/assets/blog-images/Imbalance-Data-SMOTE-1.2.png)
**Fig. 1-2:** 由于少数类别（实心圆圈）噪声数据 A 被选中用来合成新的实例，SMOTE 算法合成的新实例（实心方块）落在了多数类别（空心圆圈）中间，导致了类重叠的增加。


# Borderline-SMOTE

SMOTE 算法不加区分地以等概率方式随机选取少数类实例产生新的样本。然而数据集可能包含噪声，噪声点合成数据可能会加剧样本的重叠（Fig. 1-2）。有很多算法针对这一问题进行了优化。其中一种是 `Borderline-SMOTE` 算法。Borderline-SMOTE 算法认为，大多数的分类算法都倾向于学习类别的边界点以及其附近的数据，因此<u>在边界点附近合成更多的数据</u>可以有效的提高分类器的性能。

![1-3 Borderline-SMOTE 算法](/assets/blog-images/Imbalance-Data-SMOTE-1.3.png)
**Fig. 1-3:** Borderline-SMOTE 算法。

## 算法

假定整个训练集为 $T$, 其中少数类样本集记为 $P$（正例）, 多数类样本集记为 $N$ （负例）。

1. 对于少数类样本集中每一个样本 $x \in P$, 计算 $x$ 在整个训练集 $T$ 中的 $m$-近邻。这 $m$ 个近邻样本中属于少数类样本集 $P$ 的实例个数记为 $m'$。显然有 $0 \leq m' \leq m$。

2. 如果 $m' = m$, 即 $x$ 的所有 $m$-近邻都属于多数类实例，那么点 $x$ 被认为是噪声点，接下来不再考虑。

3. 如果 $\frac{m}{2} \leq m' \lt m$, 即 $x$ 的 $m$-近邻超过半数为多数类实例，那么 $x$ 被认为是一个易错分的边界点，将其归为 **DANGER** 类。

4. 如果 $0 \leq m' \lt \frac{m}{2}$, 即 $x$ 的 $m$-近邻超过半数为少数类实例，那么 $x$ 被认为是一个安全点，将其归为 **SAFE** 类，并参与下面的运算。

5. 只使用边界点合成新的少数类实例。有以下两种策略：

   - `Borderline-SMOTE1`: 对于每一个 $x \in \text{ DANGER}$, 寻找其在少数类样本集 $P$ 中的 $k$-近邻，然后随机选取其中一个 $k$-近邻实例 $\hat{x}$，使用以下公式合成新的少数类实例：
     
     $$
     x_{\text{new}} = x + \delta \times (\hat{x} - x)
     $$
   
     其中 $\delta = \text{rand}(0,1)$ 为 $0$ 到 $1$ 间的随机数。

   - `Borderline-SMOTE2`: 对于每一个 $x \in \text{ DANGER}$, 寻找其在整个训练集 $T$ 中的 $k$-近邻，然后随机选取其中一个 $k$-近邻 $\hat{x}$。如果 $\hat{x} \in P$, 则使用 Borderline-SMOTE1 的公式合成新的少数类实例；如果 $\hat{x} \in N$, 则使用下面的公式合成新的少数类实例：
     
     $$
     x_{\text{new}} = x + \delta \times (\hat{x} - x)
     $$
   
     其中 $\delta = \text{rand}(0.5,1)$ 为 $0.5$ 到 $1$ 间的随机数。


## 优缺点
- 边界附近的点容易被错分，Borderline-SMOTE 关注到了这个问题， 在一定程度上提高了少数类样本的识别率。
- 算法要求调整 $m$ 值，使边界样本占比适中。若边界样本选取过少，而合成的样本数量过多，会导致边界样本重叠的现象。若边界样本选取过多，则人为地放大了边界范围，由于合成边界附近的少数类样本并不能准确判断极性，不能真正地增加边界少数类样本。
- 忽略了对噪声点的处理，可能会影响少数类样本的分类性能。



# Modified-SMOTE

`Modified-SMOTE` 是另外一种改进 SMOTE 缺点的算法。与 Borderline-SMOTE 算法类似，
Modified-SMOTE 算法也少数类样本分为 3 个不同的组：安全样本（Security）、边界样本（Border）和潜在噪声样本（NOISE）。其与 Borderline-SMOTE 稍微不同的地方在于对三类点的定义和处理。具体算法如下：

## 算法

假定整个训练集为 $T$, 其中少数类样本集记为 $P$（正例）, 多数类样本集记为 $N$ （负例）。

1. 对于少数类样本集中每一个样本 $x \in P$, 计算 $x$ 在整个训练集 $T$ 中的 $k$-近邻。这 $k$ 个近邻样本中属于多数类样本集 $N$ 的实例个数记为 $k'$。显然有 $0 \leq k' \leq k$。

2. 如果 $k' = k$, 即 $x$ 的所有 $k$-近邻都属于多数类实例，那么点 $x$ 被认为是噪声点，归为 **NOISE** 组。

3. 如果 $k' = 0$, 即 $x$ 的所有 $k$-近邻都属于少数类实例，那么将 $x$ 归为 **SECURITY** 组。

4. 如果 $x$ 即不属于 **NOISE** 组，又不属于 **SECURITY** 组，那么将其归为 **BORDER** 组。

5. 对于噪声样本，不做任何处理。如果 $x \in \text{SECURITY}$, 从其 $k$-近邻随机选取一个样本；如果 $x \in \text{BORDER}$, 则选取其最近邻样本。假设被选中的样本为 $\hat{x}$, 按照如下公式构建新的样本：

   $$
   x_{\text{new}} = x + \delta \times (\hat{x} - x)
   $$
   
   其中 $\delta = \text{rand}(0,1)$ 为 $0$ 到 $1$ 间的随机数。



# Safe-Level SMOTE

Borderline-SMOTE 算法的一个缺陷在于对边界点的定义太过粗糙。考虑两个少数类样本，假定它们的 $k$-近邻分别包含了 $k$ 和 $k-1$ 个多数类实例，那么 Borderline-SMOTE 算法会将前者归为噪声样本，而将后者归位边界样本。然而从实际上考虑，这两个样本并没有太大的差异。`Safe-Level SMOTE` 算法针对 Borderline-SMOTE 的这一缺陷进行了优化。

## 算法

假定整个训练集为 $T$, 其中少数类样本集记为 $P$（正例）, 多数类样本集记为 $N$ （负例）。

1. 对于少数类样本集中每一个样本 $x \in P$, 计算 $x$ 在少数类样本集 $P$ 中的 $k$-近邻，并从中随机选取一个近邻点 $\hat{x}$。

2. 分别计算 $x$ 和 $\hat{x}$ 在整个样本集 $T$ 中的 $k$-近邻 (记为 $N_x^k$，和 $N_\hat{x}^k$), 然后根据计算出的 $k$-近邻，分别定义其 Safe Level $sl_x$ 和 $sl_\hat{x}$。对任意一个样本 $p$, 其 Safe Level 定义为 $p$ 的 $k$-近邻里包含的少数类样本数目，即：
   $$
   sl_p = \vert N^k_p \vert
   $$

3. 计算样本 $x$ 的 Safe Level Ratio $slr_x$。$slr_x$ 定义为$x$ 以及其在第1步中被随机选中的一个 $k$-近邻点 $\hat{x}$ 的 Safe Level 的比值，即:

   $$
   slr_x = \frac{sl_x}{sl_{\hat{x}}}
   $$
   
   如果 $ sl_x \to 0$, 则 $x$ 近似为噪声，如果 $sl_x \to k$, 则 $x$ 被认为是一个安全点。分以下5种情况考虑：

   - 如果 $slr_x = \infty$（即 $sl_{\hat{x}} = 0$， $\hat{x}$ 的 $k$-近邻全部属于多数类）并且 $sl_x = 0$, 则 $x$ 和 $\hat{x}$ 均属于噪声点，不做任何操作。

   - 如果 $slr_x = \infty$ 并且 $sl_x \neq 0$, 则 $\hat{x}$ 属于噪声点，复制 $x$ 的一个副本作为新合成的少数类实例（因为此时我们希望新合成的样本离噪声 $\hat{x}$ 尽可能地远）。

   - 如果 $slr_x =  1$ ，则 $x$ 和 $\hat{x}$ 的 Safe Level 一样，使用公式

      $$
      x_{\text{new}} = x + \text{rand}(0,1) \times (\hat{x} - x)
      $$
   
      在 $x$ 和 $\hat{x}$ 间合成一个新的少数类实例。
            
   - 如果 $slr_x >  1$ ，则意味着 $x$ 的 Safe Level 高于 $\hat{x}$。在 $x$ 和 $\hat{x}$ 间更靠近 $x$ 的一侧合成一个新的少数类实例：

      $$
      x_{\text{new}} = x + \text{rand}(0, \frac{1}{slr_x}) \times (\hat{x} - x)
      $$
   
   - 如果 $slr_x <  1$ ，则意味着 $\hat{x}$ 的 Safe Level 高于 $x$。在 $x$ 和 $\hat{x}$ 间更靠近 $\hat{x}$ 的一侧合成一个新的少数类实例：

      $$
      x_{\text{new}} = x + \text{rand}(1 - slr_x, 1) \times (\hat{x} - x)
      $$


# LN-SMOTE

SMOTE 算法合成新的少数类实例时没有考虑多数类样本的分布。Safe-Level SMOTE 算法虽然针对这一问题进行了优化，但是可能仍然不够充分，尤其是少数类实例在整个样本空间形成分散集簇的情况下。这种情况被称为 **Small Disjuncts** 问题。

![1-4 Small-Disjuncts 问题](/assets/blog-images/Imbalance-Data-SMOTE-1.4.png)
**Fig. 1-4:** 少数类实例在样本空间形成分散的集簇导致的过泛化问题。

考虑 Fig. 1-4 的情况，少数类实例在整个样本空间形成两个分离的集簇。在 Safe-Level SMOTE 算法的第一步，我们需要寻找一个少数类样本 $x$ 在整个少数类集合 $P$ 中的 $k$-近邻。如果 $k = 5$, 那么被随机选中做数据合成的近邻点 $\hat{x}$ 就有可能落在另一个集簇。如果 $x$ 和 $\hat{x}$ 的 Safe Level 非常相似的话， 新合成的少数类实例就很可能落在多数类的空间，造成类的重叠。

`LN-SMOTE`（Local Neighbourhood SMOTE）算法认为导致这一缺陷的原因是 Safe-Level SMOTE 在搜索 $x$ 的 $k$-近邻时，只局限于少数类样本集。为了克服这一问题，LN-SMOTE 提出需要在整个训练集的 $k$-近邻中选择 $\hat{x}$。

另外一个值得注意的情况是，假设 $x$ 属于噪声点，则其 Safe Level 等于 0 ($sl_x = 0$)。那么选出来的 $\hat{x}$ 必定属于多数类。然而由于 $x$ 的存在，$\hat{x}$ 的 Safe Level 可能等于 1 （$sl_{\hat{x}} = 1$）。按照 Safe-Level SMOTE 算法，此时需要复制 $\hat{x}$ 作为新合成的数据, 这是不合理的。LN-SMOTE 指出，碰到这种情况，需要排除掉 $x$, 转而搜索 $\hat{x}$ 的 $k+1$-近邻。


## 算法

假定整个训练集为 $T$, 其中少数类样本集记为 $P$（正例）, 多数类样本集记为 $N$ （负例）。

1. 对 $x \in P$, 在整个训练集 $T$ 中计算其 $k$-近邻，并随机选取一个近邻点 $\hat{x}$。

2. 分别计算 $x$ 和 $\hat{x}$ 的 Safe Level，以及 $x$ 的 Safe Level Ratio。需要注意的是，如果 $\hat{x} \in N$, 则在计算 $\hat{x}$ 的 $k$-近邻时要排除掉 $x$。

3. 如果 $sl_x = sl_{\hat{x}} = 0$, 则不做任何操作。否则，按照以下4种情况产生一个随机数 $\delta$：

   -  如果 $sl_{\hat{x}} = 0$ 并且 $sl_x > 0$ （即 $\hat{x}$ 的 $k$-近邻全为多数类实例，而 $x$ 的 $k$-近邻包含有少数类实例），则 $\delta = 0$。

   -  如果 $slr_{x} = 1$ ， 则 $\delta = \text{rand}(0,1)$。

   -  如果 $slr_{x} > 1$ ， 则 $\delta = \text{rand}(0,\frac{1}{slr_x})$。

   -  如果 $slr_{x} < 1$ ， 则 $\delta = 1 - \text{rand}(0,slr_x)$。

   
4. 如果 $\hat{x} \in N$, 则收缩 
   
   $$
   \delta = \delta \times \frac{sl_x}{k} 
   $$
   
5. 按照如下公式构建新的样本：

   $$
   x_{\text{new}} = x + \delta \times (\hat{x} - x)
   $$


# References

1. `SMOTE` N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "[SMOTE: synthetic minority over-sampling technique,](http://dx.doi.org/10.1613/jair.953)" *Journal of artificial intelligence research*, vol. 16, pp. 321-357, **2002**.

2. `Borderline-SMOTE` H. Han, W.-Y. Wang, and B.-H. Mao, "[Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning,](https://doi.org/10.1007/11538059_91)" in *Advances in Intelligent Computing: International Conference on Intelligent Computing, ICIC 2005, Hefei, China, August 23-26, 2005, Proceedings, Part I,* D.-S. Huang, X.-P. Zhang, and G.-B. Huang, Eds., ed Berlin, Heidelberg: Springer Berlin Heidelberg, **2005**, pp. 878-887.

3. `MSMOTE` S. Hu, Y. Liang, L. Ma, and Y. He, "[MSMOTE: Improving Classification Performance When Training Data is Imbalanced,](https://doi.org/10.1109/WCSE.2009.756)" in *2009 Second International Workshop on Computer Science and Engineering,* **2009**, pp. 13-17.

4. `Safe-Level SMOTE` C. Bunkhumpornpat, K. Sinapiromsaran, and C. Lursinsap, "[Safe-Level-SMOTE: Safe-Level-Synthetic Minority Over-Sampling TEchnique for Handling the Class Imbalanced Problem,](https://link.springer.com/chapter/10.1007/978-3-642-01307-2_43)" in *Advances in Knowledge Discovery and Data Mining: 13th Pacific-Asia Conference, PAKDD 2009 Bangkok, Thailand, April 27-30, 2009 Proceedings,* T. Theeramunkong, B. Kijsirikul, N. Cercone, and T.-B. Ho, Eds., ed Berlin, Heidelberg: Springer Berlin Heidelberg, **2009**, pp. 475-482.

5. `LN-SMOTE` T. Maciejewski and J. Stefanowski, "[Local neighbourhood extension of SMOTE for mining imbalanced data,](https://doi.org/10.1109/CIDM.2011.5949434)" in *2011 IEEE Symposium on Computational Intelligence and Data Mining (CIDM),* **2011**, pp. 104-111.


