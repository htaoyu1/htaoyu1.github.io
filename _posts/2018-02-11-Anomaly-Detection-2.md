---
layout: post
title: 异常检测（2）：Mass Estimation
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


# 引言

在分类，聚类，异常检测，以及信息检索中，我们经常会用到**密度估算**（density estimation）。**质量估算**（mass estimation）与密度估计主要区别在以下几个方面：

1. 质量分布规定了数据群（data cloud）中从核心点（core points）到边缘点（fringe points）的一个序。该序使用一个凹函数（concave function）强调了边缘点 - 边缘点的质量明显小于接近核心点的数据点的质量。 这是许多任务所需的基本性质，包括异常检测和信息检索。 相比之下，密度估计不能提供数据的一个序。

2. 质量估算比密度估算更有效，因为质量是通过简单的计数计算得到的，并且使用系综方法只需要一个很小的样本量。 密度估算（通常用于估计 $p(x \vert j)$ 和 $p(j \vert x)$ ）为了得到较好的估计，则需要较大的样本量，并且在时间和空间复杂度方面的计算量很大。

3. 质量可以被解释为与数据基础概念相关的度量，即核心点表示它们高度相关，边缘点表示它们不太相关。 由数据估计的质量向量组成的相关特征空间对于三种数据挖掘任务非常有效：信息检索，回归以及异常检测。

质量估算在效能（efficacy）和效率（efficiency）方面有两个优势。 首先，凹性质确保了在质量空间边缘点被“拉伸”到更远离核心点，从而更容易将边缘点与接近核心点的数据点分开。 这个隐藏属性可以被用在数据挖掘算法中以达到更好的结果（在论文中[^Mass1] [^Mass2]，作者通过从原始空间到质量空间的简单映射，对信息检索和回归任务中四种现有算法在任务特定方面的效能实现了显著的提升）。

其次，当问题需要排序时，质量估算提供了从数据直接导出的序（没有距离或相关昂贵的计算），可以更有效地解决问题。 一个例子是异常检测，许多方法采用距离或密度来提供所需的排序，由于计算量太大，导致这些方法的效率比较低。 例如基于密度的异常检测 LOF 算法 （ $O(n^2)$ 复杂度）在涉及 50 万个数据点时，可能需要超过两周的时间; 而使用基于质量的异常探测器，在40秒内就可以完成。

# 质量与质量估算

数据的质量（又称质量）被定义为一个区域内数据点的数目; 并且与区域的特征（例如，密度，形状或体积）无关，两组数据可以具有相同的质量。一个给定区域内的质量由一个矩形函数定义，该矩形函数在质量被测量的整个区域中具有相同的值。识别由一组数据本身占据的区域本身是一个聚类问题，但是质量可以在不进行聚类的情况下，用类似于核密度估算的方式进行估计。需要注意的是，*质量不是概率质量函数，并不能提供概率的意义*。

## 质量分布估算

### 一阶（Level-1）质量分布估算

首先使用二元分裂将数据集分成两个不相交的区域，并计算每个区域的质量。 $x$ 点处的质量分布为来自 $x$ 占据的区域的所有“加权”质量的总和。

令 $x_1 < x_2 < \cdots < x_{n-1} < x_n$，$x_i \in \mathcal{R}$ 且 $n > 1$。令 $s_i$ 为 $x_i$ 和 $x_{i+1}$ 之间的二元分割点，将数据集分裂为两个分别具有质量 $m_i^L$ 和 $m_i^R$ 的非空区域。

**定义 1：** 质量基函数（mass base function）: 数据点 $x$ 的质量基函数 $m_i(x)$ 在点 $s_i$ 分割下，定义为

$$
m_i(x)= \left \{ 
\begin{array}{l@{\quad}l} m_i^L & \mbox{ 如果 } x \mbox{ 在 } s_i \mbox{ 的左边 } \\
m_i^R & \mbox{ 如果 } x \mbox{ 在 } s_i \mbox{ 的右边 }\\ \end{array} 
\right .
$$

并且我们有 $m_i^L = n - m_i^R = i$。

考虑下面的例子，我们有五个数据点。使用 $x_1 < s_1 < x_2$ 分割数据，我们有 

$$ 
m_1(x_1) = 1, \qquad m_1(x_2) = m_1(x_3) = m_1(x_4) = m_1(x_5) = 4
$$

使用 $x_2 < s_2 < x_3$ 分割数据：

$$ 
m_2(x_1) = m_2(x_2) = 2, \qquad m_2(x_3) = m_2(x_4) = m_2(x_5) = 3
$$

使用 $x_3 < s_3 < x_4$ 分割数据：

$$ 
m_3(x_1) = m_3(x_2) = m_3(x_3)= 3, \qquad  m_3(x_4) = m_3(x_5) = 2
$$

使用 $x_4 < s_4 < x_4$ 分割数据：

$$ 
m_4(x_1) = m_4(x_2) = m_4(x_3) = m_4(x_4) = 4, \qquad  m_4(x_5) = 1
$$

![2-1 质量基函数 $m_i(x)$](/assets/blog-images/Anomaly-Detection-2.1.png)
**Fig. 2-1.** 质量基函数 $m_i(x)$。

**定义 2：** 质量分布：数据点 $x_a \in \\{x_1, x_2, \cdots, x_{n-1}, x_n\\}$ 的质量分布函数 $mass(x_a)$ 定义为一系列质量基函数 $m_i(x)$ 以 $p(s_i)$ 为权重的加权和。其中 $p(s_i)$ 为选中 $s_i$ 作为分割点的概率 $p(s_i) = (x_{i+1} - x_i)/(x_n - x_1)$ （均匀分布下随机选取分割点）。

$$
\begin{eqnarray}
mass(x_a) & = & \sum\limits_{i=1}^{n-1} m_i(x_a) p(s_i) \nonumber \\
          & = & \sum\limits_{i=a}^{n-1} m_i^L p(s_i) + \sum\limits_{j=1}^{a-1} m_j^R p(s_j) \nonumber \\
          & = & \sum\limits_{i=a}^{n-1} i p(s_i) + \sum\limits_{j=1}^{a-1} (n-j) p(s_j) 
\label{eq:mass1}
\end{eqnarray}
$$

例如对于 $x_1$, 选取 $s_1$ 作为分割点的概率为 $p(s_1)$, 在 $s_1$ 分割下，$x_1$ 的质量基函数 $m_1(x_1) = 1$；选取 $s_2$ 作为分割点的概率为 $p(s_2)$, 在 $s_2$ 分割下，$x_1$ 的质量基函数 $m_2(x_1) = 2$；选取 $s_3$ 作为分割点的概率为 $p(s_3)$, 在 $s_3$ 分割下，$x_1$ 的质量基函数 $m_3(x_1) = 3$；选取 $s_4$ 作为分割点的概率为 $p(s_4)$, 在 $s_4$ 分割下，$x_1$ 的质量基函数 $m_4(x_1) = 4$。所以我们有：

$$
mass (x_1) = 1 p(s_1) + 2 p(s_2) + 3 p(s_3) + 4 p(s_4) 
$$

同理：

$$
\begin{eqnarray}
mass (x_2) & = & 4 p(s_1) + 2 p(s_2) + 3 p(s_3) + 4 p(s_4) \\
mass (x_3) & = & 4 p(s_1) + 3 p(s_2) + 3 p(s_3) + 4 p(s_4) \\
mass (x_4) & = & 4 p(s_1) + 3 p(s_2) + 2 p(s_3) + 4 p(s_4) \\
mass (x_5) & = & 4 p(s_1) + 3 p(s_2) + 2 p(s_3) + 1 p(s_4) \\
\end{eqnarray}
$$

如果一个数据点 $x \notin \\{ x\_1, x\_2 , \cdots , x\_{n-1} , x\_n \\}$ 且 $x\_i < x < x\_{i+1}$，$mass(x)$ 的值定义为 $mass(x\_i)$ 和  $mass(x\_{i+1})$ 的插值。


**定理 1：** $\\{x_1, x_2, \cdots, x_{n-1}, x_n\\}$ 的密度分布$mass(x_a)$ 的最大值位于 $a = n/2$ 处。对于 $x_1 < x_2 < \cdots < x_{n-1} < x_n$ 中的一个数据点 $x_a$， 有如下关系

$$
\begin{eqnarray}
mass (x_a) & < & mass(x_{a+1}), \quad a < n/2 \\
mass (x_a) & > & mass(x_{a+1}), \quad a > n/2
\end{eqnarray}
$$

> 证明见论文[^Mass1]


**定理 2：** 当 $p(s_i) = (x_{i+1} - x_i)/(x_n - x_1)$ 时，$mass(x_a)$ 是基于 $\\{x_1, x_2, \cdots, x_{n-1}, x_n\\}$ 定义的一个凹函数。

> 证明见论文[^Mass1]

**推论 1：** 使用二元分割估计的质量分布规定了基于质量的从 $x_{n/2}$（具有最大质量）到边缘点（在$x_{n/2}$ 的任一侧处具有最小质量）的一个序， 该序与数据的密度分布无关。

 **推论 2：** 质量分布的凹性质意味着，边缘点的质量明显小于靠近 $x_{n/2}$ 的点的质量。

推论 2 的含义是，边缘点在质量空间中比在原始空间更“远离”中值，从而更容易将边缘点与靠近中值的点分离出来（质量空间是通过 $mass(x$ 从原始空间映射得到的）。数据挖掘算法可以利用这个隐藏属性来提高其性能。 我们将会证明，这个简单的映射能够显着提高4.1节和4.2节中四个现有算法在信息检索和回归任务中的性能。

等式 $\ref{eq:mass1}$ 足以给出相应于单峰（unimodal）密度函数或均匀（uniform）密度函数的质量分布。 为了更好地估计多峰（multi-modal）分布，则需要高阶的质量分布。 


###  $h$ 阶（Level-$h$）质量分布估算

**定义 3：** 对于数据点 $x_a \in \\{ x\_1, x\_2 , \cdots , x\_{n-1} , x\_n \\}$，其 $h$ 阶（$h < n$）质量分布估计为：

$$
\begin{eqnarray}
mass(x_a, h) & = & \sum\limits_{i=1}^{n-1} mass_i(x_a, h-1) p(s_i) \nonumber \\
          & = & \sum\limits_{i=a}^{n-1} mass_i^L(x_a, h-1) p(s_i) + \sum\limits_{j=1}^{a-1} mass_j^R(x_a, h-1) p(s_j) 
\label{eq:massh}
\end{eqnarray}
$$

这里，高阶质量分布递归地使用低阶质量分布计算得到。在计算 $h > 1$ 阶质量分布时，我们使用一个分隔点  $s_i$ 将  $h-1$ 阶质量分布分成两个区域：（

1. 位于 $s_i$ 左边的质量分布 $mass_i^L(x，h-1)$。 使用 $\\{ x\_1, x\_2 , \cdots , x_i \\}$ 定义; 

2. 位于 $s_i$ 右边的质量分布 $mass_i^R(x，h-1)$。使用 $\\{ x\_{i+1}, x\_{i+2} , \cdots , x_n \\}$ 定义;

![2-2 计算 2 阶质量分布的例子](/assets/blog-images/Anomaly-Detection-2.2.png)
**Fig. 2-2.** 计算 2 阶质量分布的例子。

**Fig. 2-2** 显示了从包含 20 个点的数据集中计算 2 阶质量估计所需的 2 个分裂（总共有19个）。 每个分裂产生两个一阶质量估计：$mass^L_i(x，h = 1)$ 和 $mass^R_i(x，h = 1)$。 注意 1 阶质量分布是凹的（定理2）。这个例子显示了两个分裂 $si = 7$ 和 $s_i = 11$ 的结果，其中每个 1 阶质量分布是凹的。

> 这里讲一下我对计算高阶质量分布的理解。对于 **Fig. 2-2** 的例子，我们有数据点  $\\{ x\_1, \cdots , x\_{20} \\}$ 。如果我们需要计算 $mass(x_a, 2)$。那么这里我们有 19 种方式分割数据, 概率记为 $p(s_i)$。假设分割点 $s_i$ 选在了 $x_i$ 和 $x_{i+1}$ 之间, 我们有  $p(s_i) = (x_{i+1} - x_i)/(x_{20}-x_1)$。如果 $x_a \leq x_i$，则我们需要使用 $\\{ x\_1, \cdots , x_i \\}$ 来计算 $x_a$ 的一阶质量分布；如果 $x_a > x_i$，则需要使用 $\\{ x\_{i+1}, \cdots , x_{20} \\}$ 来计算 $x_a$ 的一阶质量分布。
> 
> 举例来说，假如我们要计算 $mass(x_{10}, 2)$。**Fig. 2-2 (a)** 中 $s_{i=7}$ 被选中的概率为 $p(s_i) = (x_{8} - x_7)/(x_{20}-x_1)$。因为 $x_{10}$ 在分割点的右边，所有我们需要使用 $\\{ x\_{8}, \cdots , x_{20} \\}$ 来计算 $mass_7^R(x_{10}, 1)$。按照 **Fig. 2-3** 的步骤，我们可以得到 $mass_7^R(x_{10}, 1)$ 的值:
> 
> ![2-3 计算 2 阶质量分布的过程](/assets/blog-images/Anomaly-Detection-2.3.png)
**Fig. 2-3.** 计算 2 阶质量分布的过程。
>
>$$
mass_7^R(x_{10}, 1) = 12 p'(s_8) + 11 p'(s_9) + 3 p'(s_{10}) + \cdots + 12 p'(s_{19})
$$
>
> 这里 $p'(s_i) = (x\_{i+1} - x_i)/(x_{20}-x_8)$。
>
> 同理，我们有
>
>$$
mass_{11}^L(x_{10}, 1) = 9 p'(s_1) + 8 p'(s_2) + 7 p'(s_3) +  \cdots + 1 p'(s_{9})
$$
>
>这里 $p'(s_i) = (x\_{i+1} - x_i)/(x_{10}-x_1)$。
>
>有了各个分割点下的一阶质量分布 $mass_i(x_a, 1)$，我们就可以使用 $\ref{eq:massh}$ 计算数据 $x_a$ 的 2 阶质量分布 $mass(x_a, 2)$ 了。

可以证明

$$
mass(x_{a+1}, h)= mass(x_a, h) + \left \{ 
\begin{array}{l@{\quad}l} 
[mass_a^R(x_a, h-1) - mass_a^L(x_a, h-1)] p(s_a), & h > 1 \\
(n-2a)p(s_a), & h = 1\\ 
\end{array} 
\right .
\label{eq:massh2}
$$

因此，只有 $x_1$  需要使用公式 $\ref{eq:massh}$ 估计，其他的都可以通过公式  $\ref{eq:massh2}$ 估算。这种方式的时间复杂度为 $O(n^{h+1})$。而如果直接使用公式 $\ref{eq:massh}$，时间复杂度为 $O(n^{h+2})$。


**定义 4：** $h$ 阶质量分布规定了数据群中从 $\alpha$-核心点到边缘点的一个序。 假设使用某个距离函数 $dist(\cdot,\cdot)$ 定义点 $x$ 的 $\alpha$ 邻域 $N_{\alpha}(x) = \\{y \in D \vert dist(x,y) \leq \alpha \\}$。 数据群中的每个 $\alpha$-核心点 $x^{\ast}$ 具有最高的质量值 $x \in N_{\alpha}(x^{\ast})$。 一个小的 $\alpha$ 值定义了局部核心点； 一个大的覆盖 $x$ 整个值范围的  $\alpha$ 值定义了全局核心点。


![2-4 $h$ 阶（$h=1,2,3$）质量分布以及核密度估计的密度分布。(a), (b), (c) 包含了 20 个数据点，（d）包含了 50 个数据点](/assets/blog-images/Anomaly-Detection-2.4.png)
**Fig. 2-4.** $h$ 阶（$h=1,2,3$）质量分布以及核密度估计的密度分布。(a), (b), (c) 包含了 20 个数据点，（d）包含了 50 个数据点。

**Fig. 2-4** 显示了核密度估计与 $h$ 阶质量估计的比较。$h = 1$ 质量估计将数据看作一个整体，生成了一个凹函数。 因此，无论底层密度分布如何，$h = 1$ 质量估计的全局核心点总是在中值处。


对于 $h>1$ 的质量分布，尽管总体上来说，不能再保证质量分布的凹性质，但是模拟表明，数据裙中的每个簇（如果存在的话）都表现为一个凹函数，随着 $h$ 的增加, 该凹性质变得更加明显。 **Fig. 2-4(b)** 显示了一个三态密度分布的例子。 $h> 1$ 质量分布对于某个 $\alpha$ （例如 0.2）具有三个  $\alpha$-核心点。


传统上，通过使用密度或距离（非均匀密度分布），可以在一定程度上估计非均匀分布数据的核心性质或边缘性。质量允许在任何分布中不进行密度或距离计算（需要很大的计算开销）的条件下进行估计。例如，**Fig. 2-4(c)** 是一个偏斜密度分布，除非计算距离，否则仅仅使用密度的近边缘点和远边缘点之间的区别并不明显。相比之下，质量分布使用不同边缘点的质量值，描绘了它们与 $x_{median}$ 的相对距离。


在 **Fig. 2-4(d)** 的例子中，有比正常点（图左侧较大的簇）更密集的异常簇。在基于密度的异常检测异常算法中，异常值被定义为具有低密度的点。所以基于密度的异常检测会认为簇中的所有这些异常点比正常点更“正常”。与之形成鲜明对比的是，$h = 1$ 的质量估计将正确地将它们排列为具有第三低质量值的异常点。这些点被解释为离数据群中具有较高质量值的正常点比较远的边缘点。


本节从理论角度描述了质量分布的性质。虽然可以使用公式 $\ref{eq:mass1}$ 和 $\ref{eq:massh}$ 来估计质量分布，但是其受限于较高的计算成本。下一小节提出了一个实践中进行质量估算的方法。以下的表述不再区分“质量估算”和“质量分布估算”。


## 实践中进行质量估算

在实践中，数据量一般都非常大，因此使用数据集中所有的数据点进行质量估算会造成很大的计算开销。一个可行的办法是我们可以对数据集进行子采样，并基于采样的数据集对数据点的质量进行一个近似的估算。

**定义 5：** $mass(x, h \vert \mathcal{D})$ 是一个数据点 $x \in \mathcal{R}$ 的近似质量分布。该近似是基于数据集 $\mathcal{D} = \\{x_1, \cdots, x_{\psi} \\}$ 估计的。其中 $\mathcal{D}$ 是给定数据集 $D$ 的一个随机子集，$\psi \ll \vert D \vert$，$h < \psi$。


![2-5 使用 5 个子采样的数据点进行 $mass(x, h \vert \mathcal{D})$ 估计。](/assets/blog-images/Anomaly-Detection-2.5.png)
**Fig. 2-5.** 使用 5 个子采样的数据点进行 $mass(x, h \vert \mathcal{D})$ 估计。


使用子采样计算完 $\mathcal{D}$ 中数据点的质量分布后，我们可以使用矩形函数来定义 $D$ 中数据的质量分布。对于每个 $x_i \in \mathcal{D}$, $x_i$ 定义了一个矩形区域，该区域覆盖的范围为 $(x\_{i-1} + x_i)/2 \leq x < (x_i + x\_{i+1})/2$。在 $D$ 中所有落在该区域内的数据点的质量都等于 $mass(x_i, h \vert \mathcal{D})$。对于端点的数据点，范围被设置为在该点的两侧具有相等的长度。 例如 **Fig. 2-5(a)** 中 $x_5$ 覆盖的范围为 $(x\_{4} + x_5)/2 \leq x < x_5 + (x_5 - x\_{4})/2$。 

为了更精确地估计一个数据点 $x$ 的质量分布，需要对数据集 $D$ 进行多次子采样，$x$ 的质量分布估计为在这些子集 $\mathcal{D}_k$ 内的平均值

$$
\overline{mass}(x,h) = \frac{1}{c} \sum\limits_{k=1}^c mass(x,h \vert \mathcal{D}_k)
$$


使用给定数据集 $D$，$mass(x,h)$ 估计的时间复杂度为 $O(\vert D \vert ^{h+1})$；而 $\overline{mass}(x, h)$ 的时间复杂度为 $O(c \psi^{h+1})$。


实例之间的序需要相对质量来确定。 对于 $h=1$，因为相对质量是基于中位数的，中位数是一个健壮的估计量，所以一个小的子样本就能够为排序提供一个较好的估计。 然而对于 $h > 1$, 中位数的概念是不确定的，但是实证结果表明。 对于 $h > 1$ 的情况，使用子样本的质量估计也能产生良好的结果。


为了比较 $mass(x,1)$ 和 $mass(x, 1 \vert \mathcal{D}$ 的性能，**Fig 2-5(b)** 比较了两个独立数据集中基于质量值的排序结果：一维高斯密度分布和 COREL 数据集，每个数据集包含 10000 个数据点。 **Fig 2-5(b)**显示了 $mass(x,1)$ 和 $mass(x, 1 \vert \mathcal{D}$ （$\psi = 8$）给出的序之间的相关性（以斯皮尔曼等级相关系数度量）。 可以看出，当 $c \geq 100$ 时，两种方法得到的序具有非常高的相关性。可以使用小样本而非大样本是质量估算的关键特征。

# Mass-Based Formalism

令 $x_i = (x_i^1, \cdots, x^u_i]$， $x_i \in D$; $z_i = (z_i^1, \cdots, z_i^t)$, $z_i \in D'$。拟议的公式由下面三部分组成：

- **C1：** 第一步，构建一系列质量分布。 在 **定义 5** 中给出了一个质量分布 $mass(x^d, x \vert \mathcal{D})$ 的估计方式，我们可以从数据集 $D$ 中随机选取一个子集 $\mathcal{D}_k$，然后随机从 $u$ 个维度中随机选取一个维度 $d$，得到对 $\mathcal{D}_k$ 的第 $d$ 维的质量分布的一个估计。重复这个过程 $t$ 次，我们生成 $t$ 个质量分布，形成 $\widetilde{\mathbf{mass}}(\mathbf{x}) \to \mathcal{R}^t$ 的一个映射，其中  $t \gg u$ （**算法 1**）。

![算法 1](/assets/blog-images/Anomaly-Detection-2.6.png)
**算法 1**


- **C2：** 第二个步将数据集 $D$ 从 $u$ 维的原始空间中，使用$\widetilde{\mathbf{mass}}(\mathbf{x}) \to \mathbf{z}$ 将其映射到 $t$ 维的新数据集 $D'$ 中（**算法 2**）。

![算法 2](/assets/blog-images/Anomaly-Detection-2.7.png)
**算法 2**


- **C3：** 第三部分采用决策规则来确定任务的输出。 这是在新的特征空间 $\mathbf{z}$ 中应用取决于任务特定的决策函数。


在不同的任务中，上面三个步骤中的 **C1** 和 **C3** 是必须的。取决于任务任务， **C2** 是可选的。

对于信息检索和回归，任务特定的 **C3** 过程只是使用现有的算法。不同的是该过程是在新的映射质量空间执行的，而不是原始空间。 这个过程在 **算法 3** 中给出。异常检测的任务特定 **C3** 过程显示在 **算法 4** 中的 2-5 步。异常检测仅需要 **C1** 和 **C3**; 而另外两个任务需要全部三个步骤。

![算法 3](/assets/blog-images/Anomaly-Detection-2.8.png)
**算法 3**

![算法 4](/assets/blog-images/Anomaly-Detection-2.9.png)
**算法 4**

# Reference

[^iForest1]: F. T. Liu, K. M. Ting, and Z. H. Zhou, "[Isolated Forest,](https://doi.org/10.1109/ICDM.2008.17)" in *2008 Eighth IEEE International Conference on Data Mining,* **2008,** pp. 413-422.

[^iForest2]: F. T. Liu, K. M. Ting, and Z.-H. Zhou, "[Isolation-Based Anomaly Detection,](https://doi.org/10.1145/2133360.2133363)" *ACM Trans. Knowl. Discov. Data,* vol. 6, pp. 1-39, **2012.**

[^Mass1]: K. M. Ting, G.-T. Zhou, F. T. Liu, and J. S. C. Tan, "[Mass estimation and its applications,](https://doi.org/10.1145/1835804.1835929)" in *Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and data mining,* Washington, DC, USA, **2010,** pp. 989-998.

[^Mass2]: K. M. Ting, G.-T. Zhou, F. T. Liu, and S. C. Tan, "[Mass estimation,](https://link.springer.com/article/10.1007/s10994-012-5303-x)" *Machine Learning,* vol. 90, pp. 127-160, **2013.**

[^RS-Trees]: S. C. Tan, K. M. Ting, and T. F. Liu, "[Fast anomaly detection for streaming data,](https://doi.org/10.5591/978-1-57735-516-8/IJCAI11-254)" in *Proceedings of the Twenty-Second international joint conference on Artificial Intelligence* - Volume 2, Barcelona, Catalonia, Spain, **2011,** pp. 1511-1516.




