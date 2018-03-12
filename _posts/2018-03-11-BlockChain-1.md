---
layout: post
title: 区块链技术：A Quick Review
category: Machine Learning
author: Hongtao Yu
tags: 
  - blockchain
  - innovation
comments: true
use_math: true
lang: zh
---

- TOC
{:toc}

# 基本概念


**区块链技术**：`区块链`（BlockChain）是一个在`对等式网络`（peer-to-peer network）中共享，并在其中达成共识的`分布式数据库`（distributed database）。 它由一个链接的`区块`(block)序列组成，区块中记录着通过`公钥`（public-key）密码体系加密并由网络社区验证的带有时间戳的交易信息。一旦一个元素被添加到区块链中，它就不能再被篡改。因此区块链是一个记录历史活动的不变记录。[^Seebacher2017]

![区块链示例](/assets/blog-images/Block-Chain-1.1.png)

**交易（Transaction）：** 一次操作，导致账本状态的一次改变，如添加一条记录。

**区块（Block）：** 是区块链中的一条记录，记录一段时间内发生的交易和状态结果，是对当前账本状态的一次共识。

**链（Chain）：** 由一个个区块按照发生顺序串联而成，是整个状态变化的日志记录。

**挖矿（Mining）：** 指通过计算形成新的区块，是交易的支持者利用自身的计算机硬件为网络做数学计算进行交易确认和提高安全性的过程。以比特币为例：交易支持者（矿工）在电脑上运行比特币软件不断计算软件提供的复杂的密码学问题来保证交易的进行。作为对他们服务的奖励，矿工可以得到他们所确认的交易中包含的手续费，以及新创建的比特币。

**对等式网络：** 是指通过允许单个节点与其他节点直接交互，从而实现整个系统像有组织的集体一样运作的系统。以比特币为例：网络以这样一种方式构建——每个用户都在传播其他用户的交易。而且重要的是，不需要银行或其他金融机构作为第三方。

**哈希散列(Hash)：** 是密码学里的经典技术，把任意长度的输入通过哈西算法，变换成固定长度的由字母和数字组成的输出。

**数字签名（Digital Signature）：** 是一个让人可以证明所有权的数学机制。

**私钥（Private Key）：** 是一个证明你有权从一个特定的钱包消费电子货币的保密数据块，是通过数字签名来实现的 。

**双重消费（Double Spent）：** 指用户试图非法将电子货币同时支付给两个不同的收款人，是电子货币的最大风险之一。

# 分类

根据参与者的不同，可以分为`公开（Public）链`、`联盟（Consortium）链`和`私有（Private）链`。

**公开链：** 任何人都可以参与使用和维护，典型的如比特币区块链，信息是完全公开的。

如果引入许可机制，包括私有链和联盟链两种。

**私有链：** 则是集中管理者进行限制，只能得到内部少数人可以使用，信息不公开。

**联盟链：** 介于两者之间，由若干组织一起合作维护一条区块链，该区块链的使用必须是有权限的管理，相关信息会得到保护，典型如银联组织。

目前来看，公开链将会更多的吸引社区和媒体的眼球，但更多的商业价值应该在联盟链和私有链上。


# 智能合约

**智能合约：** 一个`智能合约` 是一套以数字形式定义的承诺，包括合约参与方可以在上面执行这些成活的协议。通俗地讲，智能合约时一个在计算机系统上，当一定条件被满足的情况下，可以被自动执行的合约。

<u>阻碍计算机广泛应用到智能合约场景的最大障碍是信用问题</u>。在现实世界中合约时记录在纸上，签印之后生效；计算机世界里，合约是记录在代码里，数字化的合约具有被篡改的风险，或者被黑客攻击的技术风险。如果合约只保留在服务器端，那么客户端对服务器的信任是一个很大的问题，因为如果合约在服务器上被修改，举证是一件很困难的事情，因为证据都保留在对方的计算机系统中。区块链具有去中心化，不可篡改，高可靠性的特点。不可篡改性保证用户不用担心合约的内容被更改，高可靠性保证我们不需要担心在条件满足时合约不被执行；去中心化带来了全网备份，完备的记录可以支持事后审计。


# 区块链的应用

## 云联盟[^Margheri2017]

云系统的不断增加引发了新的问题，特别是已经部署的云服务之间的互联性和合作性。例如，如何共享计算资源，控制第三方服务或者数据的使用，属于不同管理域的实体间如何协作，等。`云聚合`（cloud aggregation）可能是可能是实现这个目标的一个解决方案。

实现云聚合的一个重要手段是`云联盟`(cloud federation)。一个最近的概念是允许来自不同云提供商的服务在一个池中聚合。然而目前并没有一个被广泛接受的提案，少数可用的方案缺乏适当的监管策略：所有云联盟成员都应该是对等的网络，同等地参与监管。

### FaaS

欧洲的 [SUNFISH](http://www.sunfishproject.eu/)项目一直在致力于设计并执行一个称为 `FaaS` (Federation-as-a-Service) 的新型云联盟解决方案。

该方案基于对数据的分布式控制：

- 所有成员都拥有一致的数据副本，不能以任何方式损坏数据。

- 监管行为的民主控制：联盟按照共识进行裁定，以确保每个成员的权利不因其他成员的共谋而受到侵犯。

- 值得信赖的数据服务：访问和共享数据服务（如访问控制和数据匿名化）必须得到保护，以避免机密性和完整性的攻击。

为了实施这种治理，可以利用区块链技术作为基于 FaaS 的联盟注册基础设施。更具体地，利用智能合约，在区块链上自主部署和执行程序。

## Meta 产品的去中心化存储[^Shrestha2016]

有很多公司生产 Meta 产品。，并且都拥有自己的物联网（IoT）平台。这些平台仅适用于他们自己的特定产品。 用户信任这些产品服务提供商可以安全地保存他们的数据，并且不会破坏、滥用或出售这些数据给第三方。

我们或许可以设计一些基于云存储的平台，以便 Meta 产品数据可以跨提供商和用户共享。但是我们需要使用去中心化的存储策略来保证数据的隐私性和安全性，以及对参与节点使用激励措施。

云计算提供了服务的三种服务模式： SaaS，PaaS，和 IaaS。借助 SaaS 模型，服务提供商可为用户提供服务。开发人员使用PaaS 和 IaaS 开发他们的应用程序。例如他们可以将他们的数据存储在云服务器中。这里，基于 PaaS 模型作者提出了具有分布式存储的集成平台服务。

在集中式云中，可以将用户的数据发送给服务提供商或公司，并让在稍后的时间点将数据回收。 可是一旦数据发送出去，就脱离了我们的控制权。如果公司决定保留或者销毁数据，我们就不能保护我们的数据安全。集中式系统的另一个缺点是数据使用公司可能会在没有我们允许的情况下将数据分享给第三方。 即使数据被加密，然而如果客户的服务器崩溃或者其他的原因，数据可能被破坏。 

使用区块链的去中心化方法，可以让我们能够涵盖许多法律问题。系统中所有内容都经过加密和散列验证。使用智能合同，将客户激励和公司盈利与保护上传数据挂钩。 不同节点可以提供更便宜的服务以达到更高的盈利，因此将会有更加广阔的市场。


智能合约用来管理对保存用户数据的主机的奖励机制。
主机要求用户支付一些钱来补偿自己的费用，但他们也必须支付一些金额作为抵押以获得保持数据的资格。 当用户的数据被存储时，合同被放置进入区块链。 经过一段时间（几周或几个月）后，区块会触发目标主机展示它们的工作量证明，以表明他们仍然持有数据。 如果他们出示了工作量证明，那么他们会像比特币中的矿工一样获得奖励；否则他们将会失去他们抵押款和继续合作盈利的机会。

## 基于区块链加密的 SDN[^Basnet2017]

通过区块链，应用程序可以以分散的方式运行，中央监管或中介机构不需要监督各方之间的交易。即使在网络中存在无信任环境的情况下，也可以进行安全交易。此外，因为在每个网络互动中不需要中介，可以使得各方之间能够快速和解。

作者建议的SDN环境中选择 Pyethereum 测试器进行区块链加密解密。在云服务中可以使用 OpenStack构建数据中心。因为 OpenStack 开源，并且允许用户可以根据需要轻松管理资源。

本文中，作者设计了一种方法，结合 OpenStack，在 SDN 环境中使用区块链技术加密来构建数据中心。

本论文提供区块链，SDN 和 OpenStack 的详细描述，并提出了区块链，SDN 和 OpenStack 一起使用的方式。区块链允许我们拥有一个分布式的对等网络，在这个网络中，不信任的成员可以以可验证的方式在没有可信中介的情况下彼此交互。本文讨论了这种机制的工作原理，并且研究了智能合约脚本，描述了区块链和SDN组合如何以安全和保密的方式与作为云数据存储的 OpenStack 相关联，便于在主机间共享服务和资源（文件）。

![BSS 区块示意图](/assets/blog-images/Block-Chain-1.2.png)

## 使用区块链管理跨云敏感数据

区块链是一项创新技术，除了数据完整性之外，还能确保数据及其计算的完全去中心化控制。 在这种分散的基础设施之下，一种称为智能合约的不可变程序的执行，对参与方的不可否认性提供了保证。 除了比特币和以太坊之外，还有大量的区块链针对私有设置的系统（例如跨云集成）已经出现在市场上。[Hyperledger Fabric](www.hyperledger.org/projects/fabric) 提供了对数据可见性的控制以及智能合约执行场所的解决方案。

意大利经济和金融部（MEF）目前正面临着克服云间公共机构数据的隔离来计算警察工资的问题。具体而言，意大利的法律框架迫使内政部（MIN）成为该组织的成员警察敏感数据的独家控制者。但是，MEF需要访问这些数据以正确计算工资单。为了解决这个问题，MEF 已经与 MIN 进行了错综复杂的合作，MIN 在本地执行了一部分工资税计算，然后由 MEF 使用。但是，这导致了不受控合作中容易出现的错误和恶意破坏，例如避税或
巨额工资的突然上涨。这种欺诈行为很难发现，而且最重要的是，MEF 必须对此负责，尽管它无法控制整个薪资数据。

# References

[^Seebacher2017]: Seebacher, S. & Schüritz, R. [Blockchain Technology as an Enabler of Service Systems: A Structured Literature Review](https://link.springer.com/chapter/10.1007/978-3-319-56925-3_2), *International Conference on Exploring Services Science,* **2017,** 12-23

[^Johansen2017]: Johansen, S. K. [A Comprehensive Literature Review on the BlockChain as a Technological Enabler for Innovation](https://www.researchgate.net/publication/312592741_A_Comprehensive_Literature_Review_on_the_Blockchain_Technology_as_an_Technological_Enabler_for_Innovation), Mannheim University, **2017**

[^Margheri2017]: A. Margheri, M. S. Ferdous, M. Yang, and V. Sassone, ["A Distributed Infrastructure for Democratic Cloud Federations,"](https://doi.org/10.1109/CLOUD.2017.93) in *2017 IEEE 10th International Conference on Cloud Computing (CLOUD)*, **2017,** pp. 688-691.

[^Shrestha2016]: A. K. Shrestha and J. Vassileva, ["Towards decentralized data storage in general cloud platform for meta-products,"](https://dl.acm.org/citation.cfm?id=3016029) in *Proceedings of the International Conference on Big Data and Advanced Wireless Technologies*, Blagoevgrad, Bulgaria, **2016,** pp. 1-7.

[^Basnet2017]: S. R. Basnet and S. Shakya, ["BSS: Blockchain security over software defined network,"](http://ieeexplore.ieee.org/document/8229910/) in *2017 International Conference on Computing, Communication and Automation (ICCCA),* **2017,** pp. 720-725.


[^Nicoletti2017]: L. Nicoletti, F. Lombardi, A. Margheri, V. Sassone, and F. P. Schiavo, ["Cross-cloud management of sensitive data via Blockchain: a payslip calculation use case,"](https://eprints.soton.ac.uk/415084/2/Cross_Cloud_blockchain.pdf) **2017.**

