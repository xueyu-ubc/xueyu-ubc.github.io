---
title: ADMMdecentralized
date: 2021-02-03 20:49:48
tags: FL
categories: Distributed Learning
mathjax: true
---

# Abstract
考虑点对点的协作网络。本文解决的问题是：每个节点如何与具有相似目标的其他节点进行通信来改善本地模型？作者介绍了两种完全去中心化的算法，一种是受标签传播的启发，旨在平滑预先训练好的局部模型；第二种方法，节点基于本地数据核相邻节点进行迭代更新来共同学习和传播。 

# Introduction  
数据不断产生，当前从数据中提取信息的主要方式是收集所有用户的个人数据于一个服务器上，然后进行数据挖掘。但是，中心化的方式存在一些问题，比如说一些用户拒绝提供个人数据，带宽和设备花费问题。即使一些算法允许数据分布在用户设备上，通常需要中心端来进行聚合和协调。

在本文中，作者考虑完全去中心化的点对点网络。不同于那些求解全局模型的算法，本文关注于每个节点可以根据自身目标函数学习一个个性化模型。作者假设网络结构已知，该网络结构能够反映出不同节点的相似度（如果两个节点具有相似的目标函数，那么这两个节点在网络中是邻居），每个节点只知道与其直接相邻的节点。一个节点不仅可以根据自身数据学习模型，还可以结合它的邻居。假设每个节点只知道相邻节点的信息，不知道整个网络结构。

作者提出两个算法。第一个是 model propagation：首先，每个节点先基于自己的局部数据学习到模型参数，然后，结合整个网络结构，平滑这些参数。第二个是 collaborative learning，这个算法更加灵活，它通过优化一个模型参数正则化（平滑）和局部模型准确性上的折中问题。作者基于分布式的ADMM算法提出一个异步gossip算法。  

# Preliminaries

## Notations and Problem Setting

考虑 $n$ 个节点 $V = [n] = \{1,...,n\}$。凸的损失函数 $l: \mathbb{R}^{p} \times \mathcal{X} \times \mathcal{Y}$，节点 $i$ 的目标是学习模型参数 $\theta_{i} \in \mathbb{R}^{p}$，使得关于未知分布 $\mu_{i}$ 的期望损失 $E_{(x_{i}, y_{i})\sim \mu_{i}}l(\theta_{i}; x_{i}, y_{i})$ 很小。节点 $i$ 具有 $m_{i}$ 个来自分布 $\mu_{i}$ 的 i.i.d 的训练样本 $S_{i} = \{(x_{i}^{j}, y_{i}^{j})\}_{j=1}^{m_{i}}$。允许不同节点的样本量相差很大。每个节点可以最小化局部损失函数得到 $\theta_{i}^{sol}$:

$$
\theta_{i}^{sol} \in \argmin_{\theta \in \mathbb{R}^{p}} L_{i}(\theta) = \sum_{j=1}^{m_{i}} l(\theta;x_{i}^{j}, y_{i}^{j}).
$$

我们目标是通过结合其他节点信息，进一步改善上述模型。考虑一个加权网络结构 $G = (V, E)$，具有 $V$ 个节点，$E \subseteq V \times V$ 为无向边。定义 $W \in \mathbb{R}^{n \times n}$ 为由 $G$ 得到的对称非负加权矩阵，如果 $(i,j) \ne E$ or $i = j$， $W_{ij} = 0$。本文假设权重矩阵已知。定义对角阵 $D\in \mathbb{R}^{n \times n}$，$D_{ii} = \sum_{j=1}^{n} W_{ij}$。节点 $i$ 的邻域 ：$\mathcal{N}_{i} = \{j \ne i: W_{ij} > 0\}$。  


# Model Propagation

假设每个节点通过最小化局部损失函数得到各自的模型 $\theta_{i}^{sol}$。由于每个节点上的模型都是在不同大小数据集上考虑得到，作者使用 $c_{i} \in (0,1]$ 定义每个节点模型的可信度。 $c_{i}$ 的值应该和节点 $i$ 的样本量大小呈正相关，可以设置为 $c_{i} = \frac{m_{i}}{\max_{j} m_{j}}$。如果 $m_{i}=0$，可以设置为一个小量。

定义 $\Theta = [\theta_{1}; \theta_{2};...;\theta_{n}] \in \mathbb{R}^{n \times p}$，我们要优化的目标函数为：

![ ](https://cdn.jsdelivr.net/gh/sugar-xue/BlogImages@main/Picbed1612018569.jpg)

第一项二次函数用来平滑相邻节点的参数，当两个节点间权重越大时，节点间参数越相近；第二项的目的是使具有较高置信度的模型的参数不要太远离各自模型上的参数。具有较低置信度的模型的参数被允许具有较大的偏差，容易被相邻节点影响。$D_{ii}$ 的目的是为了normalization。

![ ](https://cdn.jsdelivr.net/gh/sugar-xue/BlogImages@main/Picbed1612018994(1).png)  
![ ](https://cdn.jsdelivr.net/gh/sugar-xue/BlogImages@main/Picbed1612019022(1).png)

计算 (4) 需要知道整个网络的信息以及所有节点的独立模型信息，这对于节点而言是未知的，因为每个节点只知道相邻节点的信息。因此，作者提出下面的迭代形式：  

![ ](https://cdn.jsdelivr.net/gh/sugar-xue/BlogImages@main/Picbed1612019342(1).png)

作者证明，无论初始值 $\Theta(0)$ 取何值，上述迭代序列收敛到 (4)。(5) 式可以进一步分解为    

![ ](https://cdn.jsdelivr.net/gh/sugar-xue/BlogImages@main/Picbed1612019480(1).png)

考虑一个同步计算：在每一步，每个节点都和其所有相邻节点进行通信，收集它们当前参数，然后使用它们的参数更新上式。同步更新会导致很大的延迟，因为任何节点都必须等剩余节点更新完后才能进行下一步更新。并且，每一步，所有节点都需要和其邻居节点进行通信，降低了算法的效率。所以作者提出一个异步算法。

## Asynchronous Gossip Algorithm

在异步设置中，每个节点都有一个局部clock ticking at times of rate 1 Poisson process. 由于节点都是独立同分布的，所以相当于在每一步时等概率激活每个节点。

在时间 $t$ 时，每个节点都存有相邻节点的信息。以数学形式表示，考虑矩阵 $\tilde{\Theta}_{i}(t) \in \mathbb{R}^{n \times p}$，第 $i$ 行 $\tilde{\Theta}_{i}^{i}(t) \in \mathbb{R}^{p}$ 为节点 $i$ 在时刻 $t$ 的模型参数，$\tilde{\Theta}_{i}^{j}(t) \in \mathbb{R}^{p} (j \ne i)$ 为节点 $i$ 储存的关于邻居节点 $j$ 的last knowledge. 对于 $j \notin \mathcal{N}_{i} \bigcup \{i\}$，$\forall t > 0$，$\tilde{\Theta}_{i}^{j}(t) = 0$。令$\tilde{\Theta} = [\tilde{\Theta}_{1}^{T}, ...,\tilde{\Theta}_{n}^{T}] \in \mathbb{R}^{n^{2} \times p}$。

如果在时间 $t$ 时，节点 $i$ wakes up，执行如下步骤：

- communication: 节点 $i$ 随机选择一个邻居节点 $j \in \mathcal{N}_{i}$，(先验概率 $\pi_{i}^{j}$)，节点 $i$ 和节点 $j$ 同时更新它们的参数：
$$
\tilde{\Theta}_{i}^{j}(t+1) = \tilde{\Theta}_{j}^{j}(t) \qquad \tilde{\Theta}_{j}^{i}(t+1) = \tilde{\Theta}_{i}^{i}(t),
$$

- update: 基于当前信息，节点 $i$ 和节点 $j$ 更新自己的模型参数：
  $$
  \tilde{\Theta}_{l}^{l}(t+1) = (\alpha +\bar{\alpha}c_{l})^{-1}(\alpha \sum_{k \in \mathcal{N}_{l}} \frac{W_{lk}}{D_{ll}}\tilde{\Theta}_{l}^{k}(t+1) + \bar{\alpha}c_{l}\theta^{sol}_{l}) \quad(l \in \{i,j\}).
  $$

  网络中的其他变量保持不变。作者提出的算法属于 gossip algorithms，每个节点每次最多只和一个邻居节点通信。

作者证明，上述算法可以收敛到使每个节点具有最优参数。

  ![ ](https://cdn.jsdelivr.net/gh/sugar-xue/BlogImages@main/Picbed1612340701.jpg)

# Collaborative Learning

上述算法先在局部节点上进行学习，然后在进行网络通信。在这部分，作者提出了一个使节点可以同时进行基于局部数据和邻居节点信息更新模型参数的算法。相较于前面的算法，该算法通信成本较高，但是估计精度高于前者。

## Problem Formulation 

优化目标：  

![ ](https://cdn.jsdelivr.net/gh/sugar-xue/BlogImages@main/Picbed1612341031(1).png)

注意到，这里的置信度通过 $\mathcal{L}_{i}$ 体现，因为 $\mathcal{L}_{i}$ 为局部节点 $i$ 上所有观测的损失函数和。

一般情况下，上述问题没有解析解，作者提出一个分散式迭代算法进行求解。

## Asynchronous Gossip Algorithm

作者基于ADMM提出了一个异步分散式算法。本文的目的不是寻找一个consensus 解，因为我们的目标是为了学习到每个节点的personalized model. 作者通过将问题 (7) 进行变换为一个部分consensus问题，使用ADMMD进行求解。

令 $\Theta_{i} \in \mathbb{R}^{(|\mathcal{N_{i}}|+1)\times p}$ 为变量 $\theta_{j} \in \mathbb{R}^{p}(j \in \mathcal{N_{i}} \bigcup \{i\})$ 的集合。定义 $\theta_{j}$ 为 $\Theta_{i}^{j}$。优化问题(7)重新写为：

  ![ ](https://cdn.jsdelivr.net/gh/sugar-xue/BlogImages@main/Picbed1612341938(1).png)

在这个目标函数中，所有的节点相互依赖，因为它们共享一个优化变量 $\in \Theta$。为了使用ADMM，需要将各个节点的优化变量独立，对于每个节点 $i$，定义一个local copy $\tilde{\Theta}_{i} \in \mathbb{R}^{(|\mathcal{N_{i}}|+1)\times p}$，添加等式约束：$\tilde{\Theta}_{i}^{i} = \tilde{\Theta}_{j}^{i}$，对于所有的 $i \in [n], j \in \mathcal{N}_{i}$。

  ![ ](https://cdn.jsdelivr.net/gh/sugar-xue/BlogImages@main/Picbed1612342502(1).png)

增广拉格朗日乘子：

  ![ ](https://cdn.jsdelivr.net/gh/sugar-xue/BlogImages@main/Picbed1612342580(1).png)

算法如下，假设时刻 $t$ 时节点 $i$ wakes up，选取邻居节点 $j \in \mathcal{N}_{i}$，定义 $e = (i,j)$，

  ![ ](https://cdn.jsdelivr.net/gh/sugar-xue/BlogImages@main/Picbed1612342728(1).png)

  ![ ](https://cdn.jsdelivr.net/gh/sugar-xue/BlogImages@main/Picbed1612342765(1).png)

# Experiments

## Collaborative Linear Classification

考虑100个节点，每个节点的目标是建立一个线性分类模型 in $\mathbb{R}^{p}$。为了方便可视化，每个节点的真实参数位于2维子空间：将其参数看作是 $\mathbb{R}^{p}$ 空间中的向量，前两项从正态分布中随机产生，剩余项为0。两个节点 $i$ 和节点 $j$ 的相似度通过参数距离的高斯核定义，定义 $\phi_{ij}$ 为两个真实参数在单位圆上投影的夹角，$W_{ij} = \exp(\cos \phi_{ij} - 1)/\sigma$，$\sigma = 0.1$。权重为负值的将被忽略。每个节点具有随机的训练样本，样本的标签为二元标签，由线性分类模型产生。以概率0.05随机使标签反转，以产生噪音数据。每个节点的损失函数为hinge损失：$l(\theta;(x_{i}, y_{i})) = \max(0, 1-y_{i}\theta^{T}x_{i})$。作者评估了模型在100个测试样本上的预测精度。

  ![ ](https://cdn.jsdelivr.net/gh/sugar-xue/BlogImages@main/Picbed1612343843(1).png)

# 参考文献

- Vanhaesebrouck, P., Bellet, A. & Tommasi, M.. (2017). Decentralized Collaborative Learning of Personalized Models over Networks. Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, in PMLR 54:509-517