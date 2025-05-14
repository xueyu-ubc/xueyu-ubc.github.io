---
title: Kolmogorov-Smirnov statistic
date: 2023-08-27 19:00:00
tags: Statistic
categories: Statistic
mathjax: true
---

# Kolmogorov-Smirnov statistic

Consider any distribution $D$ on $\mathbb{R}$, and CDF $F(t) = \mathbb{P}_{x \sim D}(x \leq t)$.

Let $X = (x_j)_{i \in [n]}$ be $n$ samples drawn from $D$. 

Def: The empirical CDF (eCDF) of $X$ is defined as $\hat{F}_n(t) = \frac{1}{n} \sum_{j=1}^n \mathbb{1}_{x_j \leq t}$.

Def: The Kolmogorov-Smirnov statistic is defined as $D_n = \sup_{t \in \mathbb{R}} \{|\hat{F}_n(t) - F(t)|\}$.

Theorem [DKW, 1956]: $\mathbb{P}(D_n > \epsilon) \leq c e^{-2n\epsilon^2}$, where $c$ is a constant.

Theorem [Massart, 1990]: $\mathbb{P}(D_n > \epsilon) \leq 2 e^{-2n\epsilon^2}$.

Theorem [Harvey, 2020]: $\mathbb{P}(D_n > \epsilon) \leq \frac{4}{\epsilon} e^{-\frac{1}{2}n\epsilon^2}$.



