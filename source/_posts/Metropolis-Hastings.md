---
title: Metropolis-Hastings
date: 2023-07-15 10:00:00
tags: Markov Chains
categories: Stochastic Process
mathjax: true
---

# Introduction

The first MCMC algorithm is the Metropolis algorithm, published by Metropolis et al. (1953). It was generalized by Hastings (1970) and Peskun (1973, 1981) towards statistical applications. After a long time, it was rediscovered by Geman and Geman (1984), Tanner and Wong
(1987), and Gelfand and Smith (1990).

Assume the probability density $\pi$ is the target distribution and $q(\cdot | \cdot)$ is the proposal transition distribution. From an initial state $X_0$, the Metropolis-Hastings algorithm aims to generate a Markov chain $\{X_1, X_2, ...\}$, such that $X_t$ converges to distribution $\pi$.



# Metropolis-Hastings algorithm

**Input:** initial value $X_0$, transition $q(\cdot | \cdot)$, number of iterations $T$.
 
**Output:** Markov chain $\{X_1, X_2, ..., X_T\}$.

For $t = 1, 2, ..., T$:

1. Generate $u$ from uniform distribution $U(0, I)$.
2. Generate $X$ from $q(X | X_{t-1})$.
3. Compute the acceptance probability $A(X, X_{t-1}) = \min\{1, \frac{\pi(X)q(X_{t-1} | X)}{\pi(X_{t-1})q(X | X_{t-1})}\}$.
4. if $u \leq A(X, X_{t-1})$, then $X_t = X$; else $X_t = X_{t-1}$.

Now, we prove that $\pi$ is one of the stationary distribution of the generated Markov chain.

Proof: Recall the detailed balance equation, for any $i, j \in \Omega$, we have
$$
\pi_i P_{i,j} = \pi_j P_{j,i},
$$
then $\pi$ is a stationary distribution of the Markov chain.

For the Metropolis-Hastings algorithm, we have the transition probability
$$
K(X_t | X_{t-1}) = q(X_t| X_{t-1}) A(X_t, X_{t-1}) + \delta(X_t = X_{t-1}) (1 - \sum_{X \in \Omega} q(X|X_{t-1}) A(X, X_{t-1})).
$$

Now we need to prove 
$$
\pi(X_{t-1})K(X_t | X_{t-1}) = \pi(X_t)K(X_{t-1} | X_t).
$$

If $X_t = X_{t-1}$, then the equation holds. If $X_t \neq X_{t-1}$, then
$$
\pi(X_{t-1})K(X_t | X_{t-1}) = \pi(X_{t-1})q(X_t| X_{t-1}) A(X_t, X_{t-1}) = \min\{\pi(X_{t-1})q(X_t| X_{t-1}), \pi(X_t)q(X_{t-1} | X_t)\}.
$$
and
$$
\pi(X_{t})K(X_{t-1} | X_{t}) = \pi(X_{t})q(X_{t-1}| X_{t}) A(X_{t-1}, X_{t}) = \min\{\pi(X_{t})q(X_{t-1}| X_{t}), \pi(X_{t-1})q(X_{t} | X_{t-1})\}.
$$
So, $\pi(X_{t-1})K(X_t | X_{t-1}) = \pi(X_{t})K(X_{t-1} | X_{t})$. $\pi$ is a stationary distribution of the Markov chain.

***Remark 1.*** The detailed balance condition is a sufficient but not necessary condition for $\pi$ to be a stationary distribution of the Markov chain. If we want to prove $\pi$ is the unique stationary distribution of the Markov chain, we need to prove the Markov chain is **irreducible and positive recurrent**. 

***Remark 2.*** The first initial state $X_0$ is randomly generated and usually removed from the sample as burn-in or warm-up. 

Q: **Since it is recurrent, it must return to the initial values. Will this initial be rejected with a high probability?**

***Remark 3.*** In practice, the performances of the algorithm are obviously highly dependent on the choice of the transition $q(\cdot | \cdot)$, since some choices see the chain unable to converge in a manageable time.

***Remark 4.*** We need to able to evaluate a function $p(x) \propto \pi(x)$. Since we only need to compute the ratio $\pi(y)/\pi(x)$, the proportionality constant is irrelevant. Similarly, we only care about $q(\cdot | \cdot)$ up to a constant

# Reference
- C.P. Robert. (2016). The Metropolis-Hastings algorithm.

