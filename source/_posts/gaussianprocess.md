---
title: Gaussian processes
date: 2023-08-06 16:00:00
tags: SDE
categories: Stochastic Process
mathjax: true
---

# Jointly Gaussian random variables

Definition: Random variables (RV) $X_1, ..., X_n$ are jointly Gaussian if any linear combination of them is Gaussian.

RV $X = [X_1, ..., X_n]^{T}$ is Gaussian $\leftrightarrows$
Given any scalars $a_1,... a_n$, the RV 
$Y = a_1 X_1 + a_2 X_2 + ... + a_n X_n$ is Gaussian distributed.

## Pdf of jointly Gaussian RVs in n dimensions

Let $X \in \mathbb{R}^n$, $\mu = \mathbb{E}[X]$, 

covariance matrix 
$$C:= \mathbb{E}[(X - \mu)(X - \mu)^T] =
\begin{pmatrix}
    \sigma_{11}^2 & \sigma_{12}^2 & \cdots & \sigma_{1n}^2 \\
    \sigma_{21}^2 & \sigma_{22}^2 & \cdots & \sigma_{2n}^2 \\
    \vdots & \vdots & \ddots & \vdots \\
    \sigma_{n1}^2 & \sigma_{n2}^2 & \cdots & \sigma_{nn}^2
\end{pmatrix}
$$
Then, the pdf of RV $X$ can be defined as
$$
p(x) = \frac{1}{(2\pi)^{n/2} \text{det}^{1/2}(C)} \exp(-\frac{1}{2}(x - \mu)^T C^{-1} (x - \mu)).
$$

- $C$ is invertible
- We can verify all linear combinations is Gaussian.
- To fully specify the probability distribution of a Gaussian vector $X$, the mean vector $\mu$ and covariance matrix $C$ suffice.
  
# Gaussian processes

Gaussian processes (GP) generalize Gaussian vectors to **infinite** dimensions.

Definition. $X(t)$ is a GP if any linear combination of values $X(t)$ is Gaussian.
That is, for arbitrary $n > 0$, times $t_1, ..., t_n$ and constants $a_1, ..., a_n$, $Y = a_1 X(t_1) + a_2 X(t_2) + ... + a_n X(t_n)$ is Gaussian distributed.

- Time index $t$ can be continuous or discrete.

- Any linear functional of $X(t)$ is Gaussian distributed.
For example, the integral $Y = \int_{t_1}^{t_2} X(t) \text{d}t$ is Gaussian distributed.


## Jointly pdf in a Gaussian process

Consider times $t_1,..., t_n$, the mean value
$\mu(t_i)$ is $$\mu(t_i) = \mathbb{E}[X(t_i)].$$

The covariance between values at time $t_i$ and $t_j$ is $C(t_i, t_j) := \mathbb{E}[(X(t_i) - \mu(t_i))(X(t_j) - \mu(t_j))^T]$. 

The covariance matrix for values $X(t_1),..., X(t_n)$ is 
$$
C(t_1,..., t_n) = 
\begin{pmatrix}
    C_{t_1, t_1} & C_{t_1, t_2} & \cdots & C_{t_1, t_n}\\
    C_{t_2, t_1} & C_{t_2, t_2} & \cdots & C_{t_2, t_n} \\
    \vdots & \vdots & \ddots & \vdots \\
    C_{t_n, t_1} & C_{t_n, t_2} & \cdots & C_{t_n, t_n}
\end{pmatrix}
$$.

The jointly pdf of $X(t_1),..., X(t_n)$ is $N([\mu(t_1), ..., \mu(t_n)]^T, C(t_1,..., t_n))$.

## Mean value and autocorrelation functions

To specify a Gaussian process, we only need to specify:

- Mean value function: $\mu(t) = \mathbb{E}[x(t)]$.
  
- Autocorrelation function (symmetric): $R(t_1, t_2) = \mathbb{E}[X(t_1) X(t_2)]$.
  
The autocovariance $C(t_1, t_2) = R(t_1, t_2) - \mu(t_1) \mu (t_2)$.

More general, we consider GP with $\mu(t) = 0$. [define new process $Y(t) = X(t) - \mu(t)$]. In this case, $C(t_1, t_2) = R(t_1, t_2)$.

All probs. in a GP can be expressed in terms of $\mu(t)$ and $R(t, t)$.
$$
p(x_t) = \frac{1}{\sqrt{2\pi (R(t,t) - \mu^2(t))}} \exp(- \frac{(x_t - \mu(t))^2}{2(R(t,t) - \mu^2(t))}).
$$

## Conditional probabilities in a GP

Consider a zero-mean GP $X(t)$, two times $t_1$ and $t_2$. The covariance matrix is 
$$
C = \begin{pmatrix}
  R(t_1, t_1) & R(t_1, t_2) \\
  R(t_1, t_2) & R(t_2, t_2)
\end{pmatrix}
$$

The jointly pdf of $X(t_1)$ and $X(t_2)$ is 
$$
p(x_{t_1}, x_{t_2}) = \frac{1}{2\pi \text{det}^{1/2} C} \exp(-\frac{1}{2}[x_{t_1}, x_{t_2}]^T C^{-1} [x_{t_1}, x_{t_2}])
$$

The conditional pdf of $X(t_1)$ given $X(t_2)$ is
$$
p_{X(t_1)| X(t_2)}(x_{t_1} | x_{t_2}) = \frac{p(x_{t_1}, x_{t_2})}{p(x_{t_2})}. \qquad (1)
$$

# Brownian motion process (a.k.a Wiener process)

Definition. A Brownian motion process (a.k.a Wiener process) satisfies

(1) $X(t)$ is normally distributed with zero mean and variance $\sigma^2 t$, $$X(t) \sim N(0, \sigma^2 t).$$

(2) Independent increments. For all times $0 < t_1 < t_2 < \cdots < t_n$, the random variables $X(t_1), X(t_2) - X(t_1), ..., X(t_n) - X(t_{n-1})$ are independent.

(3) Stationary increments. Probability distribution of increment $X(t+s) - X(s)$ is the same as probability distribution of $X(t)$. 
$$[X(t+s) - X(s)]  \sim N(0, \sigma^2 t).$$

- Brownian motion is a Markov process.
- Brownian motion is a Gaussian process.

## Mean and autocorrelation of Brownian motion

1, Mean funtion $\mu(t) = \mathbb{E}[X(t)] = 0$.

2, Autocorrelation of Brownian motion $R(t_1, t_2) = \sigma^2 \min \{t_1,t_2\}$.

Proof. Assume $t_1 < t_2$, then autocorrelation $R(t_1, t_2) = \mathbb{E}[X(t_1)X(t_2)] = \sigma^2 t_1$.

If $t_1 < t_2$, according to conditional expectations, we have
$$
\begin{aligned}
  R(t_1, t_2) = \mathbb{E}[X(t_1)X(t_2)] &= \mathbb{E}_{X(t_1)}[\mathbb{E}_{X(t_2)}[X(t_1)X(t_2) | X(t_1)]] \\
  &= \mathbb{E}_{X(t_1)}[X(t_1)\mathbb{E}_{X(t_2)}[X(t_2) | X(t_1)]]
\end{aligned}
$$
According to equation (1), the condition distribution of $X(t_2)$ given $X(t_1)$ is 
$$
[X(t_2) | X(t_1)] \sim N(X(t_1), \sigma^2 (t_2 - t_1)),
$$
thus, $\mathbb{E}_{X(t_2)}[X(t_2) | X(t_1)] = X(t_1)$. 

$$
\begin{aligned}
  R(t_1, t_2) = \mathbb{E}[X(t_1)X(t_2)] &= \mathbb{E}_{X(t_1)}[\mathbb{E}_{X(t_2)}[X(t_1)X(t_2) | X(t_1)]] \\
  &= \mathbb{E}_{X(t_1)}[X(t_1) X(t_1)] \\
  &= \mathbb{E}_{X(t_1)}[X^2(t_1)] = \sigma^2 t_1. 
\end{aligned}
$$

Similarly, if $t_2 < t_1$, $R(t_1, t_2) = \sigma^2 t_2$.

## Brownian motion with drift (BMD)
For Brownian motion, it is an unbiased random walk. Walker steps right or left with the same probability $1/2$ for each direction (one dimension).

For BMD, it is a biased random walk. Walker steps right or left with different probs.

For example, consider time interval $h$, step size $\sigma \sqrt{h}$,
$$
p(X(t+h) = x + \sigma \sqrt{h} | X(t) = x) = \frac{1}{2} (1 + \frac{\mu}{\sigma} \sqrt{h}).
$$
$$
p(X(t+h) = x - \sigma \sqrt{h} | X(t) = x) = \frac{1}{2} (1 - \frac{\mu}{\sigma} \sqrt{h}).
$$

- $\mu > 0$, biased to the right. $\mu < 0$, biased to the left.

- $h$ needs to be small enough to make $|\frac{\mu}{\sigma} \sqrt{h} | \leq 1$.

In this BMD case, $x(t) \sim N(\mu t, \sigma^2 t)$.

- Independent and stationary increments.

(We omit the proof. More details can be found at [Gaussian process](https://www.hajim.rochester.edu/ece/sites/gmateos/ECE440/Slides/block_5_stationary_processes_part_a.pdf) ).


## Geometric Brownian motion (GBM)

Definition. Suppose that $Z(t)$ is a standard Brownian motion $Z(t) \sim N(0, t)$. Parameters $\mu \in \mathbb{R}$ and $\sigma \in (0, \infty)$. Let 
$$
X(t) = \exp[(\mu - \frac{\sigma^2}{2})t + \sigma Z(t)], \qquad t \geq 0. \qquad (2)
$$
The stochastic process $\{X(t): t \geq 0\}$ is geometric Brownian motion with drift parameter $\mu$ and volatility parameter $\sigma$.


-  The process is always positive, one of the reasons that geometric Brownian motion is used to model financial and other processes that **cannot be negative**.
-  For the stochastic process 
  
  $$
(\mu - \frac{\sigma^2}{2})t + \sigma Z(t) \sim N((\mu - \frac{\sigma^2}{2})t , \sigma^2 t),
  $$
  
  it is a BMD with drift parameter $\mu - \sigma^2/2$ and scale parameter $\sigma$. Thus, the geometric Brownian motion is just the exponential of this BMD process.

- Here $X(0) = 1$, the process starts at 1. For GBM starting at $X(0) = x_0$, the process is 
$$
X(t) = x_0 \exp[(\mu - \frac{\sigma^2}{2})t + \sigma Z(t)], \qquad t \geq 0.
$$ 

- GBM is not a Gaussian process.
  
From the definition of GBM (2), we can have the following differential equation:
$$
\begin{aligned}
  \frac{\text{d}X}{\text{d} t} &= \exp[(\mu - \frac{\sigma^2}{2})t + \sigma Z(t)][(\mu - \frac{\sigma^2}{2}) + \sigma \frac{\text{d}Z}{\text{d} t}] \\
  &= X [(\mu - \frac{\sigma^2}{2}) + \sigma \frac{\text{d}Z}{\text{d} t}] \\
  &= X \tilde{\mu} + \sigma X \frac{\text{d}Z}{\text{d} t},  \qquad (\tilde{\mu} := \mu - \frac{\sigma^2}{2})
\end{aligned}
$$
thus, Geometric Brownian motion $X(t)$ satisfies the stochastic differential equation
$$
\begin{aligned}
\frac{\text{d}X}{\text{d} t}  &= X \tilde{\mu} + \sigma X \frac{\text{d}Z}{\text{d} t},  \\
  \text{d}X & = X \tilde{\mu} {\text{d} t} + \sigma X \text{d}Z.
\end{aligned}
$$

The second equation is the Black–Scholes model. In the Black–Scholes model, $X(t)$ is the stock price.

- [Geometric Brownian motion](https://stats.libretexts.org/Bookshelves/Probability_Theory/Probability_Mathematical_Statistics_and_Stochastic_Processes_(Siegrist)/18%3A_Brownian_Motion/18.04%3A_Geometric_Brownian_Motion)


# White Gaussian process

Definition. A white Gaussian noise (WGN) process $W(t)$ is a GP with 

(1) zero mean: $\mu(t) = \mathbb{E}[W(t)] = 0$ for all $t$.

(2) Delta function antocorrelation: $R(t_1, t_2) = \sigma^2 \delta(t_1 - t_2)$.

Here the Dirac delta is often thought as a function that is 0 everywhere
and infinite at 0.
$$
\delta(t) = 
\begin{cases}
\infty, & t=0 \\
0, & t\ne 0
\end{cases}.
$$
The Dirac delta is actually a distribution, a
generalization of functions, and it is defined through the integral of its product with an arbitrary function $f(t)$.
$$
\int_{a}^{b} f(t)\delta(t) \text{d} t
 = 
\begin{cases}
f(0), & a < 0 < b \\
0, & \text{otherwise}
\end{cases}.
 $$
 
<!-- since the autocorrelation function of W(t) is not really a function (it involves the
Dirac delta), WGN cannot model any real physical phenomena. Nonetheless, it is a convenient
abstraction to generate processes that can model real physical phenomena. -->

Properties of white Gaussian noise:

(1) For $t_1 \ne t_2$, $W(t_1)$ and $W(t_2)$ are uncorrelated.
$$
\mathbb{E}[W(t_1)W(t_2)] = R(t_1, t_2) = 0, \qquad t_1 \ne t_2.
$$
This means $W(t)$ at different times are independent.

(2) WGN has infinite variance (large power).
$$
\mathbb{E}[W^2(t)] = R(t, t) = \sigma^2 \delta(0) = \infty.
$$

- WGN is discontinuous almost everywhere.
- WGN is unbounded and it takes arbitrary large positive and negative values at any finite interval.


## White Gaussian noise and Brownian motion

Remember that the Brownian motion is a solution to the differential equation:
$$
\frac{\text{d} X(t)}{\text{d}t} = W(t).
$$
**Why $\frac{\text{d} X(t)}{\text{d}t}$ is called white noise ?**

Proof. Assume $X(t)$ is the integral of a WGN process $W(t)$, i.e., $X(t) = \int_{0}^{t} W(u) \text{d} u$.

Since integration is linear functional and $W(t)$ is a GP, $X(t)$ is also a GP.

A Gaussian process can be uniquely specified by its Mean value function and Autocorrelation function.

(1) The mean function:
$$
\mu(t) = \mathbb{E}[\int_{0}^{t} W(u) \text{d} u] = \int_{0}^{t} \mathbb{E} [W(u)] \text{d} u = 0.
$$
\
(2) The autocorrelation $R_{X}(t_1, t_2)$ with $t_1 < t_2$:
$$
\begin{aligned}
  R_{X}(t_1, t_2) &= \mathbb{E}[(\int_{0}^{t_1} W(u_1) \text{d} u_1)(\int_{0}^{t_2} W(u_2) \text{d} u_2)] \\
  &= \mathbb{E}[\int_{0}^{t_1} \int_{0}^{t_2} W(u_1)  W(u_2) \text{d} u_1 \text{d} u_2] \\
  &= \int_{0}^{t_1} \int_{0}^{t_2} \mathbb{E}[W(u_1)  W(u_2) ]\text{d} u_1 \text{d} u_2 \\
  &= \int_{0}^{t_1} \int_{0}^{t_2} \sigma^2 \delta(u_1 - u_2) \text{d} u_1 \text{d} u_2 \\
  &= \int_{0}^{t_1} \int_{0}^{t_1} \sigma^2 \delta(u_1 - u_2) \text{d} u_1 \text{d} u_2 + \int_{0}^{t_1} \int_{t_1}^{t_2} \sigma^2 \delta(u_1 - u_2) \text{d} u_1 \text{d} u_2 \\
  &= \int_{0}^{t_1} \int_{0}^{t_1} \sigma^2 \delta(u_1 - u_2) \text{d} u_1 \text{d} u_2 + 0\\ 
  &= \int_{0}^{t_1} \sigma^2 \text{d} u_1 \\
  &= \sigma^2 t_1.
\end{aligned}
$$
If $t_2 < t_1$, we can obtain $R_{X}(t_1, t_2) = \sigma^2 t_2$. Thus, $R_{X}(t_1, t_2) = \sigma^2 \min \{t_1, t_2\}$.

The mean function and autocorrelation function are the same as Brownian motion!

Because a Gaussian process can be uniquely determined by its mean value function and autocorrelation function. We can conclude

- The integral of WGN is a Brownian motion process.
- The derivative of Brownian motion is WGN.






# Reference
[Gaussian process](https://www.hajim.rochester.edu/ece/sites/gmateos/ECE440/Slides/block_5_stationary_processes_part_a.pdf)

[Geometric Brownian motion](https://stats.libretexts.org/Bookshelves/Probability_Theory/Probability_Mathematical_Statistics_and_Stochastic_Processes_(Siegrist)/18%3A_Brownian_Motion/18.04%3A_Geometric_Brownian_Motion)

[Geometric Brownian motion](http://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-GBM.pdf)

[White Gaussian noise](https://www.seas.upenn.edu/~ese3030/homework/week_11/week_11_white_gaussian_noise.pdf)