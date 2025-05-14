---
title: Itô Calculus and Stochastic Differential Equations
date: 2023-08-24 11:00:00
tags: SDE
categories: Generative Model
mathjax: true
---

# The Stochastic Integral of Itô

A stochastic differential equation can be transformed into a vector differential equation of the form
$$
\begin{aligned}
\frac{\text{d}x}{\text{d} t}  &= f(x, t) + \mathbb{L}(X, t) w(t), \\
\end{aligned}
$$
where $w(t)$ is a white Gaussian noise with zero mean. Since $w(t)$ is discontinuous, we can not use the ordinary differential equation to solve the above equation. Fortunately, we can reduce the problem to definition of a new king of integral, the stochastic integral of Itô.

We can integrate the SDE from initial time $t_0$ to final time $t$:
$$
\begin{aligned}
x(t) - x(t_0) &= \int_{t_0}^{t} f(x, t) \text{d} t + \int_{t_0}^{t} \mathbb{L}(x, t) w(t) \text{d} t. \\
\end{aligned}
$$
The first integral with respect to time on the right-hand side can be solved by Riemann integral or Lebesgue integral. The second integral is the problem we need to solve. We will first discuss the reason why we can not use the Riemann integral, Lebesgue integral and Stieltjes integral to solve the second integral.

First, it cannot be solved by Riemann integral. The Riemann integral is defined as
$$
\int_{t_0}^{t} \mathbb{L}(X, t) w(t) \text{d} t = \lim_{n \to \infty} \sum_{k=1}^{n} \mathbb{L}(x(t_k^{*}), t_k^{*}) w(t_k^{*}) (t_{k+1} - t_k),
$$
where $t_0 < t_1 < ...<t_n = t$, and $t_k^{*} \in [t_k, t_{k+1}]$. In Riemann integral, the upper and lower bounds of the integral are defined as the selections of $t_k^*$ such that the integral is maximized and minimized. If the upper bound and lower bound converge to the same value, the Riemann integral exists. However, the white Gaussian noise is discontinuous and not bounded, it can take arbitrarily small and large values at every finite interval, so the upper and lower bounds of the integral are not convergent. Therefore, the Riemann integral does not exist.

For Stieltjes integral, we need to interpret the increment $w(t) \text{d}t$ as an increment of another process $\beta(t)$, thus the intergal becomes
$$
\int_{t_0}^{t} \mathbb{L}(x(t), t) w(t) \text{d} t = \int_{t_0}^{t} \mathbb{L}(x(t), t) \text{d} \beta(t).
$$
Here $\beta(t)$ is a Brownian motion. Brownian motion is a continuous process. However, the Brownian motion is not differentiable, so the Stieltjes integral does not converge.

Both Stieltjes and Lebesgue integrals are defined as limits of the form
$$
\int_{t_0}^{t} \mathbb{L}(x(t), t) \text{d} \beta = \lim_{n \to \infty} \sum_{k=1}^{n} \mathbb{L}(x(t_k^{*}), t_k^{*}) (\beta(t_{k+1}) - \beta(t_k)),
$$
where $t_0 < t_1 < ...<t_n = t$, and $t_k^{*} \in [t_k, t_{k+1}]$. Both of these definitions would require the limit to be independent of the position on the interval $t_k^{*} \in [t_k, t_{k+1}]$. However, in this case, the limit is not independent of the position on the interval $t_k^{*} \in [t_k, t_{k+1}]$, so the Stieltjes and Lebesgue integrals do not exist.

## Definition of the Stochastic Integral of Itô
For Itô integral, it fixed the choice of $t_k^{*}$ to be $t_k$, thus the limit becomes unique. The Itô integral is defined as
$$
\int_{t_0}^{t} \mathbb{L}(x(t), t) \text{d} \beta = \lim_{n \to \infty} \sum_{k=1}^{n} \mathbb{L}(x(t_k), t_k) (\beta(t_{k+1}) - \beta(t_k)).
$$

The SDE can be defined to be the Itô integral of the form
$$
\begin{aligned}
x(t) - x(t_0) &= \int_{t_0}^{t} f(x(t), t) \text{d} t + \int_{t_0}^{t} \mathbb{L}(x, t) \text{d} \beta(t).
\end{aligned}
$$

The differential form is
$$
\begin{aligned}
\text{d} x &= f(x, t) \text{d} t + \mathbb{L}(x, t) \text{d} \beta(t).
\end{aligned}
$$
or
$$
\begin{aligned}
\frac{\text{d} x}{\text{d} t } &= f(x, t) + \mathbb{L}(x, t) \frac{ \text{d} \beta(t)}{\text{d} t}.
\end{aligned}
$$

- Why don't we consider more general SDEs of the form
$$
\begin{aligned}
\frac{\text{d} x}{\text{d} t } &= f(x(t), w(t), t), 
\end{aligned}
$$
where the white noise $w(t)$ enters the system through a nonlinear transformation. We can not rewrite this equation as a stochastic integral with respect to a Brownian motion, and thus we cannot define the mathematical meaning of this equation.

## Itô Formula

Consider the stochastic integral 
$$
\int_{t_0}^{t} \beta(t) \text{d} \beta(t),
$$
where $\beta(t)$ is a standard Brownian motion with zero mean and diffusion constant $q = 1$. Based on the definition of the Itô integral, we have
$$
\begin{aligned}
\int_{t_0}^{t} \beta(t) \text{d} \beta(t) &= \lim_{n \to \infty} \sum_{k=1}^{n} \beta(t_k) (\beta(t_{k+1}) - \beta(t_k)) \\
&= \lim_{n \to \infty} \sum_{k=1}^{n} [-\frac{1}{2}(\beta(t_{k+1}) - \beta(t_k))^2 +　\frac{1}{2}(\beta^2(t_{k+1}) - \beta^2(t_k))]\\
&= -\frac{1}{2}t + \frac{1}{2}\beta^2(t).
\end{aligned}
$$
where $0 = t_0 < t_1 < ... < t_n = t$ and $\lim_{n \to \infty} \sum_{k=1}^{n} (\beta(t_{k+1}) - \beta(t_k))^2$. That is because $\beta(t_{k+1}) - \beta(t_k) \sim N(0, t_{k+1} - t_k) \sim N(0, \frac{t}{n})$.

So the Itô differential of $\beta^2(t)/2$ is
$$
\begin{aligned}
\text{d} \frac{\beta^2(t)}{2} &= \beta(t) \text{d}\beta(t) + \frac{1}{2} \text{d}t.
\end{aligned}
$$
It is not the same as the ordinary differential of $\beta^2(t)/2$:
$$
\begin{aligned}
\frac{\text{d} \beta^2(t)}{2} &= \beta(t) \text{d}\beta(t).
\end{aligned}
$$
That is because the Itô integral fixes the choice of $t_k^{*}$ to be $t_k$.

Theorem Itô formula: Let $x(t)$ be an Itô process(note: $x(t)$ is a vector process) which is the solution of an SDE of the form
$$
\begin{aligned}
\text{d} x &= f(x, t) \text{d} t + \mathbb{L}(x, t) \text{d} \beta(t),
\end{aligned}
$$
where $\beta(t)$ is a Brownian motion. Consider an arbitrary **scalar** function $\phi(x(t), t)$ of the process, the Itô SDE of $\phi$ is
$$
\begin{aligned}
\text{d} \phi &= \frac{\partial \phi}{\partial t} \text{d} t + \sum_{i}\frac{\partial \phi}{\partial x_i} \text{d} x_i + \frac{1}{2} \sum_{i,j}\frac{\partial^2 \phi}{\partial x_i \partial x_j} \text{d} x_i \text{d} x_j \\
&= \frac{\partial \phi}{\partial t} \text{d} t + (\nabla \phi)^T \cdot \text{d} x + \frac{1}{2} tr\{ \nabla \nabla^T \phi\} \text{d} x \text{d} x^T
\end{aligned}
$$
provided that the required partial derivatives exist, where the mixed partial derivatives are combined according to the rules
$$
\begin{aligned}
\text{d} \beta \text{d} t &= 0, \\
\text{d} t \text{d} \beta &= 0, \\
\text{d} \beta \text{d} \beta^T &= Q \text{d} t.
\end{aligned}
$$
(Q is the diffusion matrix(covariance matrix) of the Brownian motion). It can be derived from the Taylor expansion of $\phi(x(t), t)$. Usually, in deterministic case, we could ignore the second-order, we have
$$
\begin{aligned}
\text{d} \phi &= \frac{\partial \phi}{\partial t} \text{d} t + \frac{\partial \phi}{\partial x} \text{d} x.
\end{aligned}
$$
In stochastic case, because $\text{d} \beta \text{d} \beta^T = Q \text{d} t$, which is order one, the $\text{d} x \text{d} x^T$ is potentially of order one, so we need to consider the second-order term.

- Here the Itô formula is derived for a scalar function $\phi(x(t), t)$. However, for vector function, it works for each of the components of a vector-valued function separately and thus
also includes the vector case.



Example: We can apply the Itô formula to the function $\phi(x(t), t) = x^2(t)/2$, with $x(t) = \beta(t)$, where $\beta(t)$ is a standard Brownian motion (q=1). The Itô SDE of $\phi$ is
$$
\begin{aligned}
\text{d} \phi &= \beta \text{d} \beta + \frac{1}{2} \text{d} \beta \text{d} \beta \\
&= \beta \text{d} \beta +\frac{1}{2} \text{d} t.
\end{aligned}
$$


# Reference

Simo Särkkä and Arno Solin (2019). Applied Stochastic Differential Equations. Cambridge University Press.