---
title: Convergence of Markov Chain
date: 2023-07-15 14:00:00
tags: Markov Chains
categories: Stochastic Process
mathjax: true
---

# Total Variation Distance
Define Total Variation Distance:
$$
d_{TV}(\mu(x), v(x)) = \frac{1}{2} \sum_{x \in \Omega} |\mu(x) - v(x)|
$$
This is equivalent to the following:
$$
d_{TV}(\mu(x), v(x)) = \sum_{x \in \Omega^{-}} (v(x) - \mu(x))
= \sum_{x \in \Omega^{+}} (\mu(x) - v(x))
$$
where $\Omega^{+} = \{x \in \Omega: \mu(x) \geq v(x)\}$, $\Omega^{-} = \{x \in \Omega: \mu(x) < v(x)\}$,
and
$$
d_{TV}(\mu(x), v(x)) = \max_{S \subset \Omega} |\mu(S) - v(S)|
$$
where $\mu(S) = \sum_{x \in S} \mu(x)$, $v(S) = \sum_{x \in S} v(x)$.

Proof:
Define $\Omega^{+} = \{x \in \Omega: \mu(x) \geq v(x)\}$, $\Omega^{-} = \{x \in \Omega: \mu(x) < v(x)\}$, then
$$
\begin{aligned}
d_{TV}(\mu(x), v(x)) &= \frac{1}{2} \sum_{x \in \Omega} |\mu(x) - v(x)| \\
&= \frac{1}{2} \sum_{x \in \Omega^{+}} (\mu(x) - v(x)) + \frac{1}{2} \sum_{x \in \Omega^{-}} (v(x) - \mu(x))
\end{aligned}
$$
Since $\sum_{x \in \Omega} \mu(x) = 1 = \sum_{x \in \Omega^{+}} \mu(x) + \sum_{x \in \Omega^{-}} \mu(x)$, we have
$$
\begin{aligned}
d_{TV}(\mu(x), v(x)) &= \frac{1}{2} \sum_{x \in \Omega^{+}} (\mu(x) - v(x)) + \frac{1}{2} \sum_{x \in \Omega^{-}} (v(x) - \mu(x)) \\
&= \frac{1}{2} \sum_{x \in \Omega^{-}} (v(x) - \mu(x)) + \frac{1}{2} \sum_{x \in \Omega^{-}} (v(x) - \mu(x)) \\
&= \sum_{x \in \Omega^{-}} (v(x) - \mu(x)) \\
&= \sum_{x \in \Omega^{+}} (\mu(x) - v(x))
\end{aligned}
$$
If $S = \Omega^{+}$ or $\Omega^{-}$, then
$$
\begin{aligned}
d_{TV}(\mu(x), v(x)) &= \frac{1}{2} \sum_{x \in \Omega} |\mu(x) - v(x)| \\
&=  \max_{S \in \Omega} \sum_{x \in S} |\mu(x) - v(x)| \\
\end{aligned}
$$

If $S$ contains any elements $x \in \Omega^{-}$, then $d_{TV}(\mu(x), v(x)) = |\sum_{x \in S} (\mu(x) - v(x))| \leq \sum_{x \in \Omega^{+}} (\mu(x) - v(x))$ when $S = \Omega^{+}$.

# Convergence of Markov Chain

Assume we start from state $x$, run Markov chain for $t$ steps, then we get the distribution $P_{x}^{t}$. If we want to prove $P_{x}^{t}$ converges to stationary distribution $\pi$, we need to prove $d_{TV}(P_{x}^{t}, \pi) \rightarrow 0$ as $t \rightarrow \infty$. Thus, we need to bound $d_{TV}(P_{x}^{t}, \pi)$.

Define $d_{x}(t) := d_{TV}(P_{x}^{t}, \pi)$. 

Consider all possible initial states $x \in \Omega$, we define
$d(t) := \max_{x \in \Omega} d_{TV} (P_{x}^{t}, \pi)$.

Assume there are two Markov chains $x_{t} \sim P_{x}^{t}$, $y_{t} \sim P_{y}^{t}$. $x_{t}$ and $y_{t}$ share the same transition probability matrix $P$ but start from different states. Then we define 
$$\bar{d}(t) := \max_{x, y \in \Omega} d_{TV}(P_{x}^{t}, P_{y}^{t}).$$ 

Next, we will prove this Lemma:

**Lemma 1.**
$$
d(t) \leq \bar{d}(t) \leq 2 d(t).
$$

Proof: Let's prove the second inequality $\bar{d}(t) \leq 2 d(t)$. 
$$
\begin{aligned}
\bar{d}(t) &= \max_{x, y \in \Omega} d_{TV}(P_{x}^{t}, P_{y}^{t}) \\
&= \max_{x, y \in \Omega} \frac{1}{2} \sum_{z \in \Omega} |P_{x}^{t}(z) - P_{y}^{t}(z)| \\
& = \max_{x, y \in \Omega} \frac{1}{2} \sum_{z \in \Omega} |P_{x}^{t}(z) - \pi(z) + \pi(z) - P_{y}^{t}(z)| \\
&\leq \max_{x, y \in \Omega} [\frac{1}{2} \sum_{z \in \Omega} |P_{x}^{t}(z) - \pi(z)| + \frac{1}{2} \sum_{z \in \Omega} |\pi(z) - P_{y}^{t}(z)| ]\\
&= \max_{x \in \Omega} d_{TV}(P_{x}^{t}, \pi) + \max_{y \in \Omega} d_{TV}(P_{y}^{t}, \pi) \\
&= d(t) + d(t) \\
&= 2 d(t)
\end{aligned}
$$

For the first inequality $\bar{d}(t) \leq 2 d(t)$, we need to prove $d(t) \leq \bar{d}(t)$.

Define 
$$ 
S_{x,y}^{*} = \arg\max_{S \subset \Omega} （P_{x}^{t}(S) - P_{y}^{t}(S)）
$$

$$
S_{x}^{**} = \arg\max_{S \subset \Omega, y \in \Omega} （P_{x}^{t}(S) - P_{y}^{t}(S)）
$$

$$
S_{x}^{*} = \arg\max_{S \subset \Omega} \sum_{y \in \Omega} \pi(y)（P_{x}^{t}(S) - P_{y}^{t}(S)）
$$
$$
\begin{aligned}
d(t) &= \max_{x \in \Omega} d_{TV}(P_{x}^{t}, \pi) \\
&= \max_{x \in \Omega} \max_{S \subset \Omega}|P_{x}^{t}(S) - \pi(S)| \\
&= \max_{x \in \Omega} \max_{S \subset \Omega^{+}}(P_{x}^{t}(S) - \pi(S))  \\
&= \max_{x \in \Omega} \max_{S \subset \Omega^{+}}(P_{x}^{t}(S) - \sum_{y \in \Omega} \pi(y) P_{y}^{t}(S))  \qquad  (\pi(y) = \sum_{x \in \Omega} \pi(x) P_{x,y}) \\
&= \max_{x \in \Omega} \max_{S \subset \Omega^{+}} \sum_{y \in \Omega} \pi(y) (P_{x}^{t}(S) - P_{y}^{t}(S)).
\end{aligned}
$$
Since for all $S$:
$（P_{x}^{t}(S_{x,y}^{*}) - P_{y}^{t}(S_{x,y}^{*})）\geq （P_{x}^{t}(S) - P_{y}^{t}(S)）$

we have, 
$$
\begin{aligned}
d(t) &= \max_{x \in \Omega} \max_{S \subset \Omega^{+}} \sum_{y \in \Omega} \pi(y) (P_{x}^{t}(S) - P_{y}^{t}(S)) \\
& \leq \max_{x \in \Omega} \sum_{y \in \Omega} \pi(y) (P_{x}^{t}(S_{x,y}^{*}) - P_{y}^{t}(S_{x,y}^{*})) \\
&= \max_{x \in \Omega} \sum_{y \in \Omega} \pi(y) \max_{S}  (P_{x}^{t}(S) - P_{y}^{t}(S)) \\
& \leq  \max_{x \in \Omega} \sum_{y \in \Omega} \pi(y)   (P_{x}^{t}(S_{x}^{**}) - P_{y}^{t}(S_{x}^{**})) \\
& = \max_{x \in \Omega} \sum_{y \in \Omega} \pi(y) \max_{y, S}(P_{x}^{t}(S) - P_{y}^{t}(S)) \\
&= \max_{x, y \in \Omega}\max_{ S \subset \Omega}(P_{x}^{t}(S) - P_{y}^{t}(S))   \qquad (\sum_{y \in \Omega} \pi(y) = 1) \\
&= \max_{x, y \in \Omega} d_{TV}(P_{x}^{t}, P_{y}^{t}) \\
&= \bar{d}(t).
\end{aligned}
$$
that is, $d(t) \leq \bar{d}(t) \leq 2d(t)$.

**What we have proved is that, $d(t)$ is bounded by $\bar{d}(t)$, and $\bar{d}(t)$ is controlled by $\max_{x,y \in \Omega} d_{TV}(P_{x}^{t}, P_{y}^{t})$. Let's find a way to bound $d_{TV}(P_{x}^{t}, P_{y}^{t})$!**

**Define 1. (Coupling)** Assume $x , y \in \Omega$, $x \sim \mu, y \sim v$, $\mu, v$ are two distributions. A joint distribution $w(x, y)$ on $\Omega \times \Omega$ is called **couping** if $\forall x \in \Omega$, $\sum_{y}w(x, y) = \mu(x)$, $\forall y \in \Omega$, $\sum_{x} w(x, y) = v(y)$.

**Lemma 2.** Consider $\mu, v$ defined on $\Omega$,  

a. For any coupling $w(x, y)$ of $\mu, v$, $d_{TV}(\mu, v) \leq P(x \ne y)$.  

b. There always exists a coupling $w(x, y)$ of $\mu, v$ such that $d_{TV}(\mu, v) = P(x \ne y)$.

Proof: a. $\forall z$, $w(z, z) \leq \sum_{y \in \Omega} w(z, y) = \mu(z)$. Similarly, $w(z, z) \leq v(z)$. Thus, $w(z, z) \leq \min(\mu(z), v(z))$. 

$$
\begin{aligned}
P(x \ne y) &= 1 - P(x = y) \\
&= 1 - \sum_{z \in \Omega} w(z, z) \\
&\geq 1 - \sum_{z \in \Omega} \min(\mu(z), v(z)) \\
&= \sum_{z \in \Omega} \mu(z) - \sum_{z \in \Omega} \min(\mu(z), v(z)) \\
&= \sum_{z \in \Omega^{+}} (\mu(z) - v(z)) \\
&= d_{TV}(\mu, v)
\end{aligned}
$$

b. We can construct a coupling $w(x, y)$:
$$
w(x, y) = \left\{
\begin{aligned}
& \min \{\mu(x), v(y)\}, \quad x = y \\
&\frac{(\mu(x) - w(x, x))(v(y) - w(y,y))}{1 - \sum_{z \in \Omega}w(z,z)}, \quad x \ne y
\end{aligned}
\right.
$$

For this joint distribution, we have
$$
\begin{aligned}
P(x \ne y) &= 1 - P(x = y) \\
&= 1 - \sum_{z \in \Omega} w(z, z) \\
&= 1 - \sum_{z \in \Omega} \min(\mu(z), v(z)) \\
&= \sum_{z \in \Omega} \mu(z) - \sum_{z \in \Omega} \min(\mu(z), v(z)) \\
&= \sum_{z \in \Omega^{+}} (\mu(z) - v(z)) \\
&= d_{TV}(\mu, v)
\end{aligned}
$$

Thus, this joint distribution $w(x, y)$ satisfies $d_{TV}(\mu, v) = P(x \ne y)$. Now, we need to prove that this joint distribution $w(x, y)$ is a coupling of $\mu, v$.

$\forall x \in \Omega$, 
$$
\begin{aligned}
\sum_{y \in \Omega} w(x, y) &= \sum_{y=x} \min \{\mu(x), v(y)\} + \sum_{y \in \Omega, y \ne x} \frac{(\mu(x) - w(x, x))(v(y) - w(y,y))}{1 - \sum_{z \in \Omega}w(z,z)} \\
&= w(x, x) + \frac{(\mu(x) - w(x, x))}{1 - \sum_{z \in \Omega}w(z,z)} \sum_{y \ne x} (v(y) - w(y,y)) \\
&= w(x,x) + \frac{(\mu(x) - w(x, x))}{1 - \sum_{z \in \Omega}w(z,z)} (1 - v(x) - (\sum_{z} w(z,z) - w(x,x)))  \qquad (\sum_{y \ne x} v(y) = 1 - v(x))\\
&= w(x,x) + \frac{(\mu(x) - w(x, x))}{1 - \sum_{z \in \Omega}w(z,z)} (1 - \sum_{z} w(z,z) + w(x,x) - v(x) ) \qquad (\sum_{y \ne x} w(y,y) = \sum_{z} w(z,z) - w(x, x))
\end{aligned}
$$
If $ x \in \Omega^{+} = \{x | \mu(x) \geq v(x)\}$, then $w(x,x) = v(x)$, $\sum_{y \in \Omega} w(x, y)  = v(x) + (\mu(x) - v(x)) = \mu(x)$.

If $ x \in \Omega^{-} = \{x | \mu(x) < v(x)\}$, then $w(x,x) = \mu(x)$, $\sum_{y \in \Omega} w(x, y)  = \mu(x) + 0 = \mu(x)$.

Thus, $w(x, y)$ is a coupling of $\mu, v$.

Last but not least, let's begin to prove the nonincreasing property of $d(t)$!! **Almost close to the end!! \o^o/**

**Lemma 3.** Consider two Markov chains $x^{t} \sim P_{x}^{t}$, $y^{t} \sim P_{y}^{t}$, $x^{t}$ and $y^{t}$ share the same transition probability matrix $P$ but start from different states. If $x^t = y^t$, then $x^{t+1} = y^{t+1}$, elif $x^t \ne y^t$, then $x^{t+1} \ne y^{t+1}$, $x^{t+1}$ and $y^{t+1}$ are independent.

**Define 2. (Coupling of Markov chains)** Consider two Markov chains $x^{t} \sim P_{x}^{t}$, $y^{t} \sim P_{y}^{t}$, $x^{t}$ and $y^{t}$ share the same transition probability matrix $P$ but start from different states. If 
$$
P(y^{t+1} | x^{t}, y^{t}) = P(y^{t+1} | y^{t})
$$
and
$$
P(x^{t+1} | x^{t}, y^{t}) = P(x^{t+1} | x^{t})
$$
then we say $x^{t}$ and $y^{t}$ are coupled.

Define $w^{t} := w^{t}(x^t, y^t)$ is a coupling of $P_{x}^{t}, P_{y}^{t}$, $x^t \sim P_x^t, y \sim P_{y}^{t}$, and $w^{t}$ satisfies Lemma 2(b).

$$
P_{w^{t}}(x^{t+1} \ne y^{t+1}) = P_{w^{t}}(x^{t+1} \ne y^{t+1} | x^t \ne y^t) p(x^t \ne y^t) + P_{w^{t}}(x^{t+1} \ne y^{t+1} | x^t = y^t) p(x^t = y^t)
$$
If $x^t = y^t$, according to Lemma 3, $P_{w^{t}}(x^{t+1} \ne y^{t+1}) = P_{w^{t}}(x^{t+1} = y^{t+1} | x^t = y^t) p(x^t = y^t)  \leq P_{w^{t}}(x^{t} \ne y^t)$;

If $x^t \ne y^t$, $P_{w^{t}}(x^{t+1} \ne y^{t+1}) = P_{w^{t}}(x^{t+1} \ne y^{t+1} | x^t \ne y^t) p(x^t \ne y^t) \leq P_{w^{t}}(x^{t} \ne y^t)$.

$d_x(t+1) = d_{TV}(P_{x}^{t+1}, P_y^{t+1}） \leq P_{w^{t}}(x^{t+1} \ne y^{t+1}) \leq P_{w^{t}}(x^{t} \ne y^t) = d_{TV}(P_x^t, P_y^{t}) = d_x(t)$

$d_{x}(t) := d_{TV}(P_{x}^{t}, \pi)$.

$d(t) = \max_{x \in \Omega} d_{x}(t)$.

$d(t) \leq \bar{d}(t) \leq 2d(t)$.

$\bar{d}(t) := \max_{x, y \in \Omega} d_{TV}(P_{x}^{t}, P_{y}^{t})$

Q:  if we consider $d_{TV}(P_x^t, \pi)$, then the coupling $w^{t} := w^{t}(x^t, y)$ should be a coupling of $P_{x}^{t}, \pi$, $x^t \sim P_x^t, y \sim \pi$.

$\pi$ is different from $P_y^t$ at first.

$$
d_x(t) = d_{TV}(P_{x}^{t}, \pi) \leq d(t) \leq \bar{d}(t) = \max_{x, y \in \Omega} d_{TV}(P_{x}^{t}, P_{y}^{t}) = P_{w^t}(x^t \ne y^t)
$$

Now, we have already proved that $d(t)$ is nonincreasing. Next, we will prove that $d(t)$ converges to 0.

# Useful Lectures

- [L1](https://people.eecs.berkeley.edu/~sinclair/cs294/n7.pdf)
- [Markov Chains and Coupling](https://courses.cs.duke.edu/spring13/compsci590.2/slides/lec5.pdf)
- [Markov Chains, Coupling, Stationary Distribution](https://faculty.cc.gatech.edu/~vigoda/MCMC_Course/MC-basics.pdf)
- [Stationary distributions](https://mpaldridge.github.io/math2750/S10-stationary-distributions.html)
- [Long-term behaviour of Markov chains](https://mpaldridge.github.io/math2750/S11-long-term-chains.html)
