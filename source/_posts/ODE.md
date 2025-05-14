---
title: Ordinary Differential Equation
date: 2023-07-19 16:00:00
tags: Diffusion
categories: Generative Model
mathjax: true
---

# Ordinary Differential Equation (ODE)

## The defination of ODE
An ODE is an equation in which the unknown quantity is a function, and it also involves derivatives of the unknown function.

For example, the forced spring-mass system:
$$
\frac{d^2 x(t)}{d t^2} + \gamma \frac{d x(t)}{dt} + v^2 x(t) = w(t).  \qquad (1)
$$
In this equation:

- $v$ and $\gamma$ are constants that determine the resonant angular velocity
and damping of the spring.

- $w(t)$ is a given function that may or may not depend on time.
- position variable $x$ is called dependent variable.
- time $t$ is called independent variable.
- This equation is **second order**. It contains the second derivative and doesn't have higher-order terms.
- This equation is **linear**. $x(t)$ is linearly. There is no terms like $x^2(t), \log x(t)$...
- This equation is **inhomogeneous**. Because it contains forcing term $w(t)$.
  
## The solution to ODE
It can be divided to two categories:

- particular solution: a function that satisfies the differential equation and does not contain any arbitrary constants.

- general solution: a function that satisfies the differential equation and contains free constants.
  
To exactly solve the differential equation, it is necessary to combine the general
solution with some initial conditions (i.e., $x(t_0)$, $\frac{d x(t)}{dt} |_{t_0}$) or some other (boundary) conditions of the differential equation.

## Different formulation of ODE

It is common to omit the time $t$, so equation (1) can also be writen is this form:
$$
\frac{d^2 x}{d t^2} + \gamma \frac{d x}{dt} + v^2 x = w.  
$$

Sometimes, time derivatives are also denoted with dots over the variable,
for example:

$$
\ddot{x} + \gamma \dot x + v^2 x = w.  \qquad (2)
$$ 

## State-space form of the differential equation (first-order vector differential equation)

- **order**: the order of a differential equation is the order of the highest derivative that appears in the equation.
  
-  Order $N$ ODE can convert to Order 1 vector ODE
i.e.,
if we define a state variable $\vec{x} = (x_1 = x, x_2 = \dot{x})$, then we can convert equation（2） to
$$\begin{pmatrix}
\frac{d x_1 (t)}{dt} \\
\frac{d x_2 (t)}{dt}
\end{pmatrix}
= 
\begin{pmatrix}
    0 & 1 \\
    -v^2 & -\gamma
\end{pmatrix}
\begin{pmatrix}
x_1 (t) \\
x_2 (t)
\end{pmatrix} + 
\begin{pmatrix}
0 \\
1 
\end{pmatrix}
w(t) \qquad (3)
$$

Define $\frac{d \vec{x}(t)}{dt} = \begin{pmatrix}
\frac{d x_1 (t)}{dt} \\
\frac{d x_2 (t)}{dt}
\end{pmatrix},$ $f(\vec{x}(t)) = 
\begin{pmatrix}
0 & 1 \\
-v^2 & -\gamma
\end{pmatrix}
\begin{pmatrix}
x_1 (t) \\
x_2 (t)
\end{pmatrix}$ and 
$\mathbf{L} = \begin{pmatrix}
0 \\
1 
\end{pmatrix}.
$

Equation (3) can be seen to be a special case of this form:
$$
\frac{d \vec{x}(t)}{dt} = f(\vec{x}(t), t) + \mathbf{L}(\vec{x}(t), t)\vec{w}(t)
$$
where the vector-valued function $\vec{x}(t) \in R^D$ is the state of the system. $f(\cdot, \cdot)$ and $\mathbf{L}(\cdot, \cdot)$ are arbitrary functions. $\vec{w}(t)$ is some (vector-valued) forcing function, driving function, or input to the system.


- The first-order vector differential equation representation of an $n$th-order
differential equation is often called the state-space form of the differential
equation. 

- The theory and solution methods for first-order vector differential equations are easier to analyze.
  
- And, $n$ th order differential equations can (almost) always be converted into equivalent $n$-dimensional vector-valued first-order differential equations.



## Linear ODEs

Equation (3) is also a special case of the **linear differential equations**:
$$
\frac{d \vec{x}(t)}{dt} = \mathbf{F}(t) \vec{x}(t) + \mathbf{L} \vec{w}(t).
$$
where $\mathbf{F}(t)$ is a matrix-valued function of time. $\mathbf{L}$ is a matrix. $\vec{w}(t)$ is a vector-valued function of time.

- **homogeneous**: the equation is homogeneous if the forcing function $\vec{w}(t)$ is zero for all $t$.
- **time-invariant**: the equation is time-invariant if $\mathbf{F}(t)$ is constant for all $t$.

In the following sections, we first start with the simple scalar linear time-invariant homogeneous differential equation with fixed initial condition at $t = 0$. Then, we will consider the multidimensional generalization of this equation. Besides, we also consider the linear time-invariant inhomogeneous differential equations. Finally, we will consider more general differential equations.

#### Solutions of Linear Time-Invariant Differential Equations

Consider the **scalar** linear **homogeneous** differential equation with fixed initial condition at $t = 0$:
$$
\frac{d x(t)}{dt} = F x(t),  x(0) = \text{given}, \qquad (4)
$$
where $F$ is a constant.

This equation can be solved by separation of variables:
$$
\frac{d x(t)}{x(t)} = F dt.
$$
Integrating the left-hand side from $x(0)$ to $x(t)$, and right-hand side from $0$ to $t$, we get
$$
\log \frac{x(t)}{x(0)} = F t.
$$
Combining the initial condition, we get
$$
x(t) = x(0) e^{F t}.
$$

Another way to solve this equation is to by intergrating both sides of equation (4) from $0$ to $t$. We get
$$
x(t) = x(0) + \int_0^t F x(\tau) d \tau.
$$
The we can substitute the solution back into the right-hand side of the equation, and we get
$$
\begin{aligned}
x(t) &= x(0) + \int_0^t F x(\tau) d \tau \\
&= x(0) + \int_0^t F [x(0) + \int_0^{\tau} F x(\tau') d \tau'] d \tau \\
&= x(0) + F x(0) t +  \int_0^t \int_0^{\tau}F^2 x(\tau') d \tau' d \tau \\
&= x(0) + F x(0) t +  F^2 x(0) \frac{t^2}{2} + \int_0^t \int_0^{\tau} \int_0^{\tau'}F^3 x(\tau'')  d \tau'' d \tau' d \tau \\
&= ...\\
&= x(0) + F x(0) t +  F^2 x(0) \frac{t^2}{2} + F^3 x(0) \frac{t^3}{6} + \cdots\\
&=  (1 + Ft + F^2 t^2 /2 + F^3 t^3 /3! + \cdots) x(0)\\
&=  e^{F t} x(0).  \qquad (\text{Taylor expansion})
\end{aligned}
$$

For the **multidimensional** linear **homogeneous** differential equation with fixed initial condition at $t = 0$:

$$
\frac{d \vec{x}(t)}{dt} = \mathbf{F} \vec{x}(t),  \vec{x}(0) = \text{given}, \qquad (5)
$$
where $\mathbf{F}$ is a constant matrix.

We can not use the separation of variables method to solve this equation, because the $\vec{x}(t)$ is a vector. But we can use the series-based method to solve this equation. Similar to the scalar case, we can get a solution of equation (5):
$$
\vec{x}(t) =  e^{\mathbf{F} t} \vec{x}(0).  \qquad (6)
$$
where the matrix exponential $e^{\mathbf{F} t} = \mathbf{I} + \mathbf{F}t + \frac{\mathbf{F}^2 t^2}{2!} + \cdots$.

Remark:

- The matrix exponential is not the same as the element-wise exponential of a matrix. Thus, we can not compute it by using the element-wise exponential function.
  
- But the matrix exponential function can be found as a built-in function in many programming languages, such as MATLAB and Python.

- Another way to compute it analytically, by using the Taylor expansion of the matrix exponential function, the Laplace or Fourier transform.

Next, let's move to the linear **time-invariant** **inhomogeneous** differential equations:
$$
\frac{d \vec{x}(t)}{dt} = \mathbf{F} \vec{x}(t) + \mathbf{L} \vec{w}(t),  \vec{x}(t_0) = \text{given}, \qquad (7)
$$
where $\mathbf{F}$ is a constant matrix, $\mathbf{L}$ is a constant matrix, and $\vec{w}(t)$ is a vector-valued function of time.

First, we can move the $\mathbf{F} \vec{x}(t)$ to the left-hand side of the equation and multify both sides with a integrating factor $\exp(- \mathbf{F}t)$, and get
$$
\begin{aligned}
\exp(- \mathbf{F}t) \frac{d \vec{x}(t)}{dt} - \exp(- \mathbf{F}t)\mathbf{F} \vec{x}(t) = \exp(- \mathbf{F}t) \mathbf{L} \vec{w}(t),
\end{aligned}
$$
Since $\frac{d}{dt} \exp(- \mathbf{F}t) \vec{x}(t) = \exp(- \mathbf{F}t) \frac{d \vec{x}(t)}{dt} - \exp(- \mathbf{F}t)\mathbf{F} \vec{x}(t)$, we can rewrite the above equation as
$$
\frac{d}{dt} \exp(- \mathbf{F}t) \vec{x}(t) = \exp(- \mathbf{F}t) \mathbf{L} \vec{w}(t).
$$
Integrating both sides from $t_0$ to $t$, we get
$$
\exp(- \mathbf{F}t) \vec{x}(t) - \exp(- \mathbf{F}t_0) \vec{x}(t_0) = \int_{t_0}^t \exp(- \mathbf{F}\tau) \mathbf{L} \vec{w}(\tau) d \tau.
$$
Then, we can get the solution of equation (7):
$$
\vec{x}(t) = \exp(\mathbf{F}(t - t_0)) \vec{x}(t_0) + \int_{t_0}^t \exp(\mathbf{F}(t - \tau)) \mathbf{L} \vec{w}(\tau) d \tau. \qquad (8)
$$

### Solutions of General Linear ODEs
The previous section only consider the linear time-invariant differential equations($F$ is a constant). In this section, we will consider the general linear differential equations with time-varying coefficients.

For the linear **homogeneous** **time-varying** differential equation with fixed initial condition at $t = t_0$:
$$
\frac{d \vec{x}(t)}{dt} = \mathbf{F}(t) \vec{x}(t), \quad \vec{x}(t_0) = \text{given}, \qquad (9)
$$
where $\mathbf{F}(t)$ is a matrix-valued function of time.
We can not use the exponential matrix to solve this equation, because the $\mathbf{F}(t)$ is a time-varying matrix. But the solution of this equation has a general form:
$$
\vec{x}(t) = \mathbf{\Psi}(t, t_0) \vec{x}(t_0), \qquad (10)
$$
where $\mathbf{\Psi}(t, t_0)$ is a matrix-valued function of time, and it is called the **transition matrix**. The transition matrix $\mathbf{\Psi}(t, t_0)$ satisfies the following differential properties:
$$
\begin{aligned} 
\frac{\partial \mathbf{\Psi}(\tau, t)}{\partial \tau} &= \mathbf{F}(\tau) \mathbf{\Psi}(\tau, t), \\
\frac{\partial \mathbf{\Psi}(\tau, t)}{\partial t} &= - \mathbf{\Psi}(\tau, t) \mathbf{F}(t), \\
\mathbf{\Psi}(\tau, t) &= \mathbf{\Psi}(\tau, s) \mathbf{\Psi}(s, t) \qquad (\text{11}) \\
\mathbf{\Psi}(t, \tau) &= \mathbf{\Psi}^{-1}(\tau, t)  \\
\mathbf{\Psi}(t, t) &= \mathbf{I}.
\end{aligned}
$$
In most cases, the transition matrix $\mathbf{\Psi}(t, t_0)$ does not have a closed-form solution. 

For the linear **inhomogeneous** **time-varying** differential equation with fixed initial condition at $t = t_0$:
$$
\frac{d \vec{x}(t)}{dt} = \mathbf{F}(t) \vec{x}(t) + \mathbf{L}(t) \vec{w}(t), \quad \vec{x}(t_0) = \text{given}, \qquad (12)
$$
where $\mathbf{F}(t)$ is a matrix-valued function of time, $\mathbf{L}(t)$ is a matrix-valued function of time, and $\vec{w}(t)$ is a vector-valued function of time.
The solution of this equation has a general form:
$$
\vec{x}(t) = \mathbf{\Psi}(t, t_0) \vec{x}(t_0) + \int_{t_0}^t \mathbf{\Psi}(t, \tau) \mathbf{L}(\tau) \vec{w}(\tau) d \tau. \qquad (13)
$$

When $\mathbf{F}(t)$ and $\mathbf{L}(t)$ is constant, the solution of equation (7) is a special case of (13) and we can verfiy that the $\Psi(t, t_0) = \exp(\mathbf{F}(t - t_0))$ satisfies properties (10).

Fourier tansforms and Laplace transforms are two useful methods to solve inhomogeneous linear time-invariant ODE (Note that  **time-invariant**).

## Fourier Transforms 
The Fourier transform of a function $x(t)$ is defined as:
$$
X(i w) = \mathcal{F}[x(t)] = \int_{-\infty}^{\infty} x(t) \exp(-iwt) \text{d}t, \qquad (14)
$$
where $i$ is the imaginary unit.

The inverse Fourier transform of (14) is:
$$
x(t) = \mathcal{F}^{-1}[X(i w)] = \frac{1}{2\pi} \int_{-\infty}^{\infty} X(iw) \exp(i w t) \text{d}t.
$$

Some useful properties:

- differention: $\mathcal{F}[\frac{\text{d}^n x(t)}{\text{d} t^n}] = (iw)^{n} \mathcal{F}[x(t)]$.

- convolution: $\mathcal{F}[x(t) \ast y(t)] = \mathcal{F}[x(t)] \ast \mathcal{F}[y(t)]$, where the convolution $\ast$ is defined as
  $$
  x(t) \ast y(t) = \int_{-\infty}^{\infty} x(t - \tau)y(\tau) \text{d}\tau
  $$
  
**If we want to use Fourier transform to solve ODEs, the initial condition must be 0.**

Now, let's use Fourier transform to solve the linear **time-invariant** **inhomogeneous** differential equations:
$$
\frac{d \vec{x}(t)}{dt} = \mathbf{F} \vec{x}(t) + \mathbf{L} \vec{w}(t),  \quad \vec{x}(t_0) = 0,
$$
where $\mathbf{F}$ is a constant matrix, $\mathbf{L}$ is a constant matrix, and $\vec{w}(t)$ is a vector-valued function of time.

Take Fourier tranforms componentwise give
$$
(iw)\vec{X}(iw) = \mathbf{F} \vec{X}(iw) + \mathbf{L} \vec{W}(iw), 
$$
thus, we can get
$$
\vec{X}(iw) = [(iw)\mathbf{I} - \mathbf{F}]^{-1} \ast \mathbf{L} \vec{W}(iw), 
$$
The solution is the inverse Fourier transform 
$$
x(t) = \mathcal{F}^{-1}[[(iw)\mathbf{I} - \mathbf{F}]^{-1} \ast \mathbf{L} \vec{W}(iw)] = \mathcal{F}^{-1}[((iw)\mathbf{I} - \mathbf{F})^{-1}]\ast \mathbf{L} \vec{w}(t), \qquad (15)
$$

Since $\vec{x}(t_0) = 0$, compare the solution (15) with solution (8), we can obtain
$$
 \mathcal{F}^{-1}[((iw)\mathbf{I} - \mathbf{F})^{-1}] = \exp(\mathbf{F}t) u(t),
$$
where $u(t)$ is the Heaviside step function, which is 0 for $t<0$ and 1 for $t \geq 0$.

## Laplace Transforms

The Laplace transform of a function $f(t)$ is defined on space $\{t | t\geq 0\}$:
$$
F(s) = \mathcal{L}[f(t)] = \int_{0}^{\infty} f(t)\exp(-st) \text{d}t, \qquad (16)
$$
where $s = \sigma + iw$.

The inverse transform is $f(t) = \mathcal{L}^{-1}[F(s)]$.

Remember that the Fourier transform needs the initial condition $x(0) = 0$. But Laplace transform can take the initial conditions into account. If $x(0) = \text{given}$, then
$$
\mathcal{L}[\frac{\text{d} x(t)}{\text{d} t}] = s \mathcal{L}[x(t)] - x(0) = s X(s) - x(0).
$$

$$
\mathcal{L}[\frac{\text{d}^n x(t)}{\text{d} t^n}] = s^n X(s) - s^{n-1} x(0) - \cdots - \frac{\text{d} x^{n-1}}{\text{d} t^{n-1}}(0).
$$


Now, let's use Laplace transform to solve the linear **time-invariant** **inhomogeneous** differential equations:
$$
\frac{d \vec{x}(t)}{dt} = \mathbf{F} \vec{x}(t) + \mathbf{L} \vec{w}(t),  \quad \vec{x}(t_0) = \text{given} \ne 0,
$$
where $\mathbf{F}$ is a constant matrix, $\mathbf{L}$ is a constant matrix, and $\vec{w}(t)$ is a vector-valued function of time.

Take Laplace tranforms componentwise give
$$
s X(s) - x(0) = \mathbf{F} X(s) + \mathbf{L} W(s).
$$
Then, 
$$
X(s) = [s\mathbf{I}-\mathbf{F}]^{-1} x(0) + [s\mathbf{I}-\mathbf{F}]^{-1} \ast  \mathbf{L} W(s).  \qquad (17)
$$

Compare the solution (17) with solution (8), we can obtain
$$
\mathcal{L}^{-1}[(s\mathbf{I}-\mathbf{F})^{-1}] = \exp(\mathbf{F}t)
$$
for $t \geq 0$.


# Reference
[Simo Särkkä and Arno Solin (2019). Applied Stochastic Differential Equations. Cambridge University Press.](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf)