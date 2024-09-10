# Unified Scaling Law of LLM Training and Inference

Here I propose a unified scaling law of LLMs in both training and inference:

Assume we have several key parameters

- Layer $N$
- Hidden Dimension $d$
- Data Size $S$

Then the loss function $L$ is predicted by the scaling law as follows:

$$L = -\log(p) = b + (\frac{x_1}{N^{w_1}}+ \frac{x_2}{d^{w_2}}) + \frac{x_3}{S^{w_3}}$$

For MoE, this term also includes additional coefficients that reflect the expansion of model parameters:

$$L = -\log(p) = b + (\frac{x_1}{N^{w_1}}+ \frac{x_2}{d^{w_2}})\frac{x_4}{E^{w_4}} + \frac{x_3}{S^{w_3}}$$


#### Expectation of correct probability
$$ p = \frac{1}{1+\exp(-\mu)} = e^{-L}$$  

$$\mu = -\log(e^{L}-1)$$

When we sample token logits $y$ from $y = \mathcal{N}(\mu, \sigma^2)$, $\sigma$ depends on data distribution and decoding strategies.


#### Non-CoT mode, accuracy $I$ is $p$

$$I = p$$

* CoT mode, we have $C\geq 2$ critical steps, each step increases the logits of the correct tokens. $y_0$ and $C_0$ are task-dependent constants.

$$y' = y + y_0\cdot \tanh(\frac{C\log(N)}{C_0}) = y + y_0\cdot \frac{N^{\frac{2C}{C_0}}-1}{N^{\frac{2C}{C_0}}+1}$$

$$I = (\frac{1}{1+e^{-y'}})^C$$

#### Repeated Inference w/o Verification

  - Here we change normal distribution: $$\mathcal{N}(\mu, \frac{\sigma^2}{f(R)})$$

  - Repeat is bad when the model is weak or the problem is hard.
  
  - Repeat without verification can improve performance to some extent, but not unlimited.

#### Repeated Inference w/ Perfect Verification

  Under $R$ times of repetition, we use $f(R)$ rather than $R$ because repetitions may not be independent.

$$I' = 1-(1-I)^{f(R)}$$

#### Repeated Inference w/ Partial Verification (the judger has an $\epsilon$ error)

  $$I' = (1-(1-I)^{f(R)})(1-\epsilon)^{f(R)}$$

  Under this case, too many repetitions may be suboptimal.
