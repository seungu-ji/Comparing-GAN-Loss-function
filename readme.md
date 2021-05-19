# **Compare GAN Losses by gan evaluating metrics**

In this repository, We evaluate the performance of the GAN's loss functions(ex. LSGAN or WGAN etc..) as an indicator of GAN performance such as **Inception Score(IS)** or **Frechet Inception Distance(FID)**.

## **Directors structure**

```
Compare Gan Loss Function
+---[images]
|   +---....
|
+---[trainingMethod]
|   +---__init__.py
|   |---original_gan.py
|   |---lsgan.py
|   |---wgan.py
|
|---main.py
```

## **GAN Losses**

Why GAN has many losses?

because of original GAN's problems

### 1. Oiginal GAN

paper link: https://arxiv.org/abs/1406.2661

original gan loss function is:
![Original GAN](./images/original_gan_loss_function.PNG)

many gan models suffer the following major problems:
- Non-convergence: the model parameters oscillate, destabilize and never converge
- Mode collapse: the generator collapses which produces limited varieties of samples
- Diminished gradient: the discriminator gets too successful that the generator gradient vanishes and learns nothing
- Unbalance between the generator and discriminator causing overfitting
- Highly sensitive to the hyper-parameter selections

### 2. LSGAN

paper link: https://arxiv.org/abs/1611.04076

Least Square Generative Adversarial Network, or LSGAN for short, is an extension to the GAN architecture that addresses the problem of **vanishing gradients** and **loss saturaton**.

![LSGAN](./images/lsgan.PNG)

It is motivated by the desire to provide a signal to the generator about fake samples that are far from discriminator model's decision boundary for classifying them as real or fake. The further the generated images are from the decision boundary, the larger the error signal provided to generator, encouraging the generation of more realistic images.

The LSGAN can be implemented with a minor change to the output layer of the disciminator layer and the adoption of the least squares, or L2, loss function.


least square gan loss function is:
![LSGAN LOSS](./images/lsgan_loss_function.PNG)

### WGAN

paper link: https://arxiv.org/abs/1701.07875

Wasserstein Generative Adversarial Network, or WGAN for short, is a GAN variant which uses the 1-Wasserstein distance, rather than JS-Divergence, to measure the difference between the model and target distributions. This prevents the GAN mode collapse problem.

![WGAN](./images/wgan.PNG)

![WGAN LOSS](./images/wgan_loss_function.PNG)

![WGAN ALGORITHM](./images/wgan_algorithm.PNG)


### WGAN-GP

This is the models moving forward: 

![GAN MODELS](./images/GAN_losses.PNG)


## **GAN Evaluating Metrics**

- Inception Score (IS)
- Frechet Inception Distance (FID)