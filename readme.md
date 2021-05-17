# **Compare GAN Losses by gan evaluating metrics**

In this repository, We evaluate the performance of the GAN's loss functions(ex. LSGAN or WGAN etc..) as an indicator of GAN performance such as **Inception Score(IS)** or **Frechet Inception Distance(FID)**.

## **Directors structure**

```
Compare Gan Loss Function
+---[trainingMethod]
|   +---__init__.py
|   |---original_gan.py
|   |---lsgan.py
|   |---wgan.py
|
+---[models]
|   +---__init__.py
|   |---layers.py
|   |---model.py
|
+---[utils]
|   +---utils.py
|   |---config.py
|
|---main.py
```

## **GAN Losses**

Why GAN has many losses?

because original GAN is so 불안정하다.

### Oiginal GAN

### LSGAN

- WGAN
- WGAN-GP

This is the models moving forward: EBGAN, BEGAN, DRAGAN, SGAN, RSGAN, RaGAN, HingeGAN

## **GAN Evaluating Metrics**

- Inception Score (IS)
- Frechet Inception Distance (FID)