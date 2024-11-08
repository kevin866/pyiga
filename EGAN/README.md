# Bezier-EGAN

Bezier GAN's application to UIUC airfoil database realized within the framework of Entropic GAN.

## Environment

* Python 3.7
* PyTorch 1.6.0
* Tensorboard 2.2.1
* Numpy 1.19.1
* Scipy 1.5.2

## Scripts

* **train_e**: _Training algorithm for entropic GAN._
* **train_v**: _Training algorithm for vanilla Bezier-GAN._
* **plot_latent**: _Plotting program for latent space examination._
* **metrics_exam**: _Calculate metrics for the generator._
* **models**
  * **layers**: _Elementary PyTorch modules to be embedded in cmpnts._
    * Bezier layer generating data points.
    * Regular layer combos for components in cmpnts.
  * **cmpnts**: _General PyTorch neural network components for advanced applications._
    * Basic components such as MLP, Conv1d front end etc.
    * Generators for a variety of applications.
    * Discriminators for a variety of applications.
  * **gans**: _Various GAN containers built on top of each other._
    * GAN: Trained with JS divergence.
    * InfoGAN: Child of GAN trained with additional mutual information maximization.
    * BezierGAN: Child of InfoGAN trained with additional bezier curve regularization.
    * EGAN: Child of GAN trained with entropic dual loss.
    * BezierEGAN: Child of EGAN and BeizerGAN.
  * **utils**: _Miscellaneous tools_
* **utils**
  * **dataloader**: _Data related tools for GAN's training process._
    * Dynamic UIUC dataset that can generate samples with given parameters.
    * Noise generator producing a given batch of uniform and normal noise combination.
  * **interpolation**: _Interpolation algorithm._
  * **metrics**:
    * MMD
    * Consistency
    * MLL
    * RVOD
    * Diversity
  * **shape_plot**: _Generate airfoil grids for demonstration._
* **configs**
  * default: default configuration for Bezier-EGAN.
  * vanilla: default configuration for Bezier-GAN.