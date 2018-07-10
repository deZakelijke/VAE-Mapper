## Path Planning with a Variational Autoencoder
Made by Micha de Groot

This repository contains the code used in the thesis: (link not available yet)

The overall goal is to train va VAE on image data from an unchanging location, preferrably indoors, and then generate a path from A to B consisting of a sequence of images, where the first frame is image A and the last frame is image B. The model does this by encoding A and B and the performing path planning through the latent space.

#### Useful information for running the code
The training data is a sequence of images extracted from a video. Given a video, it can be converted to th required images with the following command

ffmpeg -i video.mp4 -vf fps=10 %d.png

The model can then be trained by running train\_VAE.py
Although the training of the VAE can be done without a GPU, this is stronly discouraged.

After the VAE model is trained, path generation can be done with either gen\_path.py or with graph\_search.py

The extension with the GAN does not quite work since the training is still unstable.
