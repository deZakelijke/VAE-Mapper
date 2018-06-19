## Path Planning with a Variational Autoencoder
Made by Micha de Groot

The training data is a sequence of images extracted from a video. Given a video, it can be converted to th required images with the following command

ffmpeg -i video.mp4 -vf fps=10 %d.png

The model can then be trained by running train\_VAE.py
Although the training of the VAE can be done without a GPU, this is stronly discouraged.

After the VAE model is trained, path generation can be done with either gen\_path.py or with graph\_search.py

The extension with the GAN does not quite work since the training is still unstable.
