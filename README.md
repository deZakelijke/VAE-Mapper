## Path Planning with a Variational Autoencoder
Made by Micha de Groot

The training data is a sequence of images extracted from a video. Given a video, it can be converted to th required images with the following command

ffmpeg -i video.mp4 -vf fps=10 %d.png

One of the video's used is a video from the UvA FNWI Robotics lab and can be found at the following link:
https://youtu.be/ao6auAcw0dc
