# Double-deep-Q-learning-Space-Invaders

Agent is trained on Google Colaboratory. Serveral problems have arosed using this platform:

**Memory limitation**: Google Colab provides 12 GB of RAM memory, but that is not enough to store 1000000 (s,a,r,s',d) tuples (which is around 58 GB of memory, if you resize frames to (84,84) and use uint8 to store them as grayscale) in replay buffer. So i came up with an idea to binarize the frames so that I can use 1 bit (instead of 8) to store one pixel. As there is no data type in Python that is represented with 1 bit, I have packed them using numpy.packbits. This allows me to store 300000 tuples in the replay buffer, that uses about 9 GB of RAM (with 2 very big convolutional networks). 
**Random session termination**: Google Colab randomly terminates session if you are using GPU, because it is not made for long-running computations. To get around this problem, I have been periodically saving replay buffer and network weights on the Google Drive. Saving such a large amount of data eats a lot of RAM memory (2-3 GB).
