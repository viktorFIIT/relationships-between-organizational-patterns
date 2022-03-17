# How to use

This folder contains 4 trained implementations of the neural networks coded in Python v. 3.8.0. First of these files is better suitable for research, because it is coded solely in Python. Simples of those makes use of the gradient descent algorithm to learn its parameters. Next one use advanced optimilizers.

Implementations of these neural networks expect numeric attributes. This means one (possibly hot-encoded) column for encoded pattern name (dependent variable) and how many independent numeric attributes you want.

File ```analysis-of-optimalization-algorithms``` is a Jupyter notebook with 3 multi-layer perceptrons. These can be used to predict probability of the patterns in a dataset.

File ```softmax-classification-in-tensorflow-keras``` is implementation of the Softmax classification in TensorFlow framework (and its Keras facade) which can be used to classify patterns based on the maximum likelihood.

Check https://www.tensorflow.org/ for further info.

Tested on Intel(R) Core(TM) i5-3320M CPU @ 2.60GHz   2.60 GHz, 4GB RAM, Intel(R) GD Graphics 4000 and on HP Pavilion Gaming Laptop 15, AMD Ryzen 5 5600H (6 cores) with Radeon Graphics, 16 GB RAM

Notebooks can be used by running ```jupyter notebook``` or ```jupyter-lab```. Check https://jupyter.org/ to get it.

Best learning resources I was able to find so far are video presentations on: https://vgg.fiit.stuba.sk/teaching/neural-networks/ or book Dive Into Deep Learning from Zhang. et al (https://d2l.ai/)
