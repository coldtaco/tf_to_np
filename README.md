Quick hack for converting a trained tensorflow model to a purely numpy one. Developed for use for hosting a model on my Raspberry Pi 3 as Tensorflow isn't supported on 32-bit systems.

Currently supports softmax, ReLU, 2D convolution, Dense, Max pooling.

# To-do list
- Improve convolutions from nested for loops to (matrix multiplications)[https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication]
