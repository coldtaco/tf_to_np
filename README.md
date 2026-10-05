Quick hack for converting a trained tensorflow model to a purely numpy one. Developed for use for hosting a model on my Raspberry Pi 3 as Tensorflow isn't supported on 32-bit systems.

Currently supports softmax, ReLU, 2D convolution, Dense, Max pooling.
Check out notebooks on how to run.
Also has https://arxiv.org/abs/1604.00825

# To-do list
- Improve convolutions from nested for loops to (matrix multiplications)[https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication]


`tf_to_np/create_model.py` converts a tf model to a numpy model
`tf_to_np/Layers.py` numpy versions of tf functions
`tf_to_np/lrp_test.py` Demonstration  of Layer-Wise Relevance Propagation, using backpropagation to determine which pixels in the source image had the highest impact in the classifier's decision
`tf_to_np\Old vs new convolution speedtest.ipynb` Speedtest comparing my implementations of my np functions
`tf_to_np\speedtest.ipynb` Speed test of smaller components before final integration into methods

`discord_snippet\Layers.py` Copy of layers for the Discord bot
`discord_snippet\tamper.py` Exposure of tamper detection in discord, all operations are in memory, no local write writing required.

`train_model.ipynb` Training of the tamper detection image classifier.
