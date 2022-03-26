# How to use

Install Python and Jupyter Notebook or Jupyter Lab. Then run from this place as ```jupyter notebook```. Your app is accessible at ```http://localhost:8889/?token=<your-token>```.

Our ANN as MLP is implemented in Python 3.8.0, TensorFlow: https://www.tensorflow.org/ and Keras: https://keras.io/. Keras is a handy facade for TensorFlow for those who don't have to work with low-level API and code.

# Why we should use Artificial Neural Networks to search for hidden relationships between organizational patterns?

Simply because they are one of the most effective techniques how to encode information about the real world
in artificial representation. We encode this information in hidden layers. This is the technique which allows 
us to even learn these representations. Learned representations are considered much more efficient than hand-crafted
representations (e.g. with Prolog or through other languages for building knowledge base.)

# Why do we need to use multi-layer neural networks to search for hidden relationships between organizational patterns?

Each of the layers represent information about the pattern in some level. Combination of the layers together
solve our problem. Thus we try to decompose our bigger problem to the smaller parts - layers of the ANN.
Almost all ML techniques can be transformed to their neural network representation (ANN with 3 layers max.)

# Why should we use deep Learning techniques for this task, instead of the machine learning techniques?

Deep learning techniques enable us to solve tasks which cannot be solved with simple machine learning techniques. 
They allow us to represent information about the processes described in organizational patterns and learn these
representations automatically. This means to automatically enhance them to be better representations.

# Why do we talk about Multi-layer perceptrons instead of the neurons and / or perceptrons?

Logical neuron designed by Pitts & McCulloch in 1943 works only with integer or binary-encoded attributes.
Their logical neuron also assumes that all attributes in a dataset have a same weight. In our case, this would mean
all n-grams for some organizational pattern in the matrix of Waseeb et al. in 'Extracting Relations Between
Organizational Patterns Using Association Mining' on page 4. Table 1 would contribute same to final output
of our probability model. Activation function for this logical neuron creates another problem for us. It does
not exhibit information ahead through the network if sum of the n-gram incidents is not bigger than some threshold T.

# What is the biggest advantage of ANN and deep learning? Why we should use these techniques here?

Artificial neural networks allow us to generalize and abstract information we find in the study of organizational
patterns. Deep learning techniques allow us to learn these representations and enhance them. 

Have fun with this repo! We don't use iterators here! (https://d2l.ai/chapter_linear-networks/image-classification-dataset.html) :)
