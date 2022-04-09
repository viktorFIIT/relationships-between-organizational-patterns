# How to use

Install Python and Jupyter Notebook or Jupyter Lab. Then run from this place as ```jupyter notebook```. Your app is then accessible at ```http://localhost:8889/?token=<your-token>```.

ANN is MLP and is implemented in Python 3.8.0, with TensorFlow: https://www.tensorflow.org/ and Keras: https://keras.io/. Keras is a handy facade of TensorFlow for those who don't want to work with a low-level API.

# Why we should use artificial neural networks to search for hidden relationships between organizational patterns?

Simply because they are one of the most efficient techniques how we can encode information about the real world
in its artificial representation. We encode this information in a hidden layers of the artificial neural network. 
This is one of the accepted techniques which allows us to learn artificial representations. Learned representations
are widely considered to be much more effective than a hand-crafted representations (e.g. with Prolog or through 
other languages for building knowledge base.)

# Why do we need to use multiple-layer artificial neural networks to search for hidden relationships between organizational patterns?

Each of the layers represent information about the pattern in some level. Combination of the layers together enables us
solve our problem. Thus we try to solve our problem by decomposing bigger problem to its smaller parts - layers of the ANN.
Almost all ML techniques can be transformed back to their neural network representation (ANN with 3 layers max.)

# Why we should use deep learning techniques instead of the machine learning techniques?

Deep learning techniques enable us to solve tasks which cannot be solved with simple machine learning techniques. 
They allow us to represent information about the processes described in organizational patterns and learn these
representations automatically. This means to automatically enhance them to have better representations.

Convolutional Neural Networks are a great way how we can search for patterns as flexibly as we can. Convolutional
Neural Networks can be used to connect multiple patterns to create bigger whole, because of their ability to search
for global tendencies. Multiple-layer perceptrons, in the other hand, can be only used to search for local tendencies.

See more about the CNN if you're interested here https://d2l.ai/chapter_convolutional-neural-networks/index.html

# Why do we talk about the multi-layer perceptrons instead of the neurons or perceptrons?

Logical neuron originally designed by Pitts & McCulloch in 1943 works only with integer or with binary-encoded attributes.
Their logical neuron also assumes that all attributes in a dataset have a same weight. In our case, this would mean
that all n-grams for some organizational pattern in a matrix of Waseeb et al. in 'Extracting Relations Between
Organizational Patterns Using Association Mining' on page 4. Table 1 would contribute same to the final output
of our probability model. Activation function for this logical neuron creates another problem for us. It does
not exhibit information ahead through the network if sum of the n-gram incidents is not bigger than some predefined threshold T.

# What is the biggest advantage of ANN and deep learning techniques? Why we should use these techniques here?

Artificial neural networks allow us to generalize and abstract information we find in the study of organizational
patterns. Deep learning techniques allow us to learn these representations and enhance them. 

Have fun with this repo! We don't use iterators here! (https://d2l.ai/chapter_linear-networks/image-classification-dataset.html) :)
