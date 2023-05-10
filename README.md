# Machine learning course at University of Pavia

## Prerequisites

Prerequisites for the course include basic knowledge of GitHub, Colab and python. It is thus required before the course to go through [these](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/slides/0.Prerequisites.pdf) slides as well as the two python basics notebooks: 

* [`python_intro_part1.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/python_basics/python_intro_part1.ipynb)
    * Quickstart
    * Indentation
    * Comments
    * Variables
    * Conditions and `if` statements
    * Arrays
    * Strings
    * Loops: `while` and `for`
    * Dictionaries
* [`python_intro_part2.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/python_basics/python_intro_part2.ipynb)
    * Functions
    * Classes/Objects
    * Inheritance
    * Modules
    * JSON data format
    * Exception Handling
    * File Handling

## Machine Learning Lectures and Tutorials

### Day 1

* Lecture: ML basic concepts [slides: [1.MLBasics.pdf](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/slides/1.MLBasics.pdf)]
    * What is machine learning
    * Notation
    * Supervised Learning
    * Linear regression
    * Linear classification
    * Gradient Descent
    * Overfitting
    * Performance metrics

* Hands-on: advanced python
    * Intro to Numpy: [`numpy_intro.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/python_advance/numpy_intro.ipynb)
    * Intro to Pandas: [`pandas_intro.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/python_advance/pandas_intro.ipynb)
    * Intro to Matplotlib: [`matplotlib_intro.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/python_advance/matplotlib_intro.ipynb)

### Day 2

* Lecture: Neural Networks [slides: [2.NNbasicsAndCNN.pdf](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/slides/2.NNbasicsAndCNN.pdf)] 
    * Intro to neural networks
    * Training
    * Activation functions
    * Deep Neural Networks
    * Convolutional Neural Networks
    * Batch Normalization

* Hands-on: basic NN with Keras for LHC jet tagging task
    * Introduction to dataset and tasks [slides: [3.LHCJetTaggingIntro.pdf](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/slides/3.LHCJetTaggingIntro.pdf)]
    * Dataset exploration: [`1.LHCJetDatasetExploration.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/neural-networks/1.LHCJetDatasetExploration.ipynb)
    * MLP implementation with Keras: [`2.JetTaggingMLP.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/neural-networks/2.JetTaggingMLP.ipynb)
    * Conv2D implementation with Keras: [`3.JetTaggingConv2D.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/neural-networks/3.JetTaggingConv2D.ipynb)
    * Conv1D implementation with Keras: [`4.JetTaggingConv1D.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/neural-networks/4.JetTaggingConv1D.ipynb)

### Day 3

* Lecture: RNN and GNNs [slides: [4.RNNandGNN.pdf](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/slides/4.RNNandGNN.pdf)]
    * Vanilla RNN, LSTMs and GRUs
    * Message Passing framework for graph data
    * Graph Convolutional Neural Networks

* Hands-on: RNN and GNN implementations for different tasks
    * GRU for LHC jet tagging task: [`5.JetTaggingRNN.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/neural-networks/5.JetTaggingRNN.ipynb)
    * Intro to PyTorch: [`pytorch_intro.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/pytorch_basics/pytorch_intro.ipynb) and [`pytorch_NeuralNetworks.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/pytorch_basics/pytorch_NeuralNetworks.ipynb)
    * Intro to PyTorch Geometric (PyG): [`6.IntroToPyG.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/neural-networks/6.IntroToPyG.ipynb)
    * Node classification with PyG on Cora citation dataset: [`7.KCNodeClassificationPyG.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/neural-networks/7.KCNodeClassificationPyG.ipynb)
    * Graph classification with PyG on molecular prediction dataset: [`8.TUGraphClassification.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/neural-networks/8.TUGraphClassification.ipynb)
    * Graph classification with PyG on LHC jet dataset: [`9.JetTaggingGCN.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/neural-networks/9.JetTaggingGCN.ipynb)

 ### Day 4

 * Lecture: Attention Mechanism and Transformers [slides: [5.AttentionAndTransformers.pdf](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/slides/5.AttentionAndTransformers.pdf)]
    * Attension mechanism
    * Graph attention networks
    * Multi-head attention
    * Transformers

 * Hands-on:
    * Transformer model for LHC jet tagging with tensorflow: [`10.JetTaggingTransformer.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/neural-networks/10.JetTaggingTransformer.ipynb)

 ### Day 5

 * Lecture: Unsupervised Learning [slides: [6.UnsupervisedLearning.pdf](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/slides/6.UnsupervisedLearning.pdf)]
    * Unsupervised learning
    * Autoencoders
    * Generative Models
    * VariationalAutoencoders
    * Generative Adversarial Networks
    * Anomaly detection

 * Hands-on:
    * Generate data with vanilla GAN: [`11.VanillaGAN_FMNIST.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/neural-networks/11.VanillaGAN_FMNIST.ipynb)
    * Generate data with VAE: [`12.VAE_FMNIST.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/neural-networks/12.VAE_FMNIST.ipynb)
    * Anomaly detection for LHC jets with AE [`13.JetAnomalyDetectionAE.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/neural-networks/13.JetAnomalyDetectionAE.ipynb)
    * Anomaly detection for LHC jets with VAE [`14.JetAnomalyDetectionVAE.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/neural-networks/13.JetAnomalyDetectionVAE.ipynb)    