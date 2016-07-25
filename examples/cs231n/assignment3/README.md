# `minpy` Tutorial - CS 231n Assignment 3, v1.0

Date: July 25, 2016

This homework / tutorial is based on the [assignemnt 3](http://cs231n.github.io/assignments2016/assignment3/) of CS231n Winter 2016. [CS 231n](http://cs231n.stanford.edu/) is a Stanford course taught by Fei-Fei Li, Andrej Karpathy, and Justin Johnson. This tutorial is built on the original `numpy` version of CS 231n assignment 3 and demonstrate how to use `minpy` to speed up the common machine learning experiments with GPU acceleration and auto derivation of gradient function.

The assignment consists of four sections:
1. Image Captioning with Vanilla RNNs (`RNN_Captioning.ipynb`).
2. Image Captioning with LSTMs (`LSTM_Captioning.ipynb`)
3. Image Gradients: Saliency maps and Fooling Images (`ImageGradients.ipynb`).
4. Image Generation: Classes, Inversion, DeepDream (`ImageGeneration.ipynb`).

These four sections have already been implemented as the assignment required. Moreover, the `minpy` version is also implemented in the files with name ending with `_minpy`. For first two sections focusing on RNN implementation, we provide `minpy` version that compares the original `numpy` operations side by side. For the last two sections, since they are visualization experiments, we only attach the solution for the original version for readers' convenience.

Since currently `minpy` is still in active development and haven't ported some commonly used `numpy` functions, we will use `numpy` to process the data and use `minpy` in core neural network implementation.

As the development goes on, we will consistently improve this tutorial.