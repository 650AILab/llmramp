# Transformer Architecture : Chapter 05 - Feedforward Neural Network #

In this section you will learn about feed forward networks, their usage and how various activation functions are used inside the feed forward networks.


## What is a feedforward neural network?

In the transformer architecture, the feedforward neural network is used as a part of the position-wise fully connected layer, which is applied after the self-attention layer in both the encoder and decoder.

The position-wise feedforward network takes as input the output of the self-attention layer at each position separately, so it operates on each position independently of the others. The feedforward layer consists of two linear transformations with a ReLU activation function in between them:

```
FFN(x) = max(0, xW1 + b1)W2 + b2

```
where x is the input to the feedforward network, W1, b1, W2, and b2 are learnable weight and bias parameters.

The role of the feedforward neural network is to add non-linearity to the transformer architecture and to provide the model with the capability to capture more complex patterns in the input sequence. The feedforward layer helps the model to better model the interactions between the input tokens, which helps to improve the model's overall performance on various NLP tasks.

In summary, the feedforward neural network is an essential component of the transformer architecture, providing non-linearity and complexity to the model, and is applied after the self-attention layer in both the encoder and decoder.


## How feedforward neural network is used in transformer architecture

In the Transformer architecture, the feedforward neural network is used as a component in the encoder and decoder layers.

The feedforward neural network consists of two linear transformations separated by a non-linear activation function such as ReLU (Rectified Linear Unit). The feedforward neural network takes as input the output from the multi-head attention mechanism and passes it through two fully connected layers, each with a dropout layer and a ReLU activation function. The dropout layers are used to prevent overfitting by randomly dropping out a certain percentage of the neurons during training.

The output of the feedforward neural network is then added to the input (element-wise addition) using a residual connection. This means that the output of the feedforward neural network is added to the input of the layer before it passes through the next layer.

The use of residual connections in combination with feedforward neural networks in the Transformer architecture allows the network to better capture long-range dependencies and improve the gradient flow during training, resulting in faster convergence and better performance.

Here is an example code snippet in PyTorch that implements the feedforward neural network in the Transformer architecture:

```
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

```

In the above code, `d_model` refers to the dimensionality of the input, and `d_ff` refers to the dimensionality of the hidden layer in the feedforward neural network. The dropout parameter specifies the dropout rate for the dropout layer.

The `forward` method takes as input the output from the multi-head attention mechanism and passes it through two linear layers with a ReLU activation function in between. The output is then passed through a dropout layer and added to the input using a residual connection.

## What are the activation functions in feedforward neural network

In the feedforward neural network of the transformer architecture, the activation function used is the Rectified Linear Unit (ReLU) function. ReLU is a piecewise linear function that returns the input if it is positive and zero otherwise. Mathematically, ReLU can be defined as:

```
f(x) = max(0, x)
```

where x is the input to the function and f(x) is the output.

Another activation function used in feedforward neural networks is the sigmoid function. The sigmoid function is a S-shaped curve that maps any input to a value between 0 and 1. Mathematically, the sigmoid function can be defined as:

```
f(x) = 1 / (1 + exp(-x))
```

where x is the input to the function and f(x) is the output.

## What is the role of feedforward neural network in transformer model

In the transformer model, the feedforward neural network (FFN) is used to perform a nonlinear transformation of the output of the multi-head attention layer. The FFN consists of two linear transformations with a ReLU activation function in between. The role of the FFN is to add a new layer of abstraction to the input sequence, allowing the model to learn more complex patterns and relationships.

Specifically, after the multi-head attention layer, the output is a sequence of vectors that represents the input sequence with respect to the attention function. The FFN takes this sequence of vectors as input and applies a nonlinear transformation to each vector independently. The output of the FFN is another sequence of vectors that have been transformed to capture additional patterns and features in the input sequence.

The FFN can be represented mathematically as:

```
FFN(x) = relu(xW1 + b1)W2 + b2
```

where x is the input sequence of vectors, W1 and W2 are weight matrices, b1 and b2 are bias vectors, and relu is the rectified linear unit activation function. The output of the FFN is a sequence of vectors that have been transformed to capture additional patterns and features in the input sequence. The use of the FFN helps the transformer model to achieve state-of-the-art performance on a wide range of natural language processing tasks.



