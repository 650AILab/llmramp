# Transformer Architecture : Chapter 03 - Transformer Model Architecture #

In this section you will learn about basics of Transformer Model Architecture and various internal components associated with it, what they are and how they work together to make the transformer working.

## Transformer Model Architecture ##

The Transformer model architecture is a neural network architecture introduced in the paper "Attention Is All You Need" by Vaswani et al. in 2017. It is a sequence-to-sequence (seq2seq) model that is designed for natural language processing (NLP) tasks such as machine translation, language modeling, and text generation.

## Basic architecture of the transformer model ##

The Transformer model architecture consists of an encoder and a decoder, each of which is made up of multiple layers of self-attention and feedforward neural networks. Here is a brief overview of each component:

1. `Input Embeddings`: The input to the Transformer model is a sequence of token embeddings, which are typically learned through an embedding layer that maps each token to a high-dimensional vector.

2. `Encoder`: The encoder consists of a stack of N identical layers, each of which has two sublayers. The first sublayer is a self-attention mechanism, which allows the encoder to weigh different parts of the input sequence based on their relevance to each other. The second sublayer is a feedforward neural network, which applies a non-linear transformation to each position in the sequence independently.

3. `Decoder`: The decoder also consists of a stack of N identical layers, each of which has three sublayers. The first sublayer is a masked self-attention mechanism and the second sublayer is an encoder-decoder attention mechanism and finally the third sublayer is a feedforward neural network

- A masked self-attention mechanism, which allows each position in the output sequence to attend to all previous positions in the output sequence and compute a weighted sum of the embeddings at all positions.
- An encoder-decoder attention mechanism, which allows each position in the output sequence to attend to all positions in the input sequence and compute a weighted sum of the encoder's hidden representations.
- A position-wise feedforward neural network, which applies a non-linear transformation to each position in the sequence independently.

4. `Output Layer`: The output layer is a linear transformation followed by a softmax activation, which converts the decoder's output into a probability distribution over the vocabulary of the target language.

During training, the Transformer model is optimized to minimize the cross-entropy loss between the predicted and actual target sequences. During inference, the model generates the target sequence one token at a time by recursively applying the decoder's sublayers until an end-of-sequence token is generated.

The Transformer model's key innovation is the use of self-attention mechanisms, which allow it to capture long-range dependencies in the input sequence without requiring recurrent connections or convolutional operations. This makes it particularly effective for NLP tasks where long-term context is important, such as machine translation and language modeling.

Overall, the Transformer model architecture is highly effective at modeling long-range dependencies in natural language sequences, and has achieved state-of-the-art performance on a wide range of NLP tasks.


## Encoder-decoder structure

The Encoder-Decoder structure is a neural network architecture that is commonly used in sequence-to-sequence (Seq2Seq) modeling tasks, such as machine translation and speech recognition. The basic idea is to use one neural network (the encoder) to encode the input sequence into a fixed-length representation, and then use another neural network (the decoder) to decode this representation into the output sequence.

Here is a detailed overview of the Encoder-Decoder architecture:

1. `Encoder`: The encoder is a neural network that takes a variable-length input sequence and produces a fixed-length representation of the input. The encoder can be a simple recurrent neural network (RNN), such as a long short-term memory (LSTM) or gated recurrent unit (GRU), or it can be a more complex neural network, such as a convolutional neural network (CNN) or a Transformer model. The encoder typically uses a form of attention mechanism to focus on the most relevant parts of the input sequence.

2. `Decoder`: The decoder is a neural network that takes the fixed-length representation produced by the encoder and generates the output sequence. The decoder can also be a simple RNN, such as an LSTM or GRU, or it can be a more complex neural network, such as a Transformer model. The decoder typically uses a form of attention mechanism to focus on the most relevant parts of the input sequence.

3. `Training`: During training, the Encoder-Decoder model is optimized to minimize the cross-entropy loss between the predicted and actual output sequences. The model is trained using teacher forcing, which means that during training, the decoder is provided with the correct output sequence as input at each time step, rather than using the output sequence generated by the model at the previous time step.

4. `Inference`: During inference, the Encoder-Decoder model is used to generate the output sequence one token at a time. At each time step, the decoder generates a probability distribution over the target vocabulary, and the token with the highest probability is selected as the output. The output is then fed back into the decoder as input at the next time step, and the process is repeated until an end-of-sequence token is generated.

The Encoder-Decoder structure has been successfully applied to a wide range of Seq2Seq modeling tasks, such as machine translation, text summarization, and speech recognition. Its success is largely due to its ability to capture long-term dependencies in the input sequence and produce accurate and fluent output sequences.


## Multi-head attention mechanism

Multi-head attention is a variant of the self-attention mechanism used in the Transformer architecture. It allows the model to jointly attend to different parts of the input sequence at different positions, which enables it to capture more complex relationships between the input and output.


## Feedforward neural network

In the transformer architecture, the feedforward neural network is used as a part of the position-wise fully connected layer, which is applied after the self-attention layer in both the encoder and decoder.

The position-wise feedforward network takes as input the output of the self-attention layer at each position separately, so it operates on each position independently of the others. The feedforward layer consists of two linear transformations with a ReLU activation function in between them. 

The role of the feedforward neural network is to add non-linearity to the transformer architecture and to provide the model with the capability to capture more complex patterns in the input sequence. The feedforward layer helps the model to better model the interactions between the input tokens, which helps to improve the model's overall performance on various NLP tasks.

## Residual connections

In the transformer architecture, residual connections are used to address the vanishing gradient problem that can occur in deep neural networks. The vanishing gradient problem arises when the gradients that are propagated back through the network during training become very small, making it difficult for the network to learn effectively.

## Layer normalization

Layer normalization is a technique used in the transformer architecture to normalize the activations in each layer of the network. The normalization helps to stabilize the training process and improve the performance of the network.

## Positional encoding

The role of positional encoding in the transformer architecture is to provide the network with information about the position of the tokens in the input sequence. Unlike traditional recurrent neural networks that use sequential processing, the transformer architecture uses self-attention mechanisms to process the entire input sequence in parallel. This means that the network has no inherent notion of the order of the tokens in the sequence.
