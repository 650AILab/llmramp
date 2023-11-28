# Transformer Architecture : Chapter 08 - Positional Encoding #

In this section you will learn about positional encoding, its role and the math behind it in transformer.

## What is positional encoding?

The role of positional encoding in the transformer architecture is to provide the network with information about the position of the tokens in the input sequence. Unlike traditional recurrent neural networks that use sequential processing, the transformer architecture uses self-attention mechanisms to process the entire input sequence in parallel. This means that the network has no inherent notion of the order of the tokens in the sequence.

To address this issue, positional encoding is used to inject positional information into the input embeddings. Specifically, positional encoding adds a vector to the embedding vector of each token that encodes its position in the sequence. The positional encoding vectors are learned during training, and they are added to the input embeddings before they are passed to the encoder.

The positional encoding vectors are designed so that they do not interfere with the semantic information in the input embeddings. They are typically computed using a trigonometric function that allows the network to distinguish between tokens based on their position.

The role of positional encoding in the transformer architecture is to provide the network with information about the position of the tokens in the input sequence. This helps the network to understand the order of the tokens and process them correctly. Without positional encoding, the network would be unable to distinguish between tokens based on their position in the sequence, which would make it difficult for the network to learn long-range dependencies and produce accurate output.

In summary, the role of positional encoding in the transformer architecture is to provide the network with information about the position of the tokens in the input sequence. This helps the network to process the tokens correctly and learn long-range dependencies, which is important for many NLP tasks.


## Why it is used in transformer architecture?

Positional encoding is used in the transformer architecture to provide the model with information about the relative positions of the tokens in the input sequence. Unlike recurrent neural networks (RNNs), which process input sequences one token at a time and maintain an internal state that reflects the order of the tokens, the transformer processes all tokens in the sequence in parallel and does not have an inherent notion of position or order.

To address this issue, the transformer uses positional encoding to provide the model with a representation of the position of each token in the sequence. The positional encoding is added to the input embeddings for each token, which allows the model to distinguish between tokens based on their position in the sequence. This helps the model to learn long-range dependencies and capture important patterns in the input sequence, which is particularly important for tasks like language translation and natural language processing.

Without positional encoding, the transformer would be less effective at modeling sequential data and would likely perform poorly on tasks that require understanding of the order of the input sequence.

## How positional encoding is calculated in Transformer?

The positional encoding in the transformer architecture is calculated using a mathematical function that generates a unique encoding vector for each position in the input sequence. Specifically, the formula for computing the positional encoding vector for a given position pos and embedding dimension i is:

```
PE(pos, 2i) = sin(pos / (10000 ^ (2i / d_model)))
PE(pos, 2i+1) = cos(pos / (10000 ^ (2i / d_model)))

```
Here, PE(pos, 2i) and PE(pos, 2i+1) denote the values of the positional encoding vector at the pos-th position and the 2i-th and (2i+1)-th dimensions, respectively. d_model is the dimensionality of the input embeddings.

The formula is based on a combination of sine and cosine functions that vary sinusoidally with the position pos and the embedding dimension i. The factor (10000 ^ (2i / d_model)) is used to scale the frequencies of the sine and cosine functions for each dimension, so that each dimension has a unique encoding pattern. The even and odd dimensions are encoded using sine and cosine functions, respectively, so that adjacent positions in the sequence have different encoding vectors and can be distinguished by the model.

The positional encoding vectors are added element-wise to the input embeddings for each token, providing the model with information about the position of each token in the sequence. This allows the model to distinguish between tokens based on their position and capture long-range dependencies in the input sequence.

## Advantages of using positional encoding ?

The use of positional encoding in the transformer architecture provides several advantages:

1. Captures sequence order: The positional encoding allows the transformer to capture the order of tokens in the input sequence, which is essential for natural language processing tasks like translation and text generation. By encoding the position of each token, the model can distinguish between tokens based on their position and capture long-range dependencies in the sequence.

2. Parallel processing: The positional encoding allows the transformer to process all tokens in the sequence in parallel, unlike RNNs, which process tokens one at a time. This parallel processing is much more efficient, making the transformer faster and more scalable than traditional RNN models.

3. Generalization: The positional encoding can be applied to input sequences of varying lengths, allowing the model to generalize to new inputs of different lengths. This is particularly important for natural language processing tasks, where input sequences can vary widely in length.

4. Low computational cost: The positional encoding is simple and computationally inexpensive to calculate, making it a lightweight addition to the model that does not significantly increase the computational cost of training or inference.

Overall, the use of positional encoding in the transformer architecture is a key factor in its success in natural language processing tasks, allowing the model to effectively capture the structure and order of input sequences and achieve state-of-the-art performance on a wide range of benchmarks.

## Python Code Example

Sure, here's an example Python code that demonstrates how positional encoding is computed in the transformer architecture:

```
import torch

# Define the maximum sequence length and embedding dimension
max_seq_len = 10
embed_dim = 16

# Define the position encodings
position_encodings = torch.zeros(max_seq_len, embed_dim)

# Compute the positional encoding vectors
for pos in range(max_seq_len):
    for i in range(embed_dim):
        if i % 2 == 0:
            position_encodings[pos, i] = torch.sin(pos / (10000 ** (i / embed_dim)))
        else:
            position_encodings[pos, i] = torch.cos(pos / (10000 ** ((i - 1) / embed_dim)))

# Print the position encodings for the first 5 positions
print(position_encodings[:5])

```

In this code, we first define the maximum sequence length and embedding dimension, which are both hyperparameters of the transformer architecture. We then create a tensor to hold the position encodings, which has shape (max_seq_len, embed_dim).

Next, we loop over each position in the sequence and each dimension in the embedding vector. We use a trigonometric function to compute the value of each element in the position encoding vector, based on the position and dimension. Specifically, we use a sine function for even dimensions and a cosine function for odd dimensions, with a scaling factor that decreases exponentially with the dimension number.

Finally, we print the position encodings for the first 5 positions in the sequence. Note that the position encodings are added to the input embeddings before they are passed to the encoder in the transformer architecture. This helps to provide the network with information about the position of the tokens in the sequence, which is important for learning long-range dependencies and producing accurate output.