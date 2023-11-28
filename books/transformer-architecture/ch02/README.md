# Transformer Architecture : Chapter 02 #

In this section you will learn about self attention mechanism.

## 2.1 What is attention?

In deep learning, attention is a mechanism that allows a model to selectively focus on different parts of an input sequence when making predictions. The attention mechanism was originally developed for natural language processing (NLP) tasks, but it has since been applied to other domains, such as computer vision and speech recognition.

The basic idea behind attention is to calculate a set of weights that indicate the importance of each element in the input sequence, and then use those weights to compute a weighted sum of the elements. These weights are calculated based on some similarity measure between the current position of the model and the other positions in the input sequence. The attention weights are typically learned during training, and they allow the model to focus on different parts of the input sequence based on the context and task at hand.

One of the advantages of attention is that it allows the model to handle variable-length input sequences, which is often the case in NLP tasks. Instead of treating the entire input sequence as a single vector, the attention mechanism allows the model to focus on different parts of the sequence based on the context.

Overall, attention is a powerful mechanism for selectively focusing on different parts of an input sequence, allowing deep learning models to achieve state-of-the-art performance on many NLP tasks.


## 2.2 Types of attention mechanisms

There are several types of attention mechanisms used in deep learning models, including:

### 1. Global attention: 
Global attention computes a weighted sum of all the input vectors, where the weights are determined based on some similarity measure between the current position of the model and the other positions in the input sequence. This allows the model to take into account all the information in the input sequence, regardless of its length.

### 2. Local attention: 
Local attention only considers a subset of the input vectors when computing the attention weights. This allows the model to focus on a specific part of the input sequence, which can be useful in situations where the relevant information is only located in a small part of the sequence.

### 3. Dot product attention: 
Dot product attention computes the similarity between the query vector and the key vector using the dot product operation. This is a simple and efficient way of computing attention weights, but it can be sensitive to the scale of the input vectors.

### 4. Scaled dot product attention: 
Scaled dot product attention is similar to dot product attention, but it scales the dot product by the square root of the dimensionality of the key vectors. This helps to prevent the dot product from becoming too large or too small, and has been shown to be effective in improving the stability and performance of the attention mechanism.

### 5. Additive attention: 
Additive attention computes the similarity between the query vector and the key vector using a neural network with a single hidden layer. This allows the model to learn a more complex similarity measure than the dot product operation, but it can be more computationally expensive.

### 6. Multiplicative attention: 
Multiplicative attention computes the similarity between the query vector and the key vector by multiplying them element-wise and then passing the result through a neural network. This allows the model to learn a more complex similarity measure than the dot product operation, but it can also be more computationally expensive.

Overall, the choice of attention mechanism depends on the specific task and the characteristics of the input data. In practice, researchers often experiment with different types of attention mechanisms to find the one that works best for their particular application.

## Self-attention mechanism

Self-attention is a mechanism in which an input sequence is processed by computing a weighted sum of its own elements, where the weights are calculated based on the similarity between each pair of elements in the sequence. Self-attention is used in the transformer architecture, a deep learning model that has achieved state-of-the-art results in many natural language processing (NLP) tasks.

In self-attention, the input sequence is transformed into three sets of vectors: the query vectors, the key vectors, and the value vectors. The query vectors are used to determine which elements of the sequence are most relevant for the current position, while the key vectors are used to compute the similarity between each pair of elements in the sequence. The value vectors are used to compute the output of the self-attention mechanism.

To compute the attention weights, the model calculates the dot product between the query vectors and the key vectors for each pair of elements in the sequence. The resulting dot products are then passed through a softmax function to obtain a probability distribution over the elements in the sequence. These probabilities are used to compute a weighted sum of the value vectors, which is used as the output of the self-attention mechanism.

One of the advantages of self-attention is that it allows the model to selectively focus on different parts of the input sequence based on the context, without the need for convolutional or recurrent layers. This makes the model more computationally efficient and easier to parallelize, which is important for training large-scale models.

Self-attention has been shown to be effective in achieving state-of-the-art results on a wide range of NLP tasks, including language modeling, machine translation, and question answering. It has also been applied to other domains, such as computer vision and speech recognition, with promising results.

## 2.3 How self-attention works in transformers

In transformers, self-attention is used to compute representations of the input sequence that capture dependencies between different positions in the sequence. This is achieved through multi-head attention, which combines multiple self-attention mechanisms to capture different relationships between the input tokens.

In the transformer architecture, the input sequence is first embedded into a vector space using an embedding layer. The resulting sequence of embeddings is then fed into multiple layers of transformer blocks, each consisting of a multi-head self-attention mechanism and a position-wise feedforward network.

The self-attention mechanism in transformers works similarly to the one we described earlier. However, instead of computing self-attention for a single input sequence, we compute it for multiple sequences in parallel, where each sequence corresponds to a different attention head. The outputs of each attention head are then concatenated and projected back to the input dimension using a linear layer.

The use of multi-head attention in transformers allows the model to capture different types of dependencies between the input tokens. For example, some attention heads may focus on local dependencies between adjacent tokens, while others may capture longer-range dependencies between distant tokens. By combining the outputs of these different attention heads, the model can capture complex patterns of dependencies between the input tokens, leading to more effective representations of the input sequence.


### Question - How does model handles the variable-length input sequences?

In natural language processing (NLP) tasks, input sequences can vary in length, and handling variable-length input sequences is a challenge for many machine learning models. The transformer architecture uses a mechanism called self-attention to handle variable-length input sequences.

Self-attention allows the transformer to compute a weighted sum of all the input vectors, where the weights are determined based on the similarity between each pair of input vectors. This allows the model to assign higher weights to input vectors that are more relevant to the current position, and lower weights to those that are less relevant. By using self-attention, the transformer is able to selectively focus on different parts of the input sequence, even when the sequence is of variable length.

To compute the attention weights, the transformer uses three types of vectors: the input vectors (also known as embeddings), the query vectors, and the key vectors. The query vectors are computed based on the current position of the model, and the key vectors are computed based on all the input vectors. The dot product of the query vector and the key vector is used as a measure of similarity, and the softmax function is applied to normalize the dot products into a probability distribution. The resulting weights are then used to compute a weighted sum of the input vectors, which is used as the output of the self-attention layer.

By using self-attention, the transformer is able to handle variable-length input sequences without the need for padding or truncation. This allows the model to take into account all the information in the input sequence, regardless of its length, and has been shown to be effective in achieving state-of-the-art performance on many NLP benchmarks.


### Code Example 1

Here is a code example to show how does self attention mechanism works:

```
import torch
import torch.nn.functional as F

# Define the input sequence
x = torch.randn(3, 5, 10)  # batch size = 3, sequence length = 5, embedding dimension = 10

# Define the query, key, and value matrices
W_q = torch.randn(10, 2)  # embedding dimension = 10, attention dimension = 2
W_k = torch.randn(10, 2)  # embedding dimension = 10, attention dimension = 2
W_v = torch.randn(10, 2)  # embedding dimension = 10, attention dimension = 2

# Compute the query, key, and value vectors
q = torch.matmul(x, W_q)
k = torch.matmul(x, W_k)
v = torch.matmul(x, W_v)

# Compute the dot products between the query and key vectors
dot_products = torch.matmul(q, k.transpose(1, 2))  # shape = (3, 5, 5)

# Compute the attention weights
weights = F.softmax(dot_products, dim=-1)

# Apply the attention weights to the value vectors
output = torch.matmul(weights, v)  # shape = (3, 5, 2)

```

In above example, we first define an input sequence x with shape (3, 5, 10), which represents a batch of 3 sequences of length 5, where each element has an embedding dimension of 10.

We then define three weight matrices W_q, W_k, and W_v with shapes (10, 2), which represent the query, key, and value matrices, respectively. We use these matrices to compute the query vectors q, the key vectors k, and the value vectors v.

Next, we compute the dot products between the query and key vectors for each pair of elements in the sequence, which gives us a matrix of shape (3, 5, 5) representing the similarity between each pair of elements.

We then apply the softmax function to the dot products to obtain a probability distribution over the elements in the sequence, which gives us a matrix of shape (3, 5, 5) representing the attention weights.

Finally, we apply the attention weights to the value vectors to obtain the output of the self-attention mechanism, which is a matrix of shape (3, 5, 2) representing the weighted sum of the value vectors for each element in the sequence.


### Code Example 2

Here's an example code snippet that demonstrates how self-attention can be applied to a set of input sentences using PyTorch:


```
import torch
import torch.nn.functional as F

# Define the input sentences
sentences = [
    'The quick brown fox jumps over the lazy dog',
    'She sells seashells by the seashore',
    'Peter Piper picked a peck of pickled peppers',
]

# Tokenize the sentences and convert them to PyTorch tensors
tokens = [sentence.split() for sentence in sentences]
token_ids = [[hash(token) % 1000 for token in sentence] for sentence in tokens]
input_tensor = torch.tensor(token_ids)  # shape = (3, max_length)

# Define the self-attention module
class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.key_proj = torch.nn.Linear(input_dim, input_dim * num_heads)
        self.query_proj = torch.nn.Linear(input_dim, input_dim * num_heads)
        self.value_proj = torch.nn.Linear(input_dim, input_dim * num_heads)
        self.output_proj = torch.nn.Linear(input_dim * num_heads, input_dim)

    def forward(self, x):
        batch_size, max_length, input_dim = x.size()
        num_heads = self.num_heads

        # Project the input tensor into the key, query, and value spaces
        keys = self.key_proj(x)  # shape = (batch_size, max_length, input_dim * num_heads)
        queries = self.query_proj(x)  # shape = (batch_size, max_length, input_dim * num_heads)
        values = self.value_proj(x)  # shape = (batch_size, max_length, input_dim * num_heads)

        # Reshape the key, query, and value tensors for multi-head attention
        keys = keys.view(batch_size, max_length, num_heads, input_dim)  # shape = (batch_size, max_length, num_heads, input_dim)
        queries = queries.view(batch_size, max_length, num_heads, input_dim)  # shape = (batch_size, max_length, num_heads, input_dim)
        values = values.view(batch_size, max_length, num_heads, input_dim)  # shape = (batch_size, max_length, num_heads, input_dim)

        # Compute the dot products between the queries and keys for each head
        dot_products = torch.einsum('bjhd,bkhd->bhjk', queries, keys)  # shape = (batch_size, num_heads, max_length, max_length)

        # Scale the dot products by the square root of the input dimension
        scaled_dot_products = dot_products / (input_dim ** 0.5)

        # Apply the softmax function to obtain the attention weights
        attention_weights = F.softmax(scaled_dot_products, dim=-1)

        # Compute the weighted sum of the values for each head
        weighted_sum = torch.einsum('bhjk,bjhd->bkhd', attention_weights, values)  # shape = (batch_size, max_length, num_heads, input_dim)

        # Concatenate the output of each head and project it back to the original input dimension
        concatenated = weighted_sum.view(batch_size, max_length, -1)  # shape = (batch_size, max_length, input_dim * num_heads)
        output = self.output_proj(concatenated)  # shape = (batch_size, max_length, input_dim)

        return output

# Apply self-attention to the input tensor

self_attention = SelfAttention(input_dim=32, num_heads=4)
output = self_attention(input_tensor)
print(output.shape) # should print torch.Size([3, max_length, input_dim])

```

In this example: 
- First we define a `SelfAttention` module 
  - that takes an input tensor of shape `(batch_size, max_length, input_dim)` and applies self-attention to it. 
- The `input_dim` parameter specifies the dimensionality of each token embedding, while the `num_heads` parameter specifies the number of attention heads to use. 
- We then create an instance of the `SelfAttention` module with an `input_dim` of 32 and a `num_heads` of 4, and apply it to an input tensor of shape `(3, max_length, 32)`, where `max_length` is the maximum length of the input sequence and `3` is the batch size.

- Inside the forward method of the SelfAttention module, we first project the input tensor into the key, query, and value spaces using three separate linear layers. We then reshape the resulting key, query, and value tensors into (batch_size, max_length, num_heads, input_dim) to prepare them for multi-head attention.

- Next, we compute the dot products between the queries and keys for each attention head using torch.einsum. We scale the dot products by the square root of the input dimension and apply the softmax function to obtain the attention weights. Finally, we compute the weighted sum of the values for each head, concatenate the output of each head, and project it back to the original input dimension using another linear layer.

In this way, the self-attention mechanism allows the model to attend to different parts of the input sequence with different weights, depending on the relevance of each part to the task at hand.


