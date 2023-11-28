# Transformer Architecture : Chapter 04 - Multi-Head Attention Mechanism #

In this section you will learn about Multi-Head Attention Mechanism, how does it work, how to calculate and implement it inside the transformers.

## What is multi-head attention?

Multi-head attention is a variant of the self-attention mechanism used in the Transformer model. In multi-head attention, instead of using a single attention function to compute attention weights for the input sequence, multiple attention functions are used in parallel. Each attention function is called a "head". The outputs of these attention heads are concatenated and projected to obtain the final output of the multi-head attention layer.

The idea behind multi-head attention is to allow the model to jointly attend to different parts of the input sequence, using different attention functions. This can be useful for capturing different types of dependencies and patterns in the input data. Multi-head attention also allows the model to scale to longer sequences, since it can attend to different parts of the input in parallel.

In the Transformer model, multi-head attention is used in both the encoder and decoder layers. In the encoder, multi-head attention is used to compute attention weights for the input sequence, while in the decoder, it is used to compute attention weights for the encoder output sequence and the decoder input sequence.

Here's how multi-head attention works in the Transformer architecture:

1. `Input representation`: The input to the multi-head attention mechanism is a sequence of vectors, where each vector represents a word or a subword token in the input sequence. Each vector is first transformed into three vectors of equal size, one for each of the query, key, and value projections.

2. `Splitting into heads`: The multi-head attention mechanism splits the transformed input into multiple heads, typically 8 or 16. Each head is a separate attention mechanism that learns to attend to different parts of the input sequence.

3. `Attention computation`: For each head, the attention mechanism computes an attention score between the query and key vectors, which is then used to compute a weighted sum of the value vectors. The attention scores are computed using the dot product of the query and key vectors, followed by a softmax operation to obtain a probability distribution over the keys.

4. `Concatenation`: Once the attention scores have been computed for each head, the resulting weighted value vectors are concatenated into a single vector, which is then passed through a linear layer to produce the final output vector.


## How multi-head attention works in transformer architecture?

In the Transformer architecture, multi-head attention works by computing multiple attention heads in parallel, each with a different set of learned parameters. The computation can be broken down into three main steps:

1. Linearly projecting the input sequence into a set of query, key, and value matrices. This is done separately for each attention head, using different learned projection matrices.

2. Computing the attention weights for each head by taking the dot product of the query matrix with the key matrix. This produces a set of attention scores, which are then normalized using a softmax function to obtain the attention weights.

3. Computing the weighted sum of the value matrix, using the attention weights as weights. This produces a set of context vectors, one for each attention head.

The context vectors from each attention head are concatenated along the feature dimension, and then passed through a linear projection to obtain the final output of the multi-head attention layer.

The advantage of using multi-head attention is that it allows the model to jointly attend to different parts of the input sequence using different sets of parameters, which can help capture more complex relationships between the input tokens. Additionally, it allows the model to scale to longer sequences, since attention can be computed in parallel for different parts of the input.

## How multi-head attention is calculated in transformer?

In transformer architecture, multi-head attention is calculated using the following steps:

1. First, the input sequence is transformed into queries, keys, and values using linear projections. These projections are different for each head of the attention mechanism.
2. Next, the queries are matched against the keys using dot product attention, which results in an attention weight for each key. These attention weights determine how much each value contributes to the final output.
3. The attention weights are then normalized using the softmax function to ensure that they add up to 1.
4. The values are then multiplied by their corresponding attention weights and summed to produce a weighted sum, which is the output of the attention mechanism for that head.
5. Finally, the outputs of all the heads are concatenated and passed through another linear layer to produce the final output of the multi-head attention mechanism.

The calculation of multi-head attention can be expressed mathematically as follows:

```
$$\text{MultiHead}(Q,K,V) = \text{Concat}(head_1, head_2, ..., head_h)W^O$$
```

where $Q$, $K$, and $V$ are the input queries, keys, and values, respectively; $head_i = \text{Attention}(QW_i^Q,KW_i^K,VW_i^V)$ is the output of the $i$-th attention head; $W_i^Q$, $W_i^K$, and $W_i^V$ are the weight matrices for the linear projections of the queries, keys, and values for the $i$-th head; $\text{Attention}$ is the dot product attention function; and $W^O$ is the weight matrix for the final linear layer. The $\text{Concat}$ function concatenates the outputs of all the attention heads along the last dimension.

## What is the role of intuition behind multi-head attention?

The intuition behind multi-head attention is to enable the model to focus on different parts of the input sequence, as different parts of the input may be more important for different output tokens. By computing multiple attention heads in parallel, each with a different set of learned parameters, the model can learn to attend to different information in the input sequence and then combine this information in a meaningful way. 

This can help the model to better capture complex dependencies between different parts of the input and output sequences. Additionally, using multiple attention heads can help to regularize the model and improve its generalization performance.

## How intution is calculated in multi-head attention?

The intuition behind multi-head attention can be understood through an example. Let's say we have a machine translation task where we want to translate a sentence from English to French. The input sentence in English may have different words or phrases that have different levels of importance in the translation process. For example, the verb tense or subject pronoun may be critical for correctly translating the sentence, while certain adjectives or adverbs may have less impact on the overall meaning.

In a traditional attention mechanism, the model would apply the same attention mechanism to all parts of the input sentence to compute the context vector for each output token. However, this approach may not be optimal for capturing the complex dependencies between different parts of the input and output sequences.

In contrast, in multi-head attention, the model computes multiple attention mechanisms in parallel, each with a different set of learned parameters. Each attention head can focus on a different aspect of the input sentence, allowing the model to capture different aspects of the input and output dependencies. By combining the output of multiple attention heads, the model can better capture the complex interactions between different parts of the input and output sequences.

Overall, the intuition behind multi-head attention is to provide the model with greater flexibility and the ability to capture complex dependencies between different parts of the input and output sequences.

