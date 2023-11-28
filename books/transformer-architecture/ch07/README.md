# Transformer Architecture : Chapter 07 - Layer Normalization #

In this section you will learn about layer normalization, why it is used and how does it works inside the transformer.

## What is layer normalization?

Layer normalization is a technique used in the transformer architecture to normalize the activations in each layer of the network. The normalization helps to stabilize the training process and improve the performance of the network.

In the transformer architecture, layer normalization is applied after each sub-layer in both the encoder and decoder. Specifically, the normalization is applied to the output of the sub-layer before it is passed to the next sub-layer. The normalization is performed independently for each feature dimension of the input tensor, so it has no effect on the correlations between features.

The role of layer normalization in the transformer architecture is to make the distribution of activations more stable across the features, which helps to improve the convergence of the network during training. The normalization also helps to reduce the sensitivity of the network to the scale of the features, making it more robust to variations in the input.

In summary, the role of layer normalization in the transformer architecture is to improve the stability and performance of the network by normalizing the activations in each layer. This technique helps to make the training process more efficient and robust, which can lead to better results on a variety of NLP tasks.


## Why it is used in transformer architecture ?

Layer normalization is used in the transformer architecture for several reasons:

1. `Improved convergence`: Layer normalization helps to improve the convergence of the model during training by reducing the internal covariate shift, which is the change in the distribution of the input to a layer that occurs during training. This allows the model to learn more efficiently and converge faster.

2. `Stable gradients`: Layer normalization helps to stabilize the gradients during training, making it easier to train deep neural networks with many layers.

3. `Regularization`: Layer normalization acts as a form of regularization, helping to prevent overfitting of the model to the training data.

4. `Robustness`: Layer normalization helps to make the model more robust to changes in the input distribution, making it more effective at generalizing to new data.

Overall, layer normalization is an important component of the transformer architecture, helping to improve the stability, convergence, and generalization of the model during training and inference.

## How layer normalization works in transformer?

In the transformer architecture, layer normalization is applied to the output of each sublayer in the encoder and decoder stacks. The purpose of layer normalization is to normalize the activations of each layer to have zero mean and unit variance, which helps to stabilize the training process and improve the performance of the model.

The formula for layer normalization is:

```
LN(x) = a * (x - μ) / √(σ^2 + ε) + b

```

where x is the input to the layer, μ and σ are the mean and standard deviation of the input, respectively, ε is a small constant to avoid division by zero, and a and b are learnable parameters that scale and shift the normalized output.

The layer normalization formula is applied element-wise, meaning that each element in the input vector is normalized independently. This differs from batch normalization, which normalizes the activations over the entire batch of inputs.

Layer normalization is typically applied after the self-attention and feedforward sublayers in each transformer block. After applying layer normalization, the output is passed through a residual connection, which allows the original input to be added back to the normalized output. This helps to ensure that the model can learn both low-level and high-level features, allowing it to capture a wide range of patterns in the input data.

## What are the advantages of using layer normalization in transformer ?

There are several advantages of using layer normalization in the transformer architecture:

1. `Improved convergence`: Layer normalization helps to improve the convergence of the model during training by reducing the internal covariate shift, which is the change in the distribution of the input to a layer that occurs during training. This allows the model to learn more efficiently and converge faster.

2. `Stable gradients`: Layer normalization helps to stabilize the gradients during training, making it easier to train deep neural networks with many layers.

3. `Regularization`: Layer normalization acts as a form of regularization, helping to prevent overfitting of the model to the training data.

4. `Robustness`: Layer normalization helps to make the model more robust to changes in the input distribution, making it more effective at generalizing to new data.

5. `Parameter efficiency`: Layer normalization requires fewer parameters than other normalization methods, such as batch normalization, making it more efficient for models with large numbers of parameters.

Overall, layer normalization is an important component of the transformer architecture, helping to improve the stability, convergence, and generalization of the model during training and inference. It is widely used in many state-of-the-art natural language processing models and has been shown to be effective for a wide range of tasks, including machine translation, language modeling, and text classification.

## Python Code Example

Here's an example of how layer normalization can be implemented in PyTorch for the encoder layer of a transformer:

```
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(self.norm1(src2))
        src2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(src))))
        src = src + self.dropout(self.norm2(src2))
        return src

```

In the code above, we define an `TransformerEncoderLayer` class that implements a single encoder layer of the transformer. The layer includes two layer normalization layers (`self.norm1` and `self.norm2`) that are applied after the self-attention and feedforward layers, respectively. These layer normalization layers help to improve the stability and convergence of the model during training by reducing the internal covariate shift and stabilizing the gradients.

During the forward pass, the input `src` is passed through the self-attention layer and the resulting output is passed through the first layer normalization layer (`self.norm1`). The output of the first layer normalization layer is then added back to the original input and passed through a feedforward network with another layer normalization layer (`self.norm2`) applied to the output. The final output of the layer is returned.

