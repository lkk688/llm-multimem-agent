# Transformers
## Evolution of Sequence Models: From RNNs to Transformers

### RNNs with Attention

**Reference Links:**

- Paper: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

**Motivation:** Traditional RNNs struggled with long-range dependencies due to the vanishing gradient problem.

**Problem:** Sequential processing in RNNs created bottlenecks for parallelization and limited the model's ability to capture relationships between distant tokens.

**Solution:** Attention mechanisms allowed models to focus on relevant parts of the input sequence when generating each output token, creating direct connections between states regardless of their distance.

```python
# Simplified Attention Mechanism in RNNs
def attention(query, key_values):
    # query: current decoder state
    # key_values: encoder states
    scores = dot_product(query, key_values)  # Compute alignment scores
    weights = softmax(scores)  # Normalize to get attention weights
    context = weighted_sum(weights, key_values)  # Create context vector
    return context
```

**Mathematical Formulation:**

$$
\begin{align}
\text{score}(q, k_i) &= q^T k_i \\
\alpha_i &= \frac{\exp(\text{score}(q, k_i))}{\sum_j \exp(\text{score}(q, k_j))} \\
\text{context} &= \sum_i \alpha_i v_i
\end{align}
$$

where $q$ is the query vector, $k_i$ are key vectors, $\alpha_i$ are attention weights, and $v_i$ are value vectors.

**Popularity:** While largely superseded by Transformers, attention-augmented RNNs were a critical stepping stone in the evolution of sequence models.

**Models/Frameworks:** Early NMT systems, GNMT (Google Neural Machine Translation)

### The Transformer Revolution

**Reference Links:**

- Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

**Motivation:** Eliminate sequential computation to enable more parallelization and better capture long-range dependencies.

**Problem:** RNNs processed tokens sequentially, creating a computational bottleneck and making it difficult to capture relationships between distant tokens.

**Solution:** Replace recurrence entirely with self-attention mechanisms that directly model relationships between all tokens in a sequence, regardless of their distance.

**Self-attention:**

- Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- GitHub: [huggingface/transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py)
- **Motivation:** Enable direct modeling of relationships between any two positions in a sequence.
- **Problem:** Traditional sequence models struggled to capture long-range dependencies efficiently.
- **Solution:** Self-attention computes attention weights between all pairs of tokens in a sequence, allowing each token to attend to all other tokens directly.

```python
# Simplified Self-Attention
def self_attention(X, mask=None):
    # X: input sequence [batch_size, seq_len, d_model]
    Q = X @ W_q  # Query projection
    K = X @ W_k  # Key projection
    V = X @ W_v  # Value projection
    
    # Scaled dot-product attention
    scores = (Q @ K.transpose(-2, -1)) / sqrt(d_k)  # [batch_size, seq_len, seq_len]
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    weights = softmax(scores, dim=-1)  # Attention weights
    output = weights @ V  # Weighted aggregation of values
    return output
```

- **Mathematical Formulation:**

$$
\begin{align}
Q &= XW^Q \\
K &= XW^K \\
V &= XW^V \\
\text{Attention}(Q, K, V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\end{align}
$$

where $X$ is the input sequence, $W^Q$, $W^K$, and $W^V$ are learnable parameter matrices, and $d_k$ is the dimension of the key vectors.

- **Popularity:** Self-attention is the fundamental building block of all modern Transformer-based LLMs.

- **Models/Frameworks:** All modern LLMs (GPT, BERT, T5, Llama, etc.)


**Multi-Head Attention:**

- Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- GitHub: [huggingface/transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py)
- **Motivation:** Allow the model to jointly attend to information from different representation subspaces.
- **Problem:** A single attention mechanism might focus too narrowly on specific patterns.
- **Solution:** Run multiple attention operations in parallel with different learned projections, then concatenate and linearly transform the results.

```python
# Simplified Multi-Head Attention
def multi_head_attention(X, mask=None, num_heads=8):
    # Split embedding dimension into heads
    head_dim = d_model // num_heads
    
    # Linear projections and split into heads
    Q = X @ W_q  # [batch_size, seq_len, d_model]
    K = X @ W_k
    V = X @ W_v
    
    # Reshape for multi-head attention
    Q = Q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Scaled dot-product attention for each head
    scores = (Q @ K.transpose(-2, -1)) / sqrt(head_dim)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = softmax(scores, dim=-1)
    attention = weights @ V  # [batch_size, num_heads, seq_len, head_dim]
    
    # Reshape back and project
    attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    output = attention @ W_o  # Final linear projection
    return output
```

- **Mathematical Formulation:**

$$
\begin{align}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)W^O \\
\text{where} \quad \text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{align}
$$
where $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$, and $W^O \in \mathbb{R}^{hd_v \times d_{model}}$ are learnable parameter matrices.

- **Popularity:** Multi-head attention is a standard component in all Transformer-based models.

- **Models/Frameworks:** All modern LLMs (GPT, BERT, T5, Llama, etc.)

**Feed-Forward Networks (FFN):**

- Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- GitHub: [huggingface/transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py)
- **Motivation:** Introduce non-linearity and increase the model's representational capacity.
- **Problem:** Attention mechanisms alone provide only linear transformations of the input.
- **Solution:** Apply a position-wise feed-forward network consisting of two linear transformations with a non-linear activation in between.

```python
# Position-wise Feed-Forward Network
def feed_forward(X):
    # X: [batch_size, seq_len, d_model]
    hidden = X @ W_1 + b_1  # First linear layer
    hidden = relu(hidden)   # Non-linear activation
    output = hidden @ W_2 + b_2  # Second linear layer
    return output
```

- **Mathematical Formulation:**

$$ \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 $$

where $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$, $b_1 \in \mathbb{R}^{d_{ff}}$, and $b_2 \in \mathbb{R}^{d_{model}}$ are learnable parameters.

- **Popularity:** Standard component in all Transformer architectures.

- **Models/Frameworks:** All modern LLMs

**Layer Normalization:**

- Paper: [Layer Normalization](https://arxiv.org/abs/1607.06450)
- GitHub: [pytorch/pytorch](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/normalization.py)
- **Motivation:** Stabilize and accelerate training by normalizing activations.
- **Problem:** Deep neural networks suffer from internal covariate shift, making training unstable and slower.
- **Solution:** Normalize the activations of each layer for each training example independently, making training more stable and faster.

```python
# Layer Normalization
def layer_norm(X, gamma, beta, eps=1e-5):
    # X: [batch_size, seq_len, d_model]
    mean = X.mean(dim=-1, keepdim=True)
    var = ((X - mean) ** 2).mean(dim=-1, keepdim=True)
    X_norm = (X - mean) / torch.sqrt(var + eps)
    return gamma * X_norm + beta  # Scale and shift with learnable parameters
```

- **Mathematical Formulation:**

$$
\begin{align}
\mu &= \frac{1}{H} \sum_{i=1}^{H} x_i \\
\sigma^2 &= \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2 \\
\text{LayerNorm}(x) &= \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\end{align}
$$

where $H$ is the hidden dimension size, $\gamma$ and $\beta$ are learnable scale and shift parameters, and $\epsilon$ is a small constant for numerical stability.

- **Popularity:** Layer normalization is used in virtually all modern Transformer architectures.

**Models/Frameworks:** All modern LLMs


**Residual Connections:**

- Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Motivation:** Enable training of very deep networks by addressing the vanishing gradient problem.
- **Problem:** Deep networks become increasingly difficult to train due to vanishing gradients.
- **Solution:** Add skip connections that bypass certain layers, allowing gradients to flow more easily through the network.

```python
# Residual Connection
def residual_connection(X, sublayer):
    return X + sublayer(X)  # Add input to the output of sublayer
```

- **Mathematical Formulation:**

$$ \text{ResidualConnection}(X, \text{sublayer}) = X + \text{sublayer}(X) $$

where $\text{sublayer}$ is a function representing a transformer sublayer (attention or feed-forward network).

- **Popularity:** Residual connections are a standard component in all deep neural networks, including Transformers.

- **Models/Frameworks:** All modern LLMs

**Positional Encodings:**

- Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- GitHub: [huggingface/transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py)
- **Motivation:** Provide information about token positions in the sequence.
- **Problem:** Self-attention is permutation-invariant and doesn't inherently capture sequence order.
- **Solution:** Add positional encodings to token embeddings to inject information about token positions.

```python
# Sinusoidal Positional Encoding
def positional_encoding(seq_len, d_model):
    positions = torch.arange(seq_len).unsqueeze(1)  # [seq_len, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    
    pos_enc = torch.zeros(seq_len, d_model)
    pos_enc[:, 0::2] = torch.sin(positions * div_term)  # Even dimensions
    pos_enc[:, 1::2] = torch.cos(positions * div_term)  # Odd dimensions
    
    return pos_enc  # [seq_len, d_model]
```

- **Mathematical Formulation:**

$$
\begin{align}
\text{PE}_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right) \\
\text{PE}_{(pos, 2i + 1)} &= \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
\end{align}
$$
where $pos$ is the position index, $i$ is the dimension index, and $d_{model}$ is the embedding dimension.

- **Popularity:** While the original sinusoidal encodings have been largely replaced by learned positional embeddings or RoPE in modern LLMs, some form of positional encoding is essential in all Transformer models.

- **Models/Frameworks:** All Transformer-based models



### Transformer Architecture

Transformers are flexible architectures that fall into three broad categories:

- **Encoder-only models** — e.g., BERT, RoBERTa
- **Decoder-only models** — e.g., GPT, LLaMA
- **Encoder-Decoder (seq2seq) models** — e.g., T5, BART, Whisper

Each architecture is optimized for different tasks: classification, generation, or both.

#### 🧠 Encoder-Only Models

These models use only the encoder stack of the Transformer.

**Use Cases:** Text classification, QA, sentence embeddings, token classification.

**Key Models and Variants:**
- **BERT** — bidirectional masked language model
- **RoBERTa** — BERT with dynamic masking and more data
- **DistilBERT** — lighter BERT via distillation
- **ELECTRA** — replaces MLM with replaced-token detection

**Architectural Modifications:**
- Positional embeddings (learned vs. sinusoidal)
- Token masking (MLM-style)
- Output from `[CLS]` token


#### 🧠 Decoder-Only Models

These use only the decoder stack with **causal masking** to prevent access to future tokens.

**Use Cases:** Text generation, code completion, chatbots, LLMs.

**Key Models and Variants:**
- **GPT-2/3/4** — autoregressive causal decoder
- **LLaMA** — efficient decoder for LLM research
- **Mistral** — sliding-window attention
- **Phi-2** — small LLM trained with curriculum

**Architectural Modifications:**
- No encoder
- Causal self-attention only
- LayerNorm placement varies across versions


#### 🧠 Encoder-Decoder Models

These use both an encoder and a decoder, with **cross-attention** from decoder to encoder output.

**Use Cases:** Translation, summarization, speech-to-text.

**Key Models and Variants:**
- **T5** — unified text-to-text transformer
- **BART** — denoising autoencoder for seq2seq
- **Whisper** — speech-to-text with audio encoder

**Motivation:** Combine parallel processing (encoder) with autoregressive generation (decoder).

**Problem Solved:** Unified, end-to-end trainable architecture for sequence transduction.


#### 🔁 Architectural Comparison

| Architecture     | Self-Attention Type | Cross-Attention | Typical Tasks              |
|------------------|---------------------|------------------|----------------------------|
| Encoder-only     | Bidirectional       | ❌               | Classification, QA        |
| Decoder-only     | Causal              | ❌               | Text generation            |
| Encoder-Decoder  | Encoder: Bi / Decoder: Causal | ✅     | Translation, Summarization |


#### 📐 Mathematical Formulation

**Encoder Layer:**

$$
\hat{X} = \text{LayerNorm}(X + \text{MultiHeadAttention}(X, X, X))
$$
$$
\text{EncoderOutput} = \text{LayerNorm}(\hat{X} + \text{FFN}(\hat{X}))
$$

**Decoder Layer:**

$$
\hat{Y} = \text{LayerNorm}(Y + \text{MultiHeadAttention}(Y, Y, Y, \text{mask}))
$$
$$
\hat{Y}' = \text{LayerNorm}(\hat{Y} + \text{MultiHeadAttention}(\hat{Y}, Z, Z))
$$
$$
\text{DecoderOutput} = \text{LayerNorm}(\hat{Y}' + \text{FFN}(\hat{Y}'))
$$

where $X$ is encoder input, $Y$ is decoder input, $Z$ is encoder output, and `mask` is the causal mask.

#### 💻 Simplified Python Pseudocode

```python
# Encoder Layer
def encoder_layer(X, mask=None):
    attn_output = layer_norm(X + multi_head_attention(X, mask=mask))
    return layer_norm(attn_output + feed_forward(attn_output))

# Decoder Layer
def decoder_layer(X, encoder_output, src_mask=None, tgt_mask=None):
    self_attn = layer_norm(X + multi_head_attention(X, mask=tgt_mask))
    cross_attn = layer_norm(self_attn + multi_head_attention(
        self_attn, encoder_output, encoder_output, mask=src_mask))
    return layer_norm(cross_attn + feed_forward(cross_attn))
```
