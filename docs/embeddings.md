# Multi-modal Embeddings

This module provides a unified interface for generating embeddings using various frameworks for text, image, audio, and multimodal data. It supports multiple embedding frameworks and models, making it easy to switch between different embedding solutions.

## Embedding Theory: From Word Vectors to Multimodal Representations

This section serves as an educational resource on the evolution and theory of embeddings across different modalities.

### The Evolution of Text Embeddings

#### Word2Vec (2013)

Word2Vec revolutionized NLP by introducing dense vector representations of words based on distributional semantics. Developed by Mikolov et al. at Google, it introduced two architectures:

1. **Continuous Bag of Words (CBOW)**: Predicts a target word from surrounding context words
2. **Skip-gram**: Predicts surrounding context words given a target word

The key insight was that words appearing in similar contexts tend to have similar meanings, captured by the famous equation:

$$\vec{v}(\text{"king"}) - \vec{v}(\text{"man"}) + \vec{v}(\text{"woman"}) \approx \vec{v}(\text{"queen"})$$

##### Skip-gram Architecture

The Skip-gram model consists of:
- An input layer of one-hot encoded words
- A hidden layer with N neurons (typically 100-300 dimensions)
- An output layer using softmax to predict context words

The Skip-gram objective function maximizes:

$$J(\theta) = \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j}|w_t)$$

where $c$ is the context window size and $p(w_{t+j}|w_t)$ is modeled using the softmax function:

$$p(w_O|w_I) = \frac{\exp(v'_{w_O}^T v_{w_I})}{\sum_{w=1}^{W} \exp(v'_{w}^T v_{w_I})}$$

Here, $v_{w_I}$ is the input vector for word $w_I$ and $v'_{w_O}$ is the output vector for word $w_O$.

##### CBOW Architecture

The CBOW model works in reverse, predicting a target word from context:

$$p(w_t|w_{t-c},...,w_{t-1},w_{t+1},...,w_{t+c}) = \frac{\exp(v'_{w_t}^T \bar{v})}{\sum_{w=1}^{W} \exp(v'_{w}^T \bar{v})}$$

where $\bar{v} = \frac{1}{2c}\sum_{-c \leq j \leq c, j \neq 0} v_{w_{t+j}}$ is the average of context word vectors.

##### Optimization Techniques

To address computational challenges with large vocabularies, two key techniques were introduced:

1. **Negative Sampling**: Instead of updating all output vectors, update only the positive sample and a few (5-20) randomly selected negative samples. The objective becomes:

$$\log \sigma(v'_{w_O}^T v_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)}[\log \sigma(-v'_{w_i}^T v_{w_I})]$$

where $\sigma$ is the sigmoid function, $k$ is the number of negative samples, and $P_n(w)$ is the noise distribution.

2. **Hierarchical Softmax**: Replaces the flat softmax with a binary tree structure, reducing complexity from O(V) to O(log V). Each internal node has a vector representation, and the probability of a word is the product of probabilities along the path from root to leaf:

$$p(w|w_I) = \prod_{j=1}^{L(w)-1} \sigma(\mathbb{1}\{n(w,j+1) = \text{left}(n(w,j))\} \cdot v'_{n(w,j)}\cdot v_{w_I})$$

where $n(w,j)$ is the $j$-th node on the path from root to $w$, and $L(w)$ is the path length.

##### Implementation Details

- **Subsampling**: Frequent words are randomly discarded during training with probability $P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}$, where $t$ is a threshold (typically 10^-5) and $f(w_i)$ is the word frequency.
- **Dynamic Context Windows**: The actual window size is randomly sampled between 1 and $c$ for each target word.
- **Learning Rate Scheduling**: Decreasing learning rate as training progresses.

**Key Papers**: 
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) (Mikolov et al., 2013)
- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) (Mikolov et al., 2013)

#### GloVe: Global Vectors for Word Representation (2014)

GloVe (Global Vectors for Word Representation) combined global matrix factorization with local context window methods. Unlike Word2Vec which is predictive, GloVe is count-based, utilizing word co-occurrence statistics from a corpus.

##### Mathematical Foundation

GloVe's approach is based on the insight that ratios of co-occurrence probabilities can encode meaning. For example, the ratio of P(ice|steam)/P(ice|solid) will be small, while P(ice|water)/P(ice|solid) will be closer to 1, revealing semantic relationships.

The model starts by constructing a word-word co-occurrence matrix $X$ where $X_{ij}$ represents how often word $i$ appears in the context of word $j$. The probability of word $j$ appearing in the context of word $i$ is then $P_{ij} = P(j|i) = X_{ij}/X_i$ where $X_i = \sum_k X_{ik}$.

The core of GloVe is minimizing the following cost function:

$$J = \sum_{i,j=1}^{V} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

where:
- $X_{ij}$ is the co-occurrence count between words $i$ and $j$
- $f(X_{ij})$ is a weighting function that prevents rare co-occurrences from being overweighted
- $w_i$ and $\tilde{w}_j$ are word vectors and context vectors
- $b_i$ and $\tilde{b}_j$ are bias terms

##### Weighting Function

The weighting function $f(X_{ij})$ is crucial for balancing the influence of frequent and rare co-occurrences:

$$f(x) = \begin{cases}
(x/x_{\max})^\alpha & \text{if } x < x_{\max} \\
1 & \text{otherwise}
\end{cases}$$

where $\alpha$ is typically set to 0.75 and $x_{\max}$ is often set to 100. This function ensures that:
- Very frequent co-occurrences are not overweighted
- Very rare co-occurrences (which may be noise) do not contribute too much to the loss
- Zero co-occurrences ($X_{ij} = 0$) are excluded entirely from the optimization

##### Implementation Details

1. **Co-occurrence Matrix Construction**:
   - A fixed context window size (typically 10 words) is used
   - Context words are weighted by their distance from the target word (e.g., 1/d where d is the distance)
   - The matrix is symmetric if using symmetric windows

2. **Optimization**:
   - AdaGrad is typically used for optimization
   - Learning rates around 0.05 are common
   - Vectors are typically initialized randomly with values between -0.5 and 0.5 divided by the embedding dimension

3. **Final Word Vectors**:
   - After training, both word vectors $w_i$ and context vectors $\tilde{w}_j$ are learned
   - The final word representation is often taken as their sum or average: $w_i^{final} = w_i + \tilde{w}_i$

##### Comparison with Word2Vec

| Aspect | GloVe | Word2Vec |
|--------|-------|----------|
| Approach | Count-based with matrix factorization | Prediction-based neural network |
| Training Data | Global co-occurrence statistics | Local context windows |
| Scalability | Requires storing co-occurrence matrix | Can be trained online |
| Parallelization | Easily parallelizable | More challenging to parallelize |
| Rare Words | Explicitly handled by weighting function | Implicitly handled by subsampling |
| Performance | Often better on analogy tasks | Often better on similarity tasks |

**Key Papers**: 
- [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/) (Pennington et al., 2014)
- [Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://aclanthology.org/Q15-1016/) (Levy et al., 2015)

#### Contextual Embeddings: BERT and Beyond (2018-present)

BERT (Bidirectional Encoder Representations from Transformers) marked a paradigm shift from static to contextual embeddings. Unlike Word2Vec and GloVe which assign a single vector to each word, BERT produces dynamic representations based on surrounding context.

##### Architecture

BERT is based on the Transformer architecture, specifically using only the encoder portion. The model comes in two main variants:
- **BERT-base**: 12 layers, 12 attention heads, 768 hidden dimensions (110M parameters)
- **BERT-large**: 24 layers, 16 attention heads, 1024 hidden dimensions (340M parameters)

Each layer consists of:
1. **Multi-head self-attention mechanism**
2. **Position-wise feed-forward network**
3. **Layer normalization and residual connections**

The input representation for each token is constructed by summing:
- **Token embeddings**: Learned embeddings for each token in the vocabulary
- **Segment embeddings**: Indicating which segment (sentence A or B) a token belongs to
- **Position embeddings**: Encoding the position of each token in the sequence

##### Self-Attention Mechanism

The core of BERT is the self-attention mechanism, which allows each token to attend to all other tokens in the sequence:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where:
- $Q = XW^Q$ are the query vectors
- $K = XW^K$ are the key vectors
- $V = XW^V$ are the value vectors
- $X$ is the input matrix
- $W^Q$, $W^K$, $W^V$ are learned parameter matrices
- $d_k$ is the dimension of the key vectors (scaling factor to prevent vanishing gradients)

BERT uses multi-head attention, which allows the model to jointly attend to information from different representation subspaces:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

where each head is computed as:

$$\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)$$

##### Position-wise Feed-Forward Network

After the attention layer, each position passes through an identical feed-forward network:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

This is applied to each position separately and identically, consisting of two linear transformations with a ReLU activation in between.

##### Pre-training Objectives

BERT is pre-trained using two unsupervised tasks:

1. **Masked Language Modeling (MLM)**:
   - Randomly mask 15% of the tokens in each sequence
   - Of these masked tokens:
     - 80% are replaced with the [MASK] token
     - 10% are replaced with a random token
     - 10% are left unchanged
   - The model must predict the original token based only on its context
   - Loss function: Cross-entropy loss over the masked tokens

   $$L_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P(x_i | \tilde{x})$$

   where $\tilde{x}$ is the corrupted input and $x_i$ is the original token.

2. **Next Sentence Prediction (NSP)**:
   - Given two sentences A and B, predict whether B actually follows A in the original text
   - 50% of the time B is the actual next sentence, 50% it's a random sentence
   - The [CLS] token representation is used for this binary classification task
   - Loss function: Binary cross-entropy

   $$L_{\text{NSP}} = -\log P(\text{isNext} | \text{[CLS]})$$

   The total pre-training loss is the sum: $L = L_{\text{MLM}} + L_{\text{NSP}}$

##### Tokenization

BERT uses WordPiece tokenization, a subword tokenization method that breaks uncommon words into subword units:

1. Start with a basic vocabulary of common words
2. Iteratively add the most frequent combinations of characters
3. Tokens that are not in the vocabulary are split into subwords (marked with ##)

Example: "embeddings" might be tokenized as ["em", "##bed", "##ding", "##s"]

##### Fine-tuning for Downstream Tasks

BERT can be fine-tuned for various NLP tasks with minimal architecture modifications:

- **Sequence Classification**: Add a classification layer on top of the [CLS] token representation
- **Token Classification**: Use the final hidden states of each token for tasks like NER
- **Question Answering**: Predict start and end positions of the answer span
- **Sentence Pair Tasks**: Use the [CLS] token representation with both sentences as input

##### BERT Variants and Improvements

- **RoBERTa** (Robustly Optimized BERT Approach):
  - Removes NSP objective
  - Uses dynamic masking (different masks each epoch)
  - Trains with larger batches and more data
  - Uses byte-level BPE tokenization

- **DistilBERT**:
  - 40% smaller, 60% faster, retains 97% of BERT's performance
  - Uses knowledge distillation during pre-training

- **ALBERT** (A Lite BERT):
  - Parameter reduction techniques: factorized embedding parameterization and cross-layer parameter sharing
  - Replaces NSP with Sentence Order Prediction (SOP)

- **ELECTRA**:
  - Replaced Token Detection instead of MLM
  - Generator-Discriminator architecture for more efficient pre-training

**Key Papers**:
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (Devlin et al., 2018)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) (Liu et al., 2019)
- [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) (Sanh et al., 2019)
- [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) (Lan et al., 2020)
- [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) (Clark et al., 2020)

#### Sentence Embeddings (2017-present)

Sentence embeddings aim to represent entire sentences or paragraphs as fixed-length vectors that capture their semantic meaning. While word embeddings like Word2Vec and GloVe revolutionized word-level representations, sentence embeddings address the need for document-level understanding.

##### Early Approaches

1. **Bag-of-Words Aggregation**:
   - Simple averaging of word vectors: $\vec{s} = \frac{1}{n}\sum_{i=1}^{n}\vec{w}_i$
   - TF-IDF weighted averaging: $\vec{s} = \frac{\sum_{i=1}^{n}\text{tfidf}(w_i)\vec{w}_i}{\sum_{i=1}^{n}\text{tfidf}(w_i)}$
   - Limitations: Loses word order and complex semantic relationships

2. **Doc2Vec** (2014):
   - Extension of Word2Vec that learns paragraph vectors alongside word vectors
   - Two variants: Distributed Memory (DM) and Distributed Bag of Words (DBOW)
   - Paragraph vectors act as a memory that captures the topic of the paragraph

3. **Skip-Thought Vectors** (2015):
   - Uses an encoder-decoder architecture
   - Given a sentence, predicts the previous and next sentences
   - Encoder's output serves as the sentence embedding

##### Transformer-Based Approaches

1. **BERT [CLS] Token**:
   - The [CLS] token from the final layer of BERT can represent the entire sentence
   - Limitations: Not optimized for sentence similarity; performs poorly without fine-tuning

2. **Sentence-BERT (SBERT)** (2019):
   - Fine-tunes BERT/RoBERTa in a siamese/triplet network structure
   - Uses mean pooling over token embeddings: $\vec{s} = \frac{1}{n}\sum_{i=1}^{n}\vec{t}_i$
   - Dramatically improves performance and efficiency for similarity tasks

   **Architecture**:
   - Identical BERT networks process sentence pairs
   - Pooling layer (usually mean pooling) aggregates token embeddings
   - Optional projection layer maps to the final embedding space

   **Training Objectives**:

   a. **Classification Objective** (NLI datasets):
      - Given premise $p$ and hypothesis $h$, predict entailment, contradiction, or neutral
      - Uses concatenation of embeddings: $[\vec{u}, \vec{v}, |\vec{u}-\vec{v}|]$

   b. **Regression Objective** (STS datasets):
      - Predict similarity score between sentence pairs
      - Mean squared error loss: $L = (\text{sim}(\vec{u}, \vec{v}) - \text{label})^2$

   c. **Triplet Objective**:
      - Uses anchor $a$, positive $p$, and negative $n$ sentences
      - Contrastive loss: $L(a, p, n) = \max(||f(a) - f(p)||_2 - ||f(a) - f(n)||_2 + \text{margin}, 0)$

3. **SimCSE** (2021):
   - Uses contrastive learning with innovative positive/negative pair creation
   - **Unsupervised SimCSE**: Uses dropout as data augmentation; the same sentence through the encoder twice creates positive pairs
   - **Supervised SimCSE**: Uses NLI datasets where entailment pairs are positives and contradiction pairs are negatives

   **Training Objective**:
   - Contrastive loss with in-batch negatives:

   $$L_i = -\log \frac{e^{\text{sim}(\mathbf{h}_i, \mathbf{h}_i^+)/\tau}}{\sum_{j=1}^N e^{\text{sim}(\mathbf{h}_i, \mathbf{h}_j^+)/\tau}}$$

   where $\mathbf{h}_i$ and $\mathbf{h}_i^+$ are embeddings of positive pairs, $\tau$ is a temperature parameter, and $N$ is the batch size.

4. **DeCLUTR** (2021):
   - Creates positive pairs by sampling different spans from the same document
   - Uses contrastive learning with carefully designed span sampling strategies

5. **MPNet** and **E5** (2022-2023):
   - MPNet combines the strengths of BERT (bidirectional context) and XLNet (permutation-based training)
   - E5 uses contrastive pre-training on web-scale data with a retrieve-then-rerank approach

##### Specialized Sentence Embedding Models

1. **Universal Sentence Encoder (USE)**:
   - Trained on multiple tasks including NLI, question-answer prediction, and translation
   - Two variants: Transformer-based (higher accuracy) and DAN-based (faster inference)

2. **LaBSE (Language-agnostic BERT Sentence Embedding)**:
   - Trained on 109 languages for cross-lingual sentence retrieval
   - Uses translation pairs as positive examples in contrastive learning

3. **GTR (Generative Text Retrieval)**:
   - Uses T5 encoder for generating sentence embeddings
   - Trained with contrastive learning on MS MARCO dataset

##### Practical Considerations

1. **Pooling Strategies**:
   - Mean pooling: Average of all token embeddings (most common)
   - Max pooling: Element-wise maximum across token embeddings
   - CLS pooling: Using only the [CLS] token embedding
   - Attention pooling: Weighted average using learned attention weights

2. **Normalization**:
   - L2 normalization is crucial for cosine similarity calculations
   - Some models apply layer normalization before pooling

3. **Hard Negative Mining**:
   - Finding challenging negative examples improves model performance
   - Techniques include in-batch negatives, cross-batch negatives, and iterative mining

##### SentenceTransformers Framework

**SentenceTransformers** is the most widely adopted framework for sentence embeddings, providing a unified interface for training and using sentence embedding models. Developed by Nils Reimers, it has become the de facto standard for sentence embedding applications.

**Architecture and Design**:
- **Modular Design**: Supports various transformer models (BERT, RoBERTa, DistilBERT, etc.) as backbone encoders
- **Flexible Pooling**: Multiple pooling strategies (mean, max, CLS token, weighted mean)
- **Training Pipeline**: Streamlined training with various loss functions and evaluation metrics
- **Model Hub Integration**: Seamless integration with Hugging Face Model Hub

**Implementation Reference**: [SentenceTransformers GitHub](https://github.com/UKPLab/sentence-transformers)

**Key Components**:

1. **SentenceTransformer Class**:
   ```python
   # Core implementation in sentence_transformers/SentenceTransformer.py
   class SentenceTransformer(nn.Module):
       def __init__(self, model_name_or_path, modules=None, device=None):
           # Initialize transformer model and pooling layer
   ```
   [Implementation](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py#L89)

2. **Pooling Strategies**:
   ```python
   # sentence_transformers/models/Pooling.py
   class Pooling(nn.Module):
       def __init__(self, word_embedding_dimension, pooling_mode='mean'):
           # Implements mean, max, cls pooling strategies
   ```
   [Implementation](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py)

##### all-MiniLM-L6-v2: Deep Dive Analysis

**all-MiniLM-L6-v2** is one of the most popular sentence embedding models, offering an excellent balance between performance and efficiency. It's based on the MiniLM architecture with specific optimizations for sentence-level tasks.

**Architecture Details**:
- **Base Model**: DistilBERT-like architecture with 6 layers
- **Hidden Size**: 384 dimensions
- **Attention Heads**: 12
- **Parameters**: ~23M (significantly smaller than BERT-base's 110M)
- **Max Sequence Length**: 512 tokens
- **Output Dimensions**: 384-dimensional sentence embeddings

**Training Process**:

1. **Knowledge Distillation**: Trained using knowledge distillation from larger teacher models
   - Teacher models: Multiple large sentence embedding models
   - Student model: 6-layer MiniLM architecture
   - Distillation loss combines multiple objectives

2. **Multi-Task Training**: Trained on diverse datasets:
   - **Natural Language Inference**: SNLI, MultiNLI, XNLI
   - **Semantic Textual Similarity**: STS benchmark datasets
   - **Question-Answer Pairs**: Quora, Stack Exchange, MS MARCO
   - **Paraphrase Detection**: Various paraphrase datasets

3. **Training Objective**:
   ```python
   # Simplified training objective combining multiple losses
   total_loss = λ₁ * nli_loss + λ₂ * sts_loss + λ₃ * qa_loss + λ₄ * distillation_loss
   ```

**Performance Characteristics**:
- **Speed**: ~5x faster than BERT-base for inference
- **Memory**: ~4x less memory usage
- **Quality**: Retains ~95% of larger model performance on most tasks
- **Versatility**: Excellent performance across multiple domains and languages

**Model Card**: [all-MiniLM-L6-v2 on Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

**Usage Example**:
```python
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
sentences = ['This is an example sentence', 'Each sentence is converted']
embeddings = model.encode(sentences)
```

##### Siamese and Triplet Network Architectures

**Siamese Networks** and **Triplet Networks** are fundamental architectures for learning similarity-based embeddings, particularly effective for sentence embeddings.

**Siamese Network Architecture**:

A Siamese network consists of two identical neural networks (sharing weights) that process two inputs simultaneously:

```
Input A ──→ [Encoder] ──→ Embedding A
                │
                │ (shared weights)
                │
Input B ──→ [Encoder] ──→ Embedding B
                │
                ▼
        [Similarity Function]
                │
                ▼
            Similarity Score
```

**Implementation Steps**:

1. **Shared Encoder**: Both inputs pass through the same transformer encoder
   ```python
   # sentence_transformers/models/Transformer.py
   class Transformer(nn.Module):
       def forward(self, features):
           # Process input through transformer layers
           return self.auto_model(**features)
   ```
   [Implementation](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Transformer.py)

2. **Pooling Layer**: Convert token embeddings to sentence embeddings
3. **Similarity Computation**: Calculate cosine similarity or Euclidean distance

**Triplet Network Architecture**:

Triplet networks extend Siamese networks to work with three inputs: anchor, positive, and negative examples:

```
Anchor ────→ [Encoder] ──→ Embedding A
Positive ──→ [Encoder] ──→ Embedding P  
Negative ──→ [Encoder] ──→ Embedding N
                │
                ▼
        [Triplet Loss Function]
```

**Training Process**:
1. **Triplet Mining**: Select challenging triplets (hard negatives)
2. **Forward Pass**: Generate embeddings for all three inputs
3. **Loss Calculation**: Apply triplet loss function
4. **Backpropagation**: Update shared encoder weights

##### Loss Functions for Sentence Embeddings

**1. Triplet Loss**

Triplet loss ensures that the distance between anchor and positive is smaller than the distance between anchor and negative by a margin:

$$L_{\text{triplet}}(a, p, n) = \max(0, d(a, p) - d(a, n) + \text{margin})$$

where:
- $a$, $p$, $n$ are anchor, positive, and negative embeddings
- $d(\cdot, \cdot)$ is the distance function (usually Euclidean or cosine)
- $\text{margin}$ is a hyperparameter (typically 0.5)

**Implementation**:
```python
# sentence_transformers/losses/TripletLoss.py
class TripletLoss(nn.Module):
    def __init__(self, model, distance_metric=SiameseDistanceMetric.COSINE, triplet_margin=0.5):
        # Initialize triplet loss with specified distance metric and margin
```
[Implementation](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/TripletLoss.py)

**Triplet Mining Strategies**:
- **Random Triplets**: Randomly sample triplets from the dataset
- **Hard Triplets**: Select triplets where the negative is closer to anchor than positive
- **Semi-Hard Triplets**: Negatives that are farther than positive but within the margin
- **Online Mining**: Mine triplets during training based on current model state

**2. Contrastive Loss**

Contrastive loss works with pairs of examples, pulling similar pairs together and pushing dissimilar pairs apart:

$$L_{\text{contrastive}}(x_1, x_2, y) = y \cdot d(x_1, x_2)^2 + (1-y) \cdot \max(0, \text{margin} - d(x_1, x_2))^2$$

where:
- $y = 1$ for similar pairs, $y = 0$ for dissimilar pairs
- $d(x_1, x_2)$ is the Euclidean distance between embeddings
- $\text{margin}$ defines the minimum distance for dissimilar pairs

**Implementation**:
```python
# sentence_transformers/losses/ContrastiveLoss.py
class ContrastiveLoss(nn.Module):
    def __init__(self, model, distance_metric=SiameseDistanceMetric.EUCLIDEAN, margin=0.5):
        # Initialize contrastive loss with distance metric and margin
```
[Implementation](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/ContrastiveLoss.py)

**3. Multiple Negatives Ranking Loss (MNRL)**

MNRL is a more efficient alternative to triplet loss, using in-batch negatives to create multiple negative examples:

$$L_{\text{MNRL}} = -\log \frac{e^{\text{sim}(a, p)/\tau}}{e^{\text{sim}(a, p)/\tau} + \sum_{i=1}^{N} e^{\text{sim}(a, n_i)/\tau}}$$

where:
- $a$ is the anchor (query)
- $p$ is the positive example
- $n_i$ are negative examples (other examples in the batch)
- $\tau$ is the temperature parameter
- $\text{sim}(\cdot, \cdot)$ is the similarity function (usually cosine similarity)

**Implementation**:
```python
# sentence_transformers/losses/MultipleNegativesRankingLoss.py
class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, model, scale=20.0, similarity_fct=util.cos_sim):
        # Initialize MNRL with scaling factor and similarity function
```
[Implementation](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py)

**Advantages of MNRL**:
- **Efficiency**: Uses all examples in a batch as negatives
- **Scalability**: No need for explicit negative sampling
- **Performance**: Often outperforms triplet loss with proper batch size
- **Simplicity**: Easier to implement and tune than triplet mining strategies

**4. CoSENT Loss**

CoSENT (Cosine Sentence) loss is designed specifically for sentence similarity tasks:

$$L_{\text{CoSENT}} = \log(1 + \sum_{i=1}^{N} \sum_{j=1}^{N} \mathbb{1}_{y_i < y_j} e^{\lambda(\cos(u_i, v_i) - \cos(u_j, v_j))})$$

where:
- $(u_i, v_i)$ and $(u_j, v_j)$ are sentence pairs
- $y_i$ and $y_j$ are their similarity labels
- $\lambda$ is a scaling factor
- $\cos(\cdot, \cdot)$ is cosine similarity

**Implementation**:
```python
# sentence_transformers/losses/CoSENTLoss.py
class CoSENTLoss(nn.Module):
    def __init__(self, model, scale=20.0):
        # Initialize CoSENT loss with scaling parameter
```
[Implementation](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/CoSENTLoss.py)

##### Advanced Training Techniques

**1. Hard Negative Mining**

Hard negative mining improves model performance by focusing on challenging examples:

```python
# Example implementation of hard negative mining
def mine_hard_negatives(model, anchors, candidates, top_k=5):
    # Encode all sentences
    anchor_embeddings = model.encode(anchors)
    candidate_embeddings = model.encode(candidates)
    
    # Compute similarities
    similarities = util.cos_sim(anchor_embeddings, candidate_embeddings)
    
    # Select top-k most similar negatives (hardest negatives)
    hard_negatives = torch.topk(similarities, k=top_k, dim=1).indices
    return hard_negatives
```

**2. Curriculum Learning**

Gradually increase training difficulty by starting with easy examples and progressing to harder ones:

```python
# Curriculum learning implementation
class CurriculumSampler:
    def __init__(self, dataset, difficulty_scores):
        self.dataset = dataset
        self.difficulty_scores = difficulty_scores
        self.current_threshold = 0.1  # Start with easiest 10%
    
    def get_batch(self, epoch):
        # Gradually increase difficulty threshold
        self.current_threshold = min(1.0, 0.1 + epoch * 0.1)
        # Sample examples below difficulty threshold
        return self.sample_by_difficulty()
```

**3. Data Augmentation for Sentence Embeddings**

- **Back-translation**: Translate to another language and back
- **Paraphrasing**: Use paraphrase generation models
- **Token-level augmentation**: Random insertion, deletion, substitution
- **Dropout augmentation**: Different dropout masks for the same sentence

**Research Directions and Future Work**:

1. **Multilingual Sentence Embeddings**:
   - Cross-lingual alignment techniques
   - Language-agnostic representation learning
   - Zero-shot cross-lingual transfer
   - Papers: [LaBSE](https://arxiv.org/abs/2007.01852), [LASER](https://arxiv.org/abs/1812.10464)

2. **Domain Adaptation**:
   - Unsupervised domain adaptation for embeddings
   - Few-shot learning for new domains
   - Domain-adversarial training
   - Papers: [Domain Adaptation](https://arxiv.org/abs/2004.02349)

3. **Efficient Training Methods**:
   - Knowledge distillation for smaller models
   - Progressive training strategies
   - Mixed precision training
   - Papers: [DistilBERT](https://arxiv.org/abs/1910.01108), [TinyBERT](https://arxiv.org/abs/1909.10351)

4. **Evaluation and Benchmarking**:
   - Comprehensive evaluation frameworks
   - Bias detection in sentence embeddings
   - Robustness testing
   - Papers: [SentEval](https://arxiv.org/abs/1803.05449), [MTEB](https://arxiv.org/abs/2210.07316)

**Key Papers**:
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) (Reimers & Gurevych, 2019)
- [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821) (Gao et al., 2021)
- [DeCLUTR: Deep Contrastive Learning for Unsupervised Textual Representations](https://arxiv.org/abs/2006.03659) (Giorgi et al., 2021)
- [E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533) (Wang et al., 2022)
- [Text and Code Embeddings by Contrastive Pre-Training](https://arxiv.org/abs/2201.10005) (Neelakantan et al., 2022)
- [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813) (Reimers & Gurevych, 2020)
- [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) (Muennighoff et al., 2022)

#### Decoder-Based Embeddings: GPT and Beyond (2018-present)

While encoder models like BERT excel at understanding, decoder models like GPT (Generative Pre-trained Transformer) excel at generation. Interestingly, these decoder-based models can also produce high-quality embeddings, despite their architectural differences from traditional embedding models.

##### Architecture of Decoder-Based Models

GPT and similar decoder-based models use a unidirectional (autoregressive) architecture:

1. **Causal Self-Attention**: Each token can only attend to itself and previous tokens, implemented using an attention mask:

   $$\text{CausalAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right)V$$

   where $M$ is a mask that sets all values corresponding to future positions to $-\infty$:

   $$M_{ij} = \begin{cases}
   0 & \text{if } i \geq j \\
   -\infty & \text{if } i < j
   \end{cases}$$

2. **Position-wise Feed-Forward Network**: Similar to BERT, but with potentially different activation functions (e.g., GPT-2 uses GELU instead of ReLU).

3. **Layer Normalization**: Applied before each sub-layer, rather than after (pre-norm vs. post-norm).

##### GPT Family Evolution

1. **GPT-1** (2018):
   - 12 layers, 768 hidden dimensions, 12 attention heads (117M parameters)
   - Pre-trained on BookCorpus (800M words)
   - Fine-tuned on specific downstream tasks

2. **GPT-2** (2019):
   - Scaled up to 1.5B parameters in largest variant
   - Pre-trained on WebText (40GB of text from 8M web pages)
   - Zero-shot task transfer without fine-tuning

3. **GPT-3** (2020):
   - Massive scale-up to 175B parameters
   - Pre-trained on Common Crawl, WebText2, Books1, Books2, and Wikipedia
   - Few-shot learning capabilities through in-context learning

4. **GPT-4** (2023):
   - Multimodal capabilities (text and images)
   - Further scaling and architectural improvements
   - Significantly improved reasoning capabilities

##### Embedding Generation Approaches

1. **Last Hidden State**:
   - The simplest approach is to use the final hidden state of the last token as the sentence embedding
   - Limitation: Heavily biased toward the last tokens in the sequence

2. **Mean Pooling**:
   - Average the hidden states across all tokens
   - More balanced representation of the entire sequence

3. **Specialized Embedding Models**:
   - OpenAI's `text-embedding-ada-002` is based on a GPT-like architecture but specifically trained for embedding generation
   - Uses contrastive learning objectives similar to those in SimCSE

4. **Instruction Tuning**:
   - Models like `text-embedding-3-large` are instruction-tuned to produce embeddings optimized for specific use cases
   - Can generate different embeddings for the same text based on the provided instruction

##### Training Objectives for Embedding Generation

1. **Contrastive Learning**:
   - Similar to encoder-based models, using positive and negative pairs
   - Often uses retrieval-based tasks during training

2. **Dual Encoder Training**:
   - Training separate query and document encoders
   - Optimizing for retrieval performance

3. **Multi-task Learning**:
   - Combining generative pre-training with embedding-specific objectives
   - Balancing between generation quality and embedding quality

##### Applications of Decoder-Based Embeddings

1. **Semantic Search**:
   - OpenAI's embeddings are widely used for retrieval-augmented generation (RAG)
   - Can capture nuanced semantic relationships better than some encoder-only models

2. **Zero-shot Classification**:
   - Using embeddings to compare inputs with potential class descriptions
   - Leveraging the model's world knowledge encoded in the embeddings

3. **Content Recommendation**:
   - Representing user preferences and content in the same embedding space
   - Capturing subtle semantic relationships for better recommendations

4. **Embedding-guided Generation**:
   - Using embeddings to guide text generation toward specific semantic goals
   - Controlling style, tone, or content through embedding space manipulation

##### Advantages of Decoder-Based Embeddings

1. **World Knowledge**: Large decoder models encode vast amounts of world knowledge that can be reflected in their embeddings

2. **Contextual Understanding**: Strong ability to disambiguate based on context

3. **Adaptability**: Can be prompted or fine-tuned to produce embeddings for specific domains or tasks

4. **Alignment with Generation**: When used in retrieval-augmented generation, embeddings from the same model family can provide better alignment

##### Challenges and Limitations

1. **Computational Cost**: Larger models require significant resources

2. **Unidirectionality**: The causal attention mechanism may limit bidirectional understanding

3. **Embedding Drift**: Embeddings from different versions of models may not be compatible

4. **Black-box Nature**: Commercial embeddings like those from OpenAI have limited transparency

##### Embedding Extraction from Decoder Models

**Last Token Embeddings**:
For decoder models, embeddings are typically extracted from the last token's hidden state:

```python
# Example with Hugging Face Transformers
from transformers import GPT2Model, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# Add padding token
tokenizer.pad_token = tokenizer.eos_token

def get_gpt_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract last token embeddings
    last_token_embeddings = outputs.last_hidden_state[:, -1, :]
    return last_token_embeddings
```

**Mean Pooling for Decoder Models**:
Alternatively, mean pooling can be applied to all token embeddings:

```python
def get_gpt_embeddings_mean_pooled(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Apply attention mask and mean pool
    embeddings = outputs.last_hidden_state
    masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
    mean_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    
    return mean_embeddings
```

**Implementation Reference**: [Hugging Face Transformers GPT Models](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gpt2)

##### OpenAI Text Embeddings API

OpenAI provides specialized embedding models optimized for various tasks:

**text-embedding-ada-002**:
- 1536-dimensional embeddings
- Optimized for semantic search and similarity tasks
- Cost-effective and high-performance

**text-embedding-3-small** and **text-embedding-3-large**:
- Newer models with improved performance
- Configurable output dimensions
- Better multilingual support

```python
# OpenAI Embeddings API usage
import openai

def get_openai_embeddings(texts, model="text-embedding-3-small"):
    response = openai.Embedding.create(
        input=texts,
        model=model
    )
    return [data['embedding'] for data in response['data']]
```

**API Documentation**: [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)

**Key Papers and Resources**:
- [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) (Radford et al., 2018)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford et al., 2019)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Brown et al., 2020)
- [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368) (Neelakantan et al., 2024)
- [OpenAI Embeddings Documentation](https://platform.openai.com/docs/guides/embeddings)

### Multimodal Embeddings

Multimodal embeddings extend beyond text to incorporate visual, audio, and other modalities, enabling cross-modal understanding and retrieval.

#### Vision-Language Models

##### CLIP: Contrastive Language-Image Pre-training (2021)

**CLIP** revolutionized multimodal understanding by learning joint representations of images and text through contrastive learning.

**Architecture**:
- **Text Encoder**: Transformer-based (similar to GPT-2)
- **Image Encoder**: Vision Transformer (ViT) or ResNet
- **Joint Embedding Space**: Both modalities mapped to the same dimensional space

**Training Objective**:
CLIP uses contrastive learning on image-text pairs:

$$L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)}$$

where:
- $I_i$ and $T_i$ are image and text embeddings for the $i$-th pair
- $\text{sim}(\cdot, \cdot)$ is cosine similarity
- $\tau$ is a learnable temperature parameter
- $N$ is the batch size

**Implementation**:
```python
# Using OpenAI's CLIP
import clip
import torch
from PIL import Image

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Process image and text
image = preprocess(Image.open("image.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a photo of a cat", "a photo of a dog"]).to(device)

# Generate embeddings
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Calculate similarity
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
```

**Implementation Reference**: [OpenAI CLIP GitHub](https://github.com/openai/CLIP)

**Key Features**:
- **Zero-shot Classification**: Can classify images without task-specific training
- **Cross-modal Retrieval**: Find images using text queries and vice versa
- **Robust Representations**: Learned from 400M image-text pairs from the internet

##### Vision Transformer (ViT) for Image Embeddings

**Vision Transformer** applies the transformer architecture directly to image patches, treating them as sequences.

**Architecture**:
1. **Patch Embedding**: Divide image into fixed-size patches and linearly embed them
2. **Position Embedding**: Add learnable position embeddings to patch embeddings
3. **Transformer Encoder**: Standard transformer layers with self-attention
4. **Classification Head**: MLP head for classification or embedding extraction

**Patch Embedding Process**:
```python
# Simplified ViT patch embedding
def create_patch_embeddings(image, patch_size=16):
    # image shape: (batch_size, channels, height, width)
    batch_size, channels, height, width = image.shape
    
    # Calculate number of patches
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    
    # Reshape to patches
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(batch_size, channels, -1, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4).contiguous()
    patches = patches.view(batch_size, -1, channels * patch_size * patch_size)
    
    return patches
```

**Implementation Reference**: [Hugging Face ViT](https://github.com/huggingface/transformers/tree/main/src/transformers/models/vit)

**Usage Example**:
```python
from transformers import ViTModel, ViTFeatureExtractor
from PIL import Image

# Load model and feature extractor
model = ViTModel.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Process image
image = Image.open('image.jpg')
inputs = feature_extractor(images=image, return_tensors="pt")

# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs)
    # Use CLS token embedding
    image_embedding = outputs.last_hidden_state[:, 0, :]
```

#### Audio Embeddings

##### Wav2Vec 2.0: Self-Supervised Audio Representations

**Wav2Vec 2.0** learns powerful audio representations through self-supervised learning on raw audio waveforms.

**Architecture**:
1. **Feature Encoder**: CNN layers that process raw audio
2. **Contextualized Representations**: Transformer layers for sequence modeling
3. **Quantization Module**: Discretizes latent representations

**Training Objective**:
Contrastive learning with masked prediction:

$$L = -\log \frac{\exp(\text{sim}(c_t, q_t) / \tau)}{\sum_{\tilde{q} \in Q_t} \exp(\text{sim}(c_t, \tilde{q}) / \tau)}$$

where:
- $c_t$ is the contextualized representation at time step $t$
- $q_t$ is the quantized target representation
- $Q_t$ is the set of distractors

**Implementation**:
```python
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch
import librosa

# Load model and processor
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

def get_audio_embeddings(audio_path):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Process audio
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pool over time dimension
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings
```

**Implementation Reference**: [Hugging Face Wav2Vec2](https://github.com/huggingface/transformers/tree/main/src/transformers/models/wav2vec2)

##### OpenAI Whisper for Audio Understanding

**Whisper** is a robust speech recognition model that can also provide audio embeddings:

```python
import whisper

# Load model
model = whisper.load_model("base")

def get_whisper_embeddings(audio_path):
    # Load and process audio
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    
    # Generate mel spectrogram
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    # Encode audio
    with torch.no_grad():
        audio_features = model.encoder(mel.unsqueeze(0))
    
    return audio_features
```

**Implementation Reference**: [OpenAI Whisper GitHub](https://github.com/openai/whisper)

#### Multimodal Fusion Techniques

##### Early Fusion
Combine features from different modalities at the input level:

```python
class EarlyFusionModel(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, text_features, image_features):
        text_proj = self.text_proj(text_features)
        image_proj = self.image_proj(image_features)
        
        # Concatenate and fuse
        fused = torch.cat([text_proj, image_proj], dim=-1)
        output = self.fusion_layer(fused)
        
        return output
```

##### Late Fusion
Combine predictions from separate modality-specific models:

```python
class LateFusionModel(nn.Module):
    def __init__(self, text_model, image_model, num_classes):
        super().__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.fusion_weights = nn.Parameter(torch.ones(2))
        
    def forward(self, text_input, image_input):
        text_logits = self.text_model(text_input)
        image_logits = self.image_model(image_input)
        
        # Weighted combination
        weights = F.softmax(self.fusion_weights, dim=0)
        fused_logits = weights[0] * text_logits + weights[1] * image_logits
        
        return fused_logits
```

##### Cross-Attention Fusion
Use attention mechanisms to model cross-modal interactions:

```python
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, text_features, image_features):
        # text_features: (seq_len, batch, embed_dim)
        # image_features: (num_patches, batch, embed_dim)
        
        # Cross-attention: text attends to image
        attended_text, _ = self.cross_attention(
            query=text_features,
            key=image_features,
            value=image_features
        )
        
        # Residual connection and layer norm
        output = self.layer_norm(text_features + attended_text)
        
        return output
```

**Research Directions in Multimodal Embeddings**:

1. **Large-Scale Multimodal Models**:
   - DALL-E, DALL-E 2, Stable Diffusion
   - GPT-4V (Vision), LLaVA, BLIP-2
   - Papers: [DALL-E](https://arxiv.org/abs/2102.12092), [LLaVA](https://arxiv.org/abs/2304.08485)

2. **Video Understanding**:
   - Temporal modeling in video embeddings
   - Action recognition and video retrieval
   - Papers: [VideoBERT](https://arxiv.org/abs/1904.01766), [Video-ChatGPT](https://arxiv.org/abs/2306.05424)

3. **3D and Spatial Embeddings**:
   - Point cloud representations
   - 3D scene understanding
   - Papers: [PointNet](https://arxiv.org/abs/1612.00593), [NeRF](https://arxiv.org/abs/2003.08934)

4. **Efficient Multimodal Training**:
   - Parameter-efficient fine-tuning
   - Modality-specific adapters
   - Papers: [AdapterFusion](https://arxiv.org/abs/2005.00247), [LoRA](https://arxiv.org/abs/2106.09685)

**Key Papers**:
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) (Radford et al., 2021)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2021)
- [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) (Baevski et al., 2020)
- [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](https://arxiv.org/abs/2212.04356) (Radford et al., 2022)
- [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) (Li et al., 2022)

### Image Embeddings

#### Convolutional Neural Networks (CNNs)

CNNs revolutionized computer vision by learning hierarchical features from images. The convolutional operation is defined as:

$$S(i, j) = (I * K)(i, j) = \sum_m \sum_n I(i+m, j+n) K(m, n)$$

where $I$ is the input image, $K$ is the kernel, and $S$ is the output feature map.

##### CNN Architecture Components

1. **Convolutional Layers**: The core building block that applies filters to detect features:

   $$\mathbf{h}_{i,j,d} = \sum_{c=1}^{C} \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \mathbf{W}_{m,n,c,d} \cdot \mathbf{x}_{i+m, j+n, c} + \mathbf{b}_d$$

   where:
   - $\mathbf{h}_{i,j,d}$ is the output at position $(i,j)$ for the $d$-th output channel
   - $\mathbf{W}$ is the kernel of size $k \times k \times C \times D$ (height, width, input channels, output channels)
   - $\mathbf{x}$ is the input tensor
   - $\mathbf{b}_d$ is the bias term for the $d$-th output channel
   - $C$ is the number of input channels

2. **Pooling Layers**: Reduce spatial dimensions while preserving important features:
   - Max Pooling: $\mathbf{h}_{i,j} = \max_{0\leq m<s, 0\leq n<s} \mathbf{x}_{s\cdot i+m, s\cdot j+n}$
   - Average Pooling: $\mathbf{h}_{i,j} = \frac{1}{s^2}\sum_{m=0}^{s-1} \sum_{n=0}^{s-1} \mathbf{x}_{s\cdot i+m, s\cdot j+n}$

   where $s$ is the stride/pool size.

3. **Normalization Layers**:
   - Batch Normalization: $\hat{\mathbf{x}} = \frac{\mathbf{x} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \cdot \gamma + \beta$
   - Layer Normalization: Normalizes across channels for each sample

4. **Activation Functions**:
   - ReLU: $f(x) = \max(0, x)$
   - Leaky ReLU: $f(x) = \max(\alpha x, x)$ where $\alpha$ is a small constant
   - ELU: $f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$

5. **Fully Connected Layers**: Transform feature maps into embeddings:
   - $\mathbf{h} = \mathbf{W} \cdot \mathbf{x} + \mathbf{b}$

Models like ResNet introduced skip connections to address the vanishing gradient problem:

$$y = F(x, \{W_i\}) + x$$

where $F$ represents the residual mapping to be learned.

##### Major CNN Architectures for Embeddings

1. **AlexNet** (2012):
   - 5 convolutional layers, 3 fully connected layers
   - First major CNN success on ImageNet
   - 60 million parameters
   - Introduced ReLU activations, dropout, and data augmentation

2. **VGG** (2014):
   - Simple, uniform architecture with 3×3 convolutions
   - Very deep (16-19 layers)
   - 138 million parameters (VGG-16)
   - Embedding dimension: 4096 (fc7 layer)

3. **ResNet** (2015):
   - Introduced residual connections: $\mathbf{h} = F(\mathbf{x}) + \mathbf{x}$
   - Solved vanishing gradient problem in very deep networks
   - Variants from 18 to 152 layers
   - Embedding dimension: 2048 (final layer before classification)

4. **Inception/GoogLeNet** (2014):
   - Multi-scale processing using parallel convolutions
   - Efficient use of parameters (6.8 million)
   - Embedding dimension: 1024 (pool5 layer)

5. **EfficientNet** (2019):
   - Compound scaling of depth, width, and resolution
   - State-of-the-art performance with fewer parameters
   - Variants from B0 (5.3M parameters) to B7 (66M parameters)
   - Embedding dimension: varies by model size (1280 for B0)

##### CNN Embedding Extraction Techniques

1. **Global Average Pooling (GAP)**:
   - Average all spatial locations in the final convolutional layer
   - $\mathbf{h}_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} \mathbf{x}_{i,j,c}$
   - Dimension equals number of channels in final conv layer
   - Spatially invariant representation

2. **Global Max Pooling (GMP)**:
   - Take maximum activation across spatial dimensions
   - More sensitive to distinctive features

3. **Fully Connected Layer Activations**:
   - Use activations from penultimate layer (before classification)
   - Higher dimensional but more discriminative

4. **Multi-level Feature Aggregation**:
   - Combine features from multiple layers for richer representation
   - $\mathbf{h} = [\text{GAP}(\mathbf{x}^{(l_1)}), \text{GAP}(\mathbf{x}^{(l_2)}), ..., \text{GAP}(\mathbf{x}^{(l_n)})]$
   - Captures both low-level and high-level features

##### Training Objectives for CNN Embeddings

1. **Supervised Classification**:
   - Traditional cross-entropy loss: $L = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(p_{i,c})$
   - Embeddings emerge as a byproduct of classification training

2. **Metric Learning**:
   - Contrastive loss: $L = \sum_{i=1}^{N} \sum_{j=1}^{N} y_{i,j} d(\mathbf{h}_i, \mathbf{h}_j)^2 + (1-y_{i,j}) \max(0, m - d(\mathbf{h}_i, \mathbf{h}_j))^2$
   - Triplet loss: $L = \sum_{i=1}^{N} \max(0, d(\mathbf{h}_i, \mathbf{h}_i^+) - d(\mathbf{h}_i, \mathbf{h}_i^-) + m)$
   - N-pair loss, angular loss, etc.

3. **Self-supervised Learning**:
   - Pretext tasks: rotation prediction, jigsaw puzzles, colorization
   - Contrastive predictive coding
   - SimCLR, MoCo, BYOL, etc.

##### Applications of CNN Embeddings

1. **Image Retrieval**:
   - Content-based image retrieval systems
   - Reverse image search
   - Product recommendation

2. **Face Recognition**:
   - FaceNet, ArcFace, CosFace use CNN embeddings
   - Verification via embedding distance

3. **Transfer Learning**:
   - Feature extraction for downstream tasks
   - Fine-tuning on domain-specific data

4. **Image Clustering and Organization**:
   - Unsupervised grouping of similar images
   - Visual data exploration

##### Implementation Considerations

1. **Feature Normalization**:
   - L2 normalization: $\hat{\mathbf{h}} = \frac{\mathbf{h}}{\|\mathbf{h}\|_2}$
   - Improves performance in similarity calculations

2. **Dimensionality Reduction**:
   - PCA, t-SNE, or UMAP for visualization
   - Linear projection layers for efficiency

3. **Data Augmentation**:
   - Random crops, flips, rotations, color jittering
   - Improves robustness and generalization

4. **Fine-tuning Strategies**:
   - Layer-wise learning rates
   - Progressive unfreezing

**Key Papers**:
- [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) (Krizhevsky et al., 2012)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (Simonyan & Zisserman, 2014)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (He et al., 2015)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (Tan & Le, 2019)
- [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) (Chen et al., 2020)

#### Vision Transformers (ViT) (2020-present)

Vision Transformers (ViT) revolutionized computer vision by adapting the Transformer architecture from NLP to images, demonstrating that self-attention mechanisms can effectively process visual data without convolutional operations.

##### ViT Architecture

1. **Image Patching and Embedding**:
   - The input image $x \in \mathbb{R}^{H \times W \times C}$ is divided into $N$ non-overlapping patches $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$
   - Typically, patches are of size $P \times P$ (e.g., 16×16 pixels)
   - Each patch is flattened and linearly projected to a $D$-dimensional embedding space: $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$

2. **Sequence Construction**:
   - A learnable classification token $x_{class} \in \mathbb{R}^D$ is prepended to the sequence
   - Position embeddings $E_{pos} \in \mathbb{R}^{(N+1) \times D}$ are added to retain positional information
   - The resulting sequence is: $$z_0 = [x_{class}; x_p^1 E; x_p^2 E; ...; x_p^N E] + E_{pos}$$

3. **Transformer Encoder**:
   - The sequence is processed through $L$ Transformer encoder blocks
   - Each block contains:
     - Multi-head self-attention (MSA): $\text{MSA}(\text{LN}(z_{l-1}))$
     - Layer normalization (LN): $\text{LN}(z)$
     - MLP with GELU activation: $\text{MLP}(\text{LN}(z'))$
     - Residual connections: $z_l = \text{MLP}(\text{LN}(z')) + z'$ where $z' = \text{MSA}(\text{LN}(z_{l-1})) + z_{l-1}$

4. **Output Representation**:
   - For classification, the representation of the classification token from the final layer $z_L^0$ is used
   - For embedding generation, either the classification token or a pooled representation of all patch tokens can be used

##### Multi-Head Self-Attention in ViT

The self-attention mechanism in ViT follows the standard Transformer formulation:

1. **Query, Key, Value Projections**:
   - $Q = z W_Q$, $K = z W_K$, $V = z W_V$ where $W_Q, W_K, W_V \in \mathbb{R}^{D \times d_k}$

2. **Attention Calculation**:
   - $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

3. **Multi-Head Mechanism**:
   - $\text{MSA}(z) = [\text{head}_1; \text{head}_2; ...; \text{head}_h]W^O$
   - $\text{head}_i = \text{Attention}(zW_Q^i, zW_K^i, zW_V^i)$
   - $W^O \in \mathbb{R}^{(h \cdot d_k) \times D}$

##### ViT Variants and Improvements

1. **DeiT** (Data-efficient Image Transformer):
   - Introduced distillation token and teacher-student training
   - Enabled training on smaller datasets without extensive pre-training
   - Distillation loss: $L = \alpha L_{CE}(y_{cls}, y) + \beta L_{CE}(y_{dist}, y) + \gamma L_{KL}(y_{dist}, y_{teacher})$

2. **Swin Transformer**:
   - Hierarchical architecture with shifted windows
   - Computational complexity reduced from $O(N^2)$ to $O(N)$
   - Window-based self-attention: $\text{Attention}(Q_w, K_w, V_w)$ for each window $w$

3. **CvT** (Convolutional vision Transformer):
   - Incorporates convolutional projections for tokens
   - Combines strengths of CNNs and Transformers

4. **MViT** (Multiscale Vision Transformer):
   - Pooling-based dimension reduction across layers
   - Creates a pyramid of feature resolutions

5. **ViT-G** (Giant):
   - Scaled up to 2 billion parameters
   - Pre-trained on JFT-3B dataset
   - State-of-the-art performance on many benchmarks

##### Training Strategies for ViT

1. **Pre-training Approaches**:
   - Supervised pre-training on large labeled datasets (e.g., JFT-300M)
   - Self-supervised pre-training (e.g., DINO, MAE, BEiT)
   - Hybrid approaches combining different objectives

2. **Self-Supervised Learning for ViT**:
   - **DINO** (Self-Distillation with No Labels):
     - Uses a teacher-student architecture
     - Momentum encoder and multi-crop strategy
     - Loss: $L = -\sum_i p_t^i \log p_s^i$ where $p_t$ and $p_s$ are teacher and student probability distributions

   - **MAE** (Masked Autoencoders):
     - Randomly masks a high proportion of image patches (e.g., 75%)
     - Reconstructs the masked patches using a lightweight decoder
     - Loss: $L = \frac{1}{|M|} \sum_{i \in M} ||x_i - \hat{x}_i||_2^2$ where $M$ is the set of masked patches

   - **BEiT** (BERT Pre-training of Image Transformers):
     - Predicts visual tokens from a discrete VAE instead of raw pixels
     - Adapts the MLM objective from BERT

3. **Fine-tuning Techniques**:
   - Layer-wise learning rate decay
   - Head regularization
   - Stochastic depth
   - Mixup and CutMix augmentations

##### Embedding Extraction from ViT

1. **CLS Token Embedding**:
   - Use the final layer representation of the classification token: $h_{CLS} = z_L^0$
   - Simple but effective for many tasks

2. **Mean Patch Embedding**:
   - Average the final layer representations of all patch tokens: $h_{mean} = \frac{1}{N} \sum_{i=1}^{N} z_L^i$
   - More comprehensive representation of the entire image

3. **Attention-Weighted Embedding**:
   - Weight patch tokens by their attention scores to the CLS token
   - $h_{att} = \sum_{i=1}^{N} \alpha_i z_L^i$ where $\alpha_i$ are attention weights

4. **Multi-layer Aggregation**:
   - Combine representations from multiple layers
   - $h_{multi} = \sum_{l=1}^{L} w_l \cdot \text{Pool}(z_l)$
   - Captures both low-level and high-level features

##### Applications of ViT Embeddings

1. **Image Retrieval**:
   - DINO embeddings show strong performance for instance-level retrieval
   - Self-supervised ViT embeddings capture semantic similarities effectively

2. **Zero-shot Transfer**:
   - ViT embeddings generalize well to unseen domains and tasks
   - Particularly effective when pre-trained on diverse, large-scale datasets

3. **Visual Localization**:
   - Attention maps from ViT can localize objects without explicit supervision
   - Useful for weakly supervised object detection

4. **Image Segmentation**:
   - Patch-level embeddings can be used for semantic segmentation
   - Self-attention maps provide object boundary information

5. **Cross-modal Applications**:
   - ViT embeddings can be aligned with text embeddings (as in CLIP)
   - Enables text-to-image retrieval and generation

##### Advantages and Limitations

**Advantages**:
- Global receptive field from the first layer
- Strong scaling properties with model and data size
- Flexibility in handling variable input resolutions
- State-of-the-art performance when properly trained

**Limitations**:
- Quadratic complexity with respect to sequence length
- Data hunger (requires more training data than CNNs)
- Positional encoding limitations for very high resolutions
- Computationally intensive training

**Key Papers**:
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020)
- [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) (Touvron et al., 2021)
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) (Liu et al., 2021)
- [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294) (Caron et al., 2021)
- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) (He et al., 2021)

#### CLIP: Contrastive Language-Image Pre-training (2021-present)

CLIP (Contrastive Language-Image Pre-training) represents a breakthrough in multimodal learning by aligning visual and textual representations in a shared embedding space through contrastive learning at scale. This approach enables remarkable zero-shot capabilities and has become a foundation for numerous downstream applications.

##### CLIP Architecture

CLIP consists of two parallel encoders:

1. **Image Encoder**:
   - Can be either a CNN (ResNet) or a Vision Transformer (ViT)
   - Processes an image $I$ to produce an image embedding $i = E_I(I) \in \mathbb{R}^d$
   - The embedding is L2-normalized: $\hat{i} = i / \|i\|_2$
   - ViT variants generally outperform ResNet variants

2. **Text Encoder**:
   - Transformer-based architecture similar to GPT
   - Processes text $T$ to produce a text embedding $t = E_T(T) \in \mathbb{R}^d$
   - The embedding is L2-normalized: $\hat{t} = t / \|t\|_2$
   - Uses causal attention masks but takes the final token's representation

3. **Projection Layers**:
   - Both encoders include a final linear projection layer to map to the shared embedding space
   - These projections align the dimensionality and distribution of the embeddings

##### Training Methodology

1. **Contrastive Learning Objective**:
   - CLIP uses a symmetric cross-entropy loss over cosine similarities
   - For a batch of $N$ (image, text) pairs, the loss is:

   $$L = \frac{1}{2}\left(L_{i\rightarrow t} + L_{t\rightarrow i}\right)$$

   where:
   
   $$L_{i\rightarrow t} = -\frac{1}{N}\sum_{m=1}^{N} \log \frac{\exp(\text{sim}(i_m, t_m)/\tau)}{\sum_{n=1}^N \exp(\text{sim}(i_m, t_n)/\tau)}$$
   
   $$L_{t\rightarrow i} = -\frac{1}{N}\sum_{m=1}^{N} \log \frac{\exp(\text{sim}(t_m, i_m)/\tau)}{\sum_{n=1}^N \exp(\text{sim}(t_m, i_n)/\tau)}$$

   - $\text{sim}(i, t) = i^T t$ is the cosine similarity between normalized embeddings
   - $\tau$ is a learnable temperature parameter that scales the logits

2. **Training Data**:
   - 400 million (image, text) pairs collected from the internet
   - Minimal filtering for English text and image dimensions
   - No human annotation or curation
   - Wide diversity of concepts, styles, and domains

3. **Training Process**:
   - Trained from scratch (no pre-training)
   - Adam optimizer with decoupled weight decay
   - Cosine learning rate schedule with warmup
   - Mixed-precision training
   - Large batch sizes (32,768 pairs)

##### CLIP Variants and Scaling

1. **Model Scales**:
   - ResNet variants: ResNet-50, ResNet-101, ResNet-50×4, ResNet-50×16, ResNet-50×64
   - ViT variants: ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px
   - Largest model has 428 million parameters

2. **Improved Variants**:
   - **OpenCLIP**: Open-source implementation with additional training on LAION datasets
   - **CLIP-ViT-H**: Larger model with ViT-H/14 architecture
   - **DeCLIP**: Adds self-supervised objectives to improve with less data
   - **SLIP**: Combines contrastive language-image pre-training with self-supervised learning
   - **EVA-CLIP**: Enhanced visual representation with masked image modeling

3. **Efficiency Improvements**:
   - **LiT** (Locked-image Tuning): Freezes pre-trained image encoder and only trains text encoder
   - **FLAVA**: Unified foundation model for joint vision-and-language understanding

##### Embedding Properties and Extraction

1. **Embedding Dimensionality**:
   - Typically 512 or 768 dimensions depending on model size
   - Embeddings are L2-normalized to lie on a unit hypersphere

2. **Extraction Methods**:
   - **Image Embeddings**: Forward pass through image encoder + projection
   - **Text Embeddings**: Forward pass through text encoder + projection
   - Both can be used independently for unimodal tasks

3. **Embedding Properties**:
   - Semantic alignment between modalities
   - Compositional understanding (e.g., "a red cube on a blue sphere")
   - Robust to distribution shifts
   - Captures both fine-grained and abstract concepts

##### Zero-Shot Capabilities

1. **Classification**:
   - Construct text prompts for each class (e.g., "a photo of a {class}")
   - Encode each prompt with the text encoder
   - Encode the query image with the image encoder
   - Predict the class with highest cosine similarity

2. **Prompt Engineering**:
   - Performance can be significantly improved with better prompts
   - Ensemble of prompts (e.g., "a photo of a {class}", "a picture of a {class}", etc.)
   - Context-specific prompts (e.g., "a satellite image of a {class}")

3. **Few-Shot Learning**:
   - CLIP embeddings can be used as features for linear probing
   - Requires significantly fewer examples than traditional approaches

##### Applications of CLIP Embeddings

1. **Cross-Modal Retrieval**:
   - Text-to-image search: Find images matching a text description
   - Image-to-text search: Generate captions or find relevant text
   - Enables semantic search beyond keyword matching

2. **Zero-Shot Recognition**:
   - Object classification without task-specific training
   - Domain adaptation across visual distributions
   - Out-of-distribution detection

3. **Content Creation**:
   - Guidance for text-to-image generation models (DALL-E, Stable Diffusion)
   - Image editing through textual directions
   - Style transfer based on textual descriptions

4. **Multimodal Understanding**:
   - Visual question answering
   - Image captioning
   - Visual reasoning

5. **Representation Learning**:
   - Foundation for fine-tuning on downstream tasks
   - Transfer learning to specialized domains
   - Feature extraction for classical ML pipelines

##### Limitations and Challenges

1. **Biases**:
   - Reflects and potentially amplifies biases in internet data
   - Social biases (gender, race, etc.) are encoded in the embeddings
   - Geographical and cultural biases due to data distribution

2. **Reasoning Limitations**:
   - Limited understanding of spatial relationships
   - Struggles with counting and numerical reasoning
   - Difficulty with fine-grained visual details

3. **Computational Requirements**:
   - Large models require significant compute for training
   - Inference can be resource-intensive for real-time applications

4. **Domain Gaps**:
   - Performance drops on specialized domains (medical, scientific, etc.)
   - May require domain-specific fine-tuning

##### Implementation Considerations

1. **Prompt Design**:
   - Critical for optimal performance
   - Domain-specific prompts often work better
   - Ensembling multiple prompts improves robustness

2. **Embedding Caching**:
   - Pre-compute embeddings for efficiency in retrieval systems
   - Approximate nearest neighbor search for large-scale applications

3. **Fine-tuning Strategies**:
   - Linear probing vs. full fine-tuning
   - Adapter layers for parameter-efficient tuning
   - Domain-specific contrastive tuning

**Key Papers and Resources**:
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (Radford et al., 2021)
- [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918) (Jia et al., 2021)
- [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991) (Zhai et al., 2022)
- [FLAVA: A Foundational Language And Vision Alignment Model](https://arxiv.org/abs/2112.04482) (Singh et al., 2022)
- [EVA-CLIP: Improved Training Techniques for CLIP at Scale](https://arxiv.org/abs/2303.15389) (Sun et al., 2023)

### Audio Embeddings

#### Wav2Vec and Wav2Vec 2.0

Wav2Vec learns representations from raw audio by solving a contrastive task that requires distinguishing true future audio samples from distractors. Wav2Vec 2.0 extends this with a masked prediction task similar to BERT's MLM.

The contrastive loss in Wav2Vec 2.0 is:

$$L_c = -\log \frac{\exp(\text{sim}(c_t, q_t)/\kappa)}{\sum_{\tilde{t} \in \{t\} \cup N_t} \exp(\text{sim}(c_{\tilde{t}}, q_t)/\kappa)}$$

where $c_t$ is the true quantized latent speech representation, $q_t$ is the context network output, and $N_t$ is a set of distractors.

**Key Papers**:
- [wav2vec: Unsupervised Pre-training for Speech Recognition](https://arxiv.org/abs/1904.05862) (Schneider et al., 2019)
- [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) (Baevski et al., 2020)

#### Whisper

Whisper is a robust speech recognition system trained on a large and diverse dataset of audio-text pairs. It uses a sequence-to-sequence Transformer architecture with an encoder-decoder design:

1. The encoder processes the audio spectrograms
2. The decoder generates text transcriptions autoregressively

Whisper's encoder uses a convolutional frontend to process the mel spectrogram before the Transformer layers:

$$X_0 = \text{Conv2d}(\text{MelSpectrogram}(\text{audio}))$$

Followed by Transformer encoder layers:

$$X_{l+1} = X_l + \text{Attention}(\text{LayerNorm}(X_l)) + \text{FFN}(\text{LayerNorm}(X_l + \text{Attention}(\text{LayerNorm}(X_l))))$$

**Key Paper**: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) (Radford et al., 2022)

#### HuBERT and WavLM

HuBERT (Hidden-Unit BERT) applies masked prediction to audio by first clustering the continuous speech signal into discrete units. WavLM extends HuBERT with denoising and speaker disentanglement objectives.

The HuBERT pre-training objective is:

$$L = \sum_{t \in M} \log p(c_t | \tilde{X})$$

where $M$ is the set of masked indices, $c_t$ is the cluster assignment of the true frame, and $\tilde{X}$ is the masked input sequence.

**Key Papers**:
- [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447) (Hsu et al., 2021)
- [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) (Chen et al., 2021)

### Multimodal Embeddings

Multimodal embeddings aim to create unified representations across different modalities (text, image, audio). The key challenge is aligning these diverse modalities in a shared semantic space.

#### Joint Embedding Space Models

These models project different modalities into a common embedding space where semantically similar content is positioned closely regardless of modality.

The alignment objective often uses contrastive learning:

$$L = \sum_{i=1}^N \sum_{j=1}^N -y_{ij} \log \frac{\exp(\text{sim}(x_i, x_j)/\tau)}{\sum_{k=1}^N \exp(\text{sim}(x_i, x_k)/\tau)}$$

where $y_{ij} = 1$ if $x_i$ and $x_j$ are semantically related across modalities, and 0 otherwise.

#### Multimodal Transformers

Models like CLIP, ALIGN, and FLAVA use separate encoders for different modalities followed by alignment layers. More recent approaches like Flamingo and GPT-4 integrate multiple modalities more deeply within a single architecture.

The cross-attention mechanism often used in these models is:

$$\text{CrossAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q$ comes from one modality and $K, V$ from another.

**Key Papers**:
- [FLAVA: A Foundational Language And Vision Alignment Model](https://arxiv.org/abs/2112.04482) (Singh et al., 2022)
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) (Alayrac et al., 2022)
- [ImageBind: One Embedding Space To Bind Them All](https://arxiv.org/abs/2305.05665) (Girdhar et al., 2023)

## Features

- **Multiple Frameworks**: Support for various embedding frameworks including SentenceTransformers, OpenAI, Google Gemini, CLIP, Wav2Vec2, Whisper, and more.
- **Modality Support**: Text, image, audio, and multimodal embedding capabilities with a consistent interface.
- **Unified Interface**: Consistent API across different frameworks and modalities.
- **Dynamic Framework Detection**: Automatically detects available frameworks based on installed packages.
- **Batch Processing**: Efficient batch embedding generation for multiple inputs.
- **Similarity Calculation**: Built-in methods for calculating cosine similarity between embeddings.

## Supported Frameworks

### Text Embedding Frameworks

- **SentenceTransformers**: High-quality text embeddings using Hugging Face models
- **OpenAI**: State-of-the-art embeddings via OpenAI's API
- **Google Gemini**: Google's embedding models
- **Jina**: Jina AI's embedding models
- **NVIDIA NeMo**: NVIDIA's NV-Embed models
- **Stella**: Stella AI's embedding models
- **ModernBERT**: Modern BERT-based embedding models
- **Cohere**: Cohere's embedding models
- **HuggingFace**: Direct access to Hugging Face's embedding models

### Image Embedding Frameworks

- **CLIP**: OpenAI's CLIP models for image embeddings
- **OpenAI**: OpenAI's image embedding API
- **Google Gemini**: Google's multimodal embedding models
- **PyTorch Image Models (timm)**: Various image models from the timm library
- **Vision Transformer (ViT)**: Transformer-based image embedding models
- **ResNet**: ResNet-based image embedding models

### Audio Embedding Frameworks

- **Wav2Vec2**: Facebook AI's self-supervised speech representation models
- **Whisper**: OpenAI's speech recognition and transcription models
- **HuBERT**: Facebook AI's self-supervised speech representation models
- **WavLM**: Microsoft's state-of-the-art speech representation model
- **Data2Vec**: Facebook AI's multi-modal self-supervised model
- **OpenAI**: OpenAI's audio embedding API
- **Google Gemini**: Google's multimodal embedding models

## Installation

The core module has minimal dependencies, but each framework requires its own dependencies to be installed.

```bash
# Core dependencies
pip install numpy pillow matplotlib

# SentenceTransformers
pip install sentence-transformers

# OpenAI
pip install openai

# Google Gemini
pip install google-generativeai

# CLIP
pip install ftfy regex tqdm git+https://github.com/openai/CLIP.git

# PyTorch Image Models
pip install timm

# Vision Transformer
pip install transformers

# ResNet
pip install torch torchvision

# Audio dependencies
pip install torchaudio librosa soundfile

# Wav2Vec2, Whisper, HuBERT, WavLM, Data2Vec
pip install transformers
```

## Usage

### Text Embedding

```python
from llm_multi_core.embedder import create_text_embedder

# Create a text embedder with SentenceTransformers
embedder = create_text_embedder(framework="sentence-transformers")

# Generate embedding for a single text
embedding = embedder.embed("Hello, world!")

# Generate embeddings for multiple texts
texts = ["Hello, world!", "How are you?"]
embeddings = embedder.embed_batch(texts)

# Calculate similarity between two texts
similarity = embedder.similarity("Hello, world!", "Hi, world!")
print(f"Similarity: {similarity}")
```

### Image Embedding

```python
from llm_multi_core.embedder import create_image_embedder
from PIL import Image

# Create an image embedder with CLIP
embedder = create_image_embedder(framework="clip")

# Generate embedding for a single image
image = Image.open("image.jpg")
embedding = embedder.embed(image)

# Generate embeddings for multiple images
images = [Image.open(f"image_{i}.jpg") for i in range(3)]
embeddings = embedder.embed_batch(images)

# Calculate similarity between two images
similarity = embedder.similarity("image1.jpg", "image2.jpg")
print(f"Similarity: {similarity}")
```

### Audio Embedding

```python
from llm_multi_core.embedder import create_audio_embedder
import librosa

# Create an audio embedder with Wav2Vec2
embedder = create_audio_embedder(framework="wav2vec2")

# Generate embedding for a single audio file
audio, sr = librosa.load("audio.wav", sr=16000)
embedding = embedder.embed(audio)

# Generate embeddings for multiple audio files
audio_files = [f"audio_{i}.wav" for i in range(3)]
audio_data = [librosa.load(file, sr=16000)[0] for file in audio_files]
embeddings = embedder.embed_batch(audio_data)

# Calculate similarity between two audio files
similarity = embedder.similarity("audio1.wav", "audio2.wav")
print(f"Similarity: {similarity}")
```

### Multimodal Embedding

```python
from llm_multi_core.embedder import create_multimodal_embedder
from PIL import Image
import librosa

# Create a multimodal embedder
embedder = create_multimodal_embedder(
    text_framework="sentence-transformers",
    image_framework="clip",
    audio_framework="wav2vec2"
)

# Generate embeddings for mixed inputs
inputs = [
    "A beautiful sunset",  # Text
    Image.open("sunset.jpg"),  # Image
    "A cute puppy",  # Text
    Image.open("puppy.jpg"),  # Image
    librosa.load("bird_chirping.wav", sr=16000)[0]  # Audio
]

embeddings = embedder.embed_batch(inputs)

# Calculate similarity between different modalities
similarity_text_image = embedder.similarity("A beautiful sunset", "sunset.jpg")
print(f"Text-Image Similarity: {similarity_text_image}")

similarity_image_audio = embedder.similarity("sunset.jpg", "bird_chirping.wav")
print(f"Image-Audio Similarity: {similarity_image_audio}")

similarity_text_audio = embedder.similarity("Bird sounds", "bird_chirping.wav")
print(f"Text-Audio Similarity: {similarity_text_audio}")
```

### Checking Available Frameworks

```python
from llm_multi_core.embedder import get_available_embedders

# Get available frameworks for all modalities
available = get_available_embedders()

# Print available text frameworks
print("Available Text Frameworks:")
for framework, available in available["text"].items():
    status = "Available" if available else "Not available"
    print(f"  - {framework}: {status}")

# Print available image frameworks
print("\nAvailable Image Frameworks:")
for framework, available in available["image"].items():
    status = "Available" if available else "Not available"
    print(f"  - {framework}: {status}")

# Print available audio frameworks
print("\nAvailable Audio Frameworks:")
for framework, available in available["audio"].items():
    status = "Available" if available else "Not available"
    print(f"  - {framework}: {status}")
```

## Examples

See the `examples.py` file for complete examples of using the embedder module with different frameworks and modalities.

## Practical Applications of Embeddings

### Information Retrieval and Search

Embeddings enable semantic search beyond keyword matching. Documents and queries are embedded in the same vector space, allowing retrieval based on semantic similarity rather than lexical overlap.

The retrieval process typically involves:

1. Offline indexing: Embed all documents in a collection
2. Query processing: Embed the user query
3. Similarity search: Find documents with embeddings closest to the query embedding

The similarity score between query $q$ and document $d$ is often computed as:

$$\text{score}(q, d) = \frac{\vec{q} \cdot \vec{d}}{||\vec{q}|| \cdot ||\vec{d}||}$$

### Recommendation Systems

Embeddings can represent users and items in a shared space, enabling content-based and collaborative filtering approaches. The recommendation score is often the dot product of user and item embeddings:

$$\text{score}(u, i) = \vec{u} \cdot \vec{i}$$

### Clustering and Classification

Embeddings transform raw data into a space where traditional distance-based algorithms can capture semantic relationships. For clustering, algorithms like K-means can be applied directly to embeddings:

$$\text{cluster}_k = \arg\min_{\mu_k} \sum_{x_i \in S_k} ||x_i - \mu_k||^2$$

where $S_k$ is the set of points in cluster $k$ and $\mu_k$ is the centroid.

### Cross-Modal Retrieval

Multimodal embeddings enable searching across modalities, such as finding images based on text descriptions or retrieving audio clips that match a textual query.

### Zero-Shot Learning

Models like CLIP enable classifying images into arbitrary categories without specific training examples, by comparing image embeddings with text embeddings of class names.

## Architecture

The embedder module is organized into the following components:

- **BaseEmbedder**: Abstract base class defining the common interface for all embedders.
- **TextEmbedder**: Implementation for text embedding using various frameworks.
- **ImageEmbedder**: Implementation for image embedding using various frameworks.
- **AudioEmbedder**: Implementation for audio embedding using various frameworks.
- **MultiModalEmbedder**: Implementation for multimodal embedding, combining text, image, and audio embedders.

## Evaluating Embedding Quality

Assessing the quality of embeddings is crucial for both research and practical applications. Different evaluation methods are appropriate for different modalities and use cases.

### Intrinsic Evaluation

Intrinsic evaluation measures how well embeddings capture semantic relationships without considering downstream tasks.

#### Word Similarity and Relatedness

For word embeddings, standard benchmarks include:

- **WordSim-353**: Measures correlation between human similarity judgments and cosine similarity of word embeddings
- **SimLex-999**: Focuses on similarity rather than relatedness
- **MEN**: Contains 3,000 word pairs with human-assigned similarity scores

The evaluation metric is typically Spearman's rank correlation coefficient:

$$\rho = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$$

where $d_i$ is the difference between the ranks of corresponding values and $n$ is the number of pairs.

#### Analogy Tasks

Analogy tasks evaluate whether embeddings capture relational similarities, such as "man is to woman as king is to queen."

The accuracy is calculated as:

$$\text{Accuracy} = \frac{\text{Number of correctly solved analogies}}{\text{Total number of analogies}}$$

#### Clustering and Visualization

Techniques like t-SNE and UMAP can visualize embeddings in 2D or 3D space, allowing qualitative assessment of how well semantically similar items cluster together.

### Extrinsic Evaluation

Extrinsic evaluation measures how well embeddings perform on downstream tasks.

#### Text Classification

Embeddings are used as features for classifiers, with performance measured using metrics like accuracy, F1-score, and AUC:

$$F1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

#### Information Retrieval

Embeddings are evaluated on retrieval tasks using metrics like Mean Average Precision (MAP) and Normalized Discounted Cumulative Gain (NDCG):

$$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

where:

$$\text{DCG@k} = \sum_{i=1}^{k} \frac{\text{rel}_i}{\log_2(i+1)}$$

#### Cross-Modal Retrieval

For multimodal embeddings, evaluation often involves retrieving items of one modality given a query in another modality (e.g., text-to-image retrieval). Metrics include Recall@K and Median Rank.

### Benchmarks for Modern Embeddings

- **MTEB (Massive Text Embedding Benchmark)**: Evaluates text embeddings across 56 datasets spanning classification, clustering, retrieval, and more
- **BEIR (Benchmarking IR)**: Focuses on zero-shot information retrieval across diverse domains
- **CLIP Score**: Measures alignment between images and text in multimodal models
- **ImageNet**: Standard benchmark for image embeddings
- **SUPERB (Speech processing Universal PERformance Benchmark)**: Evaluates speech representations across various tasks

## Future Directions in Embedding Research

The field of embeddings continues to evolve rapidly. Here are some promising research directions:

### Multimodal Foundation Models

Models that can seamlessly process and align multiple modalities (text, image, audio, video, 3D) in a single architecture are becoming increasingly important. Research is focusing on:

- **Cross-modal transfer learning**: Leveraging knowledge from one modality to improve representations in another
- **Unified representation spaces**: Creating embedding spaces that maintain semantic relationships across all modalities
- **Emergent capabilities**: Understanding how multimodal training leads to capabilities not present in single-modality models

### Efficiency and Compression

As embedding models grow larger, research on making them more efficient becomes crucial:

- **Distillation**: Transferring knowledge from large teacher models to smaller student models
- **Quantization**: Reducing the precision of model weights (e.g., from 32-bit to 8-bit or 4-bit)
- **Pruning**: Removing less important weights or neurons from models
- **Sparse representations**: Using embeddings where most dimensions are zero

### Interpretability and Fairness

Understanding what information is encoded in embeddings and ensuring they are fair and unbiased:

- **Probing tasks**: Designing experiments to determine what linguistic or visual concepts are captured in embeddings
- **Debiasing techniques**: Methods to remove unwanted social biases from embeddings
- **Causal analysis**: Understanding how embeddings relate to causal factors in the data

### Compositional and Hierarchical Embeddings

Developing embeddings that better capture compositional structure:

- **Hierarchical representations**: Embeddings that represent information at multiple levels of abstraction
- **Compositional generalization**: Creating embeddings that generalize to novel combinations of familiar concepts
- **Structured representations**: Incorporating explicit structure (e.g., graphs, trees) into embedding spaces

### Continual Learning and Adaptation

Enabling embedding models to adapt to new data and tasks without forgetting:

- **Parameter-efficient fine-tuning**: Methods like LoRA, adapters, and prompt tuning
- **Rehearsal mechanisms**: Techniques to prevent catastrophic forgetting
- **Meta-learning**: Learning to learn, enabling rapid adaptation to new tasks
