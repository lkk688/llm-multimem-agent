# The Rise of Self-Supervised Learning: From Word2Vec to Multimodal Foundation Models

## 1. Introduction

Self-Supervised Learning (SSL) has transformed the way we approach representation learning. Instead of relying on **manually labeled datasets**—which are expensive and time-consuming to create—SSL methods construct *pretext tasks* where the supervision signal is derived automatically from the input data itself.

**Core idea**:  
> Predict parts of the data from other parts of the data.

This paradigm:
- **Scales naturally** with massive unlabeled corpora (text, audio, images, video).
- **Captures richer structures** than task-specific supervised models.
- **Enables transfer learning** across domains with minimal fine-tuning.

---

## 2. Historical Evolution of SSL

### 2.1 Distributional Semantics and Word2Vec

**Background**:  
Prior to deep learning, word embeddings were often derived from **count-based methods** (e.g., Latent Semantic Analysis). Word2Vec (Mikolov et al., 2013) introduced efficient **predictive models** for word vectors.

Two architectures:
- **Skip-gram**: Predict context given a target word.
- **CBOW**: Predict target given its context.

**Skip-gram Objective**:

\[
\mathcal{L}_{\text{SG}} = \frac{1}{T} \sum_{t=1}^T \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} \mid w_t)
\]

With **negative sampling**:

\[
\log \sigma(\mathbf{v}'_{w_o} \cdot \mathbf{v}_{w_i}) + \sum_{k=1}^K \mathbb{E}_{w_k \sim P_n(w)} [\log \sigma(-\mathbf{v}'_{w_k} \cdot \mathbf{v}_{w_i})]
\]

**Impact**:
- Moved from **sparse vectors** to dense continuous embeddings.
- Inspired later *contextualized* embeddings (ELMo, GPT, BERT).

---

### 2.2 GPT — Causal Language Modeling

**Key insight**: Treat **next-token prediction** as SSL.  
Given sequence \( w_1, w_2, ..., w_T \), maximize:

\[
\mathcal{L}_{\text{CLM}} = - \sum_{t=1}^T \log P_\theta(w_t \mid w_{<t})
\]

**Characteristics**:
- Unidirectional context (left-to-right).
- Captures long-range dependencies using the Transformer architecture.
- Pretraining → Fine-tuning paradigm.

**Historical notes**:
- GPT-1 (2018): First to show that large-scale language modeling transfers well.
- GPT-2: Demonstrated zero-shot generalization.
- GPT-3: Scaling laws emerged (Brown et al., 2020).

---

### 2.3 BERT — Masked Language Modeling (MLM)

BERT’s **bidirectional** pretraining captures full sentence context.

**MLM Objective**:

\[
\mathcal{L}_{\text{MLM}} = - \sum_{i \in \mathcal{M}} \log P_\theta(w_i \mid \mathbf{w}_{\setminus i})
\]

Where \(\mathcal{M}\) is the set of masked tokens (typically 15% of input).  
Also used **Next Sentence Prediction (NSP)** to learn inter-sentence relationships.

**Advantages**:
- Encodes both left and right context.
- Stronger performance in sentence-level understanding tasks (GLUE, SQuAD).

**Limitations**:
- Pretrain-finetune mismatch (masking not present at inference).
- Expensive to train for large vocabularies.

---

### 2.4 Beyond GPT and BERT — Unified Pretraining

Recent variants adapt masking and autoregressive tasks:
- **T5** (Raffel et al., 2020): Text-to-text span corruption.
- **UniLM** (Dong et al., 2019): Unified LM for uni/bidirectional tasks.
- **Instruction-tuned LLMs** (InstructGPT, FLAN, Alpaca): Aligning LLMs with human preferences.
- **Multimodal extensions**: Integrating vision and language (PaLI, Flamingo, GPT-4V).

---

## 3. Modality-Specific SSL

### 3.1 Audio — Wav2Vec and HuBERT

#### Wav2Vec 2.0
Steps:
1. Raw waveform → Conv feature encoder → Latent representation \( z_t \).
2. Random masking in latent space.
3. Transformer context network → context vector \( c_t \).
4. Predict quantized target representation \( q_t \) among distractors.

**Contrastive loss**:

\[
\mathcal{L}_{\text{contrast}} = - \log \frac{\exp(\text{sim}(c_t, q_t) / \kappa)}{\sum_{\tilde{q} \in \mathcal{Q}_t} \exp(\text{sim}(c_t, \tilde{q}) / \kappa)}
\]

**Key benefits**:
- Large gains on speech recognition with <10h labeled data.
- Structure emerges: phoneme-level information without labels.

#### HuBERT
Uses *iterative pseudo-labeling*:
1. Cluster acoustic features.
2. Predict cluster IDs with a Transformer (masked prediction).
3. Recompute clusters using new embeddings.
4. Repeat until convergence.

---

### 3.2 Vision — Contrastive and Masked Pretraining

#### Contrastive Learning (SimCLR, MoCo)
- Augment image twice → positive pair.
- Other images → negatives.
- **NT-Xent Loss**:

\[
\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}
\]

#### Masked Image Modeling (MAE)
- Randomly mask 75% patches.
- Encode visible patches → reconstruct masked patches.
- Improves data efficiency, complements supervised pretraining.

---

### 3.3 Multimodal — Aligning Modalities

#### CLIP
- Dual encoder: \( f_I(\text{image}), f_T(\text{text}) \).
- Align embeddings with **symmetric contrastive loss**.

\[
\mathcal{L}_{\text{CLIP}} = \frac{1}{2} (\mathcal{L}_{I \to T} + \mathcal{L}_{T \to I})
\]

**Implications**:
- Enables zero-shot classification: *"Which text matches this image?"*
- Generalizes across domains with minimal adaptation.

---

### 3.4 Latest Vision-Language Models

| Model       | Core Idea | Key Innovation |
|-------------|-----------|----------------|
| Flamingo    | Frozen LM + cross-attn vision module | Few-shot multimodal |
| BLIP-2      | Q-former bridging frozen vision & LLM | Efficient training |
| PaLI        | Large unified multilingual multimodal | Multi-task learning |
| GPT-4V      | Vision-augmented LLM | Unified reasoning |

---

## 4. Research Insights

### 4.1 Why SSL Works
- **Information bottleneck**: SSL forces the model to encode only task-relevant information.
- **Inductive bias**: Pretext task defines the structure of learned representations.
- **Data priors**: Leveraging massive unlabeled datasets injects domain knowledge implicitly.

### 4.2 Scaling Laws
Kaplan et al. (2020) showed that loss scales predictably with:
\[
L(N, D, P) \approx L_\infty + a N^{-\alpha_N} + b D^{-\alpha_D} + c P^{-\alpha_P}
\]
where \(N\): model size, \(D\): dataset size, \(P\): compute budget.

### 4.3 Open Challenges
- **Negative sampling strategies** for contrastive methods.
- **Multimodal alignment drift** when training scales beyond paired data.
- **Bias and fairness**: SSL may encode harmful correlations present in data.
- **Energy efficiency**: SSL models are compute-hungry.

---

## 5. Practical Tips for Researchers

1. **Start small**: Pretrain on a subset before scaling to billions of samples.
2. **Modality-specific augmentations**: Critical for contrastive learning success.
3. **Mixed objectives**: Combining MLM + contrastive can improve robustness.
4. **Evaluation suite**: Use diverse downstream tasks to measure generalization.

---

## 6. Key References

1. Mikolov, T., et al. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv:1301.3781.  
2. Devlin, J., et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805.  
3. Radford, A., et al. (2018). *Improving Language Understanding by Generative Pre-Training*. OpenAI.  
4. Brown, T., et al. (2020). *Language Models are Few-Shot Learners*. NeurIPS.  
5. Baevski, A., et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*. NeurIPS.  
6. Hsu, W.-N., et al. (2021). *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units*. arXiv:2106.07447.  
7. Chen, T., et al. (2020). *A Simple Framework for Contrastive Learning of Visual Representations*. ICML.  
8. He, K., et al. (2022). *Masked Autoencoders Are Scalable Vision Learners*. CVPR.  
9. Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML.  
10. Li, J., et al. (2023). *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*. arXiv:2301.12597.

---