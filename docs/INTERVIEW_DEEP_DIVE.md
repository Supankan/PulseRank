# PulseRank: Production Recommender System & Content Intelligence Deep Dive

This document details the system design, mathematical foundations, architectural decisions, and trade-offs underpinning **PulseRank** ? a production-style personalized news recommendation and content intelligence platform built around the Microsoft News Dataset (MIND).

---

## 1. End-to-End Multi-Stage Recommendation Architecture

*Industry Design Target*: Modern large-scale recommender systems target sub-50ms latency budgets across millions of catalog items. In local Python/CPU evaluation, PulseRank instruments and logs measured per-request latency via `time.perf_counter()`. A single monolithic deep network cannot evaluate millions of candidate items in real time. PulseRank implements the canonical 4-stage funnel:

```
[Full Catalog: 50k+ Articles]
              │
  Stage 1: Multi-Channel Candidate Retrieval (MIPS & Inverted Index)
  ├── Semantic Content Bi-Encoder (FAISS IndexFlatIP cosine similarity) -> Top-50
  ├── Implicit ALS Matrix Factorization (Latent collaborative factors) -> Top-50
  └── Recency-Decayed Popularity (Half-life decaying Poisson baseline) -> Top-50
              │
  Stage 2: Candidate Fusion & Deduplication
  ├── Reciprocal Rank Fusion (RRF) for scale-invariant channel combination
  └── Multi-channel consensus bonus
              │
  Stage 3: Pointwise Learning-to-Rank (LTR)
  └── LightGBM LambdaMART (7 canonical features: semantic similarity, ALS score,
      popularity score, category affinity, subcategory affinity, entity overlap, history length)
              │
  Stage 4: Post-Processing & Policy Guardrails
  ├── Maximal Marginal Relevance (MMR) for Intra-List Diversity (ILD)
  ├── Epsilon-Greedy / UCB Exploration for cold and tail item serendipity
  └── Pairwise Cross-Encoder Precision Reranking (optional top-10 tier)
              │
[Served User Feed: Top-10 Slate]
```

### Key Architectural Rationale:
- **Retrieval vs Ranking separation**: High-recall candidate generation narrows the full catalog down to a compact candidate pool (~150 candidates). The multi-channel retrieval tier leverages vector indexing and matrix factorization for efficient nearest-neighbor search, allowing the heavier pointwise gradient boosting decision tree (GBDT) and optional pairwise cross-encoders to evaluate only those retrieved candidates.
- **RRF vs Pointwise Ensemble**: Pointwise linear blending is vulnerable to extreme score calibrations across channels (e.g. ALS raw logits vs cosine similarity in [0, 1]). Reciprocal Rank Fusion (RRF) relies strictly on ordinal rank positions:
  $$\text{RRF}(d) = \sum_{c \in \text{channels}} \frac{w_c}{k_{\text{rrf}} + \text{rank}_c(d)}$$
  which guarantees numerical robustness and order invariance.

---

## 2. Representation Learning: Bi-Encoder vs. Cross-Encoder Trade-offs

| Dimension | Bi-Encoder (Sentence-Transformers) | Pairwise Cross-Encoder |
| :--- | :--- | :--- |
| **Architecture** | Dual-tower: independently encodes user context and candidate | Single-tower: jointly attends across $(u, d)$ tokens via full cross-attention |
| **Complexity** | Offline indexing $O(N)$, online scoring $O(K \cdot d)$ inner product | Online joint encoding $O(K \cdot L^2)$ transformer forward passes |
| **MIPS Capable** | Yes (FAISS supports efficient vector similarity retrieval suitable for first-stage candidate generation) | No (requires runtime pair forward passes) |
| **Relevance Signal** | Dense cosine vector similarity | Token-level cross-interaction & fine-grained semantic nuance |
| **Serving Placement** | First-stage Retrieval (Top-50 candidates) | Optional second-stage reranking on Top-10 slate |

---

## 3. MIND Benchmark & Recommender Science Nuances

### 3.1 Impression-Level Evaluation vs. Pointwise Accuracy
In news recommendation, user feedback occurs in discrete impression sessions containing a list of news headlines displayed simultaneously:
$$\text{Impression } b = (u, \mathcal{H}_u, [d_1:y_1, d_2:y_2, \dots, d_m:y_m])$$
Evaluating pointwise ROC-AUC across pooled impressions is **flawed** because user engagement baselines vary widely. PulseRank enforces strict impression-level metrics:
- **Impression AUC**: Evaluates intra-impression ranking discrimination:
  $$\text{AUC}_{\text{imp}} = \frac{\sum_{i \in \text{pos}} \sum_{j \in \text{neg}} \mathbb{I}(s_i > s_j)}{|\text{pos}| \cdot |\text{neg}|}$$
- **Mean Reciprocal Rank (MRR)**: Evaluates position of the first clicked article:
  $$\text{MRR} = \frac{1}{|B|} \sum_{b \in B} \frac{1}{\text{rank}(d^*_b)}$$
- **Normalized Discounted Cumulative Gain (nDCG@K)**: Measures ranking quality with logarithmic position discounting:
  $$\text{DCG}@K = \sum_{i=1}^K \frac{2^{y_i} - 1}{\log_2(i + 1)}, \quad \text{nDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

### 3.2 Preventing Train/Serve Skew and Temporal Contamination
1. **Zero Fake Freshness**: Articles lack synthetic publish timestamps. Freshness features that do not exist in production or training logs create severe distribution skew.
2. **Impression-Aware Sequence**: Catalog retrieval evaluation must NOT pool all future validation clicks into a single target set per user, as this leaks future reading sessions. Evaluation evaluates $b.\mathcal{H} \to \text{Top-K} \cap b.\text{positives}$.
3. **ALS Deduplication**: Multiple behavior rows for the same user repeatedly duplicate the user's historical reading list. PulseRank aggregates unique interaction pairs $(u, d)$ before ALS training.

---

## 4. Mathematical Formulations & Metrics

### 4.1 Maximal Marginal Relevance (MMR)
Balances individual article relevance with slate diversity:
$$\text{MMR}(d) = \arg\max_{d \in R \setminus S} \left[ \lambda \cdot \text{Rel}(d, u) - (1 - \lambda) \max_{s \in S} \text{Sim}(d, s) \right]$$
Where $\lambda \in [0, 1]$ controls the accuracy-diversity trade-off frontier.

### 4.2 Intra-List Diversity (ILD)
Measures the average semantic pairwise distance across items in a recommendation slate $S$:
$$\text{ILD}(S) = \frac{2}{|S|(|S|-1)} \sum_{i < j} (1 - \cos(\mathbf{e}_i, \mathbf{e}_j))$$

### 4.3 Novelty (Mean Self-Information)
Quantifies unexpectedness by measuring negative log-likelihood of article global popularity:
$$\text{Novelty}(S) = \frac{1}{|S|} \sum_{d \in S} -\log_2 \left( \frac{\text{pop}(d)}{\sum_{j} \text{pop}(j)} \right)$$

### 4.4 Population Stability Index (PSI) Drift Analysis
Detects distribution divergence between baseline training corpus $P$ and online serving distribution $Q$:
$$\text{PSI} = \sum_{k=1}^K (Q_k - P_k) \cdot \ln\left( \frac{Q_k}{P_k} \right)$$
- $\text{PSI} < 0.10$: Distribution stable.
- $0.10 \le \text{PSI} < 0.25$: Moderate drift alert.
- $\text{PSI} \ge 0.25$: Significant drift; raises high-priority drift warning flag (industry design target for triggering automated retraining pipelines in production orchestrators).

### 4.5 Paired Bootstrap Hypothesis Testing
Evaluates if Model B is statistically superior to Model A without parametric assumptions:
1. Resample $N$ impression records with replacement $B = 500$ times.
2. Compute $\Delta_b = \text{Metric}_B^{(b)} - \text{Metric}_A^{(b)}$.
3. Calculate 95% empirical confidence interval: $[\Delta_{2.5\%}, \Delta_{97.5\%}]$.
4. Compute two-sided empirical $p$-value:
   $$p = 2 \cdot \min\left( \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\Delta_b \le 0), \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\Delta_b \ge 0) \right)$$

---

## 5. MLOps & Production Engineering

1. **Deterministic Dataset Fingerprinting**: Every model artifact saves a SHA-256 fingerprint computed across sorted training article IDs, category histograms, and interaction counts. Serving verifies fingerprint compatibility at model load time.
2. **Model Registry & Promotion Gates (Optional MLOps Utility - Option B)**: Standalone registry utility where models progress through `staging` -> `production` -> `archived`. Automated gates require candidate models to exceed production champion on primary validation metrics (AUC/nDCG) before promotion.
3. **Active Learning & Pre-Labeling**: Unlabeled articles are prioritized for human triage using Prediction Entropy $H(p) = -\sum p_i \log_2 p_i$ and Margin Uncertainty $1 - (p_{\text{top1}} - p_{\text{top2}})$. Mock LLM reasoning assists human labelers, with Cohen's $\kappa$ tracking inter-annotator agreement.
