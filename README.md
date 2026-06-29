# 🎵 Personalized Music Recommender System for Solving User and Item Cold-Start Problems

An end-to-end multimodal music recommender system designed to address **user and item cold-start problems** using **self-supervised contrastive learning**, enabling high-quality recommendations without requiring initial interaction data.

##  Live Demo

Interactive web application deployed using Streamlit for real-time recommendations and multimodal inference.

##  Key Features

* End-to-end **multimodal recommendation pipeline** for addressing both user and item cold-start scenarios.
* Integrates information from **audio**, **lyrics**, and **album artwork** modalities.
* Employs **self-supervised contrastive learning** to learn meaningful song representations without relying on user interactions.
* Models sequential listening behavior using a **Transformer-based SASRec architecture**.
* Supports scalable deployment and efficient inference using precomputed multimodal embeddings.

---

##  System Architecture

### 1. Multimodal Feature Extraction

The system processes three complementary modalities from the **Music4AllOnion dataset**:

* **Audio Features** using a custom **AudioTransformer**
* **Lyrics Features** using a custom **LyricsTransformer**
* **Visual Features** extracted using **ResNet-4096**

These representations are combined adaptively using a **GatedFusion** module to produce a unified multimodal song embedding.

---

### 2. Self-Supervised Representation Learning

The multimodal encoder is trained using **InfoNCE Contrastive Loss**, enabling the model to learn semantic relationships between songs without explicit labels or interaction histories.

This approach significantly improves recommendation quality in cold-start settings where historical user interactions are unavailable.

---

### 3. Sequential Recommendation

User listening preferences are modeled using a **SASRec-style Transformer architecture** optimized with **Bayesian Personalized Ranking (BPR) Loss**.

The sequential model captures temporal listening behavior and personalized preference patterns for next-song recommendation.

---

##  Experimental Results

Evaluation was performed on a **10,000-song subset** of the dataset.

| Metric  | Score     |
| ------- | --------- |
| HR@10   | **0.507** |
| NDCG@10 | **0.465** |

The proposed multimodal approach substantially outperformed single-modality baselines while maintaining an architecture designed to scale seamlessly beyond **100,000 tracks**.

---

##  Deployment

The complete recommendation pipeline has been deployed as an interactive **Streamlit web application**, enabling:

* Real-time recommendation generation
* Interactive song exploration
* Multimodal inference
* User-friendly visualization of recommendations

---

## 🛠️ Technology Stack

* Python
* PyTorch
* Streamlit
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib
* Transformer Architectures
* Contrastive Learning
* SASRec
* Bayesian Personalized Ranking (BPR)

---

## 📚Dataset

The project uses the **Music4AllOnion** dataset containing multimodal music information including:

* Audio features
* Song lyrics
* Album artwork
* Metadata

---

##  Research Contributions

* Developed a unified multimodal representation learning framework for music recommendation.
* Addressed both **user cold-start** and **item cold-start** problems simultaneously.
* Combined self-supervised contrastive learning with sequential recommendation.
* Demonstrated significant improvements over unimodal baselines.
* Designed the system for practical deployment and large-scale scalability.

---

##  Author

**Ranveer Bharti and Ashwini Kumar **

B.Tech Project — Personalized Music Recommendation using Transformer-Based Deep Learning Models.

## References

*Several research papers concepts were used to meet standard architecture followed by several transformers
