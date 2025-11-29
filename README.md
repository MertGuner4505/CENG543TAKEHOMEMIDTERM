# Deep Learning Architectures for NLP: From RNNs to RAG

**Course:** CENG 543 - Information Retrieval  
**Instructor:Prof. Aytuğ ONAN
**Project Type:** Take-Home Midterm Exam

##  Project Overview

This repository contains a comprehensive comparative analysis of sequence modeling architectures in Natural Language Processing. The project investigates the transition from recurrence-based methods to self-attention mechanisms and retrieval-augmented systems through five distinct experimental tasks.

The study covers:
1.  **Sequence Classification:** Comparison of Bi-LSTM vs. Bi-GRU with Static (GloVe) and Contextual (BERT) embeddings.
2.  **Attention Mechanisms:** Implementation and analysis of Additive (Bahdanau), Multiplicative (Luong), and Scaled Dot-Product attention in Neural Machine Translation.
3.  **Transformer Architectures:** Building Transformers from scratch, integrating BERT-fused encoders, and conducting ablation studies on depth.
4.  **RAG (Retrieval-Augmented Generation):** Building an Open-Domain QA system using BM25/Dense Retrieval and FLAN-T5.
5.  **Interpretability:** Using Integrated Gradients (Captum) to visualize model decision-making and diagnose failure modes.

##  Repository Structure

* `Q1_Sentiment_Analysis.ipynb`: Training RNNs and analyzing latent spaces with t-SNE.
* `Q2_Machine_Translation_Attention.ipynb`: Seq2Seq NMT (English-German) with various attention mechanisms.
* `Q3_Transformers_Translation.ipynb`: Transformer implementation and comparison against RNN baselines.
* `Q4_RAG_System.ipynb`: Retrieval-Augmented Generation pipeline on HotpotQA.
* `Q5_Interpretability.ipynb`: Diagnostic evaluation and feature attribution analysis.

##  Key Findings

* **Efficiency Wins:** In low-resource settings (like the Multi30k dataset), simpler architectures (1-Layer Transformer) often outperformed deeper, pre-trained models due to data scarcity constraints.
* **Attention Quality:** Scaled Dot-Product attention provided the sharpest alignment and highest translation quality, outperforming multiplicative variants.
* **Retrieval Matters:** In the RAG system, switching from sparse (BM25) to dense (Sentence-BERT) retrieval significantly improved answer generation fidelity.
## IMPORTANT NOTE
* **README file requirements.txt file and source code was made by generative artificial intelligence
##  Installation & Setup

To reproduce the results, use the following "Golden Setup" to avoid dependency conflicts between PyTorch, TorchText, and NumPy.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
