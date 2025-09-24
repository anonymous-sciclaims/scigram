# SciGram: A Large-Scale Dataset for Scientific Diagram Understanding

This repository contains the official code accompanying the paper:
"SciGram: A Large-Scale Dataset for Scientific Diagram Understanding"

Our goal is to facilitate research on the automatic understanding of scientific diagrams by providing the tools necessary to reconstruct the SciGram dataset from publicly available sources.


## 🔍 Overview

Dataset Purpose: Enable research on diagram understanding, and scientific knowledge extraction.

Repository Goal: Provide scripts and instructions to recreate SciGram using linked resources.

**Disclaimer: We do not redistribute copyrighted images. Instead, we provide links and metadata.**


## 🧩 Approach

The SciGram dataset creation follows these steps:

1. TERMINOLOGY EXTRACTION

2. SCIENTIFIC CLAIM GENERATION

3. DIAGRAM RETRIEVAL

4. CAPTION SYNTHESIS

5. MULTIPLE-CHOICE QUESTION SYNTHESIS

6. CURATED DATASETS COLLECTION


## ⚙️ Installation
```
git clone https://github.com/anonymous-sciclaims/scigram.git
cd scigram
pip install -r requirements.txt
```

## 🚀 Usage

To re-create the dataset:

1. TERMINOLOGY EXTRACTION
```
python terminology_extraction/generate_tqa_metadata.py
python terminology_extraction/extract_tqa_vocab.py
```
2. SCIENTIFIC CLAIM GENERATION
```
python scientific_claim_gen/generate_prompts.py
python scientific_claim_gen/get_claims.py
python scientific_claim_gen/clean_claims.py
```
3. DIAGRAM RETRIEVAL
```
python diagram_retrieval/add_urls.py
python diagram_retrieval/clean_urls.py
```
4. CAPTION SYNTHESIS
```
python scigram_construction/generate_captions.py
```
5. MULTIPLE-CHOICE QUESTION SYNTHESIS
```
python scigram_construction/generate_mcqa.py
```
7. CURATED DATASETS COLLECTION
```
python scigram_construction/process_ai2d.py
python scigram_construction/process_science_qa_full.py
python scigram_construction/generate_m3_balanced.py
```


## 📄 Disclaimer

The SciGram dataset does not contain any images.

We provide only URLs/links pointing to the original figures.

All images are copyrighted by their respective authors and publishers.

Usage of the dataset must comply with the terms of the source repositories.


## 📚 Citation

To be completed...
