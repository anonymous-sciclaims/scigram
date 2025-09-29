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
conda create -n scigram python==3.11.0
conda activate scigram
pip install -r requirements.txt
```

## 🚀 Usage

To re-create the dataset:

1. TERMINOLOGY EXTRACTION
```
sh data/download_tqa.sh
sh data/download_bnc.sh
python terminology_extraction/tqa_frequencies.py --tqa_path data/tqa_train_val_test --output_path data/term_extraction/tqa_frequencies.json
python terminology_extraction/bnc_frequencies.py --bnc_path data/term_extraction/BNC/Texts/ --output_path data/term_extraction/bnc_frequencies.json
python terminology_extraction/generate_tqa_metadata.py --tqa_path data/tqa_train_val_test --output_path data/term_extraction/tqa_metadata.json --bnc_freqs_path data/term_extraction/bnc_frequencies.json --tqa_freqs_path data/term_extraction/tqa_frequencies.json --stopwords_path data/term_extraction/more_stopwords.json
python terminology_extraction/extract_tqa_vocab.py --metadata_path data/term_extraction/tqa_metadata.json --output_path data/term_extraction/tqa_vocab.json
python terminology_extraction/get_tqa_sentences.py --metadata_path data/term_extraction/tqa_metadata.json --output_path data/term_extraction/tqa_sentences.jsonl
python terminology_extraction/terminology_wise_sentences.py --vocab_path data/term_extraction/tqa_vocab.json --sentence_path data/term_extraction/tqa_sentences.jsonl --output_path data/term_extraction/tqa_sentences_with_terms.jsonl

```
2. SCIENTIFIC CLAIM GENERATION
```
python scientific_claim_gen/generate_prompts.py --input_path data/term_extraction/tqa_sentences.jsonl --output_path data/claim_gen/claim_prompts.jsonl --hf_token <your huggingface token>
python scientific_claim_gen/get_claims.py --input_path data/claim_gen/claim_prompts.jsonl --output_dir data/claim_gen/ --hf_token <your huggingface token>
python scientific_claim_gen/clean_claims.py --input_path data/claim_gen/claims/claims_part<n>.jsonl --output_path data/claim_gen/claims/claims_clean_part<n>.jsonl
```
3. DIAGRAM RETRIEVAL
```
python diagram_retrieval/add_urls.py --input_path data/claim_gen/claims/claims_part<n>.jsonl --output_path data/urls/urls_part<n>.jsonl
python diagram_retrieval/clean_urls.py --input_path data/claim_gen/urls/urls_part<n>.jsonl --output_path data/urls/urls_clean_part<n>.jsonl
```
4. CAPTION SYNTHESIS
```
python scigram_construction/generate_captions.py --image_dir_path data/ --input_path scigram_base.jsonl --output_path data/scigram_alignment/scigram_alignment_part1.json
python scigram_construction/generate_captions.py --image_dir_path data/ --input_path scigram_base.jsonl --output_path data/scigram_alignment/scigram_alignment_part2.json
python scigram_construction/generate_captions.py --image_dir_path data/ --input_path scigram_base.jsonl --output_path data/scigram_alignment/scigram_alignment_part3.json
python scigram_construction/merge_alignments.py
```
5. MULTIPLE-CHOICE QUESTION SYNTHESIS
```
python scigram_construction/generate_mcqa.py --image_dir_path data/ --input_path scigram_base.jsonl --output_path data/scigram_vit/scigram_vit.json
```
7. CURATED DATASETS COLLECTION
```
sh data/download_ai2d.sh
python scigram_construction/process_ai2d.py
python scigram_construction/process_science_qa_full.py
python scigram_construction/generate_m3.py --ai2d_path data/ai2d/train_fix.jsonl --ai2d_path data/ai2d/train_full.jsonl --tqa_path tqa_train_test_val/tqa_v1_train.json --output_path data/scigram_m3/scigram_m3.json
```


## 📄 Disclaimer

The SciGram dataset does not contain any images.

We provide only URLs/links pointing to the original figures.

All images are copyrighted by their respective authors and publishers.

Usage of the dataset must comply with the terms of the source repositories.


## 📚 Citation

To be completed...
