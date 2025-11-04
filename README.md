# Policy-Grounded RAG for Explainable Song Lyrics Content Moderation


**Authors:** Arya Doshi  
**Institution:** Vishwakarma Institute of Technology, Pune, India  

---

## 🎯 Overview

This repository contains the code, datasets, and trained models for our research on content moderation using RAG and fine-tuned transformers. We compare two state-of-the-art approaches for song lyrics classification:

- **RAG System:** Policy-grounded retrieval with FLAN-T5 (92% accuracy on quality data)
- **BERT System:** Fine-tuned RoBERTa (89.8% accuracy at scale)

### Key Findings
- ✅ RAG excels with high-quality but limited data (< 1,500 examples)
- ✅ BERT dominates at scale (> 3,000 examples, 13.6× faster)
- ✅ Annotation quality causes 36% accuracy swing
- ✅ Model consensus identifies annotation errors with 100% precision

---

## 📊 Main Results

| System | Training Data | Test Set | Accuracy | F1 Score |
|--------|--------------|----------|----------|----------|
| RAG | 40 policies | 100 (manual) | **92.0%** | 0.92 |
| BERT-Small | 900 songs | 100 (manual) | 83.0% | 0.82 |
| BERT-Large | 4,500 songs | 500 (auto) | **89.8%** | 0.90 |
| RAG | 40 policies | 500 (auto) | 84.8% | 0.85 |

---

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/policy-rag-content-moderation.git
cd policy-rag-content-moderation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running RAG System
```bash
# Classify a single song
python src/rag_classifier.py --lyrics "your song lyrics here"

# Batch evaluation
python src/evaluate_rag.py --test_file data/test_100_manual.csv
```

### Running BERT System
```bash
# Fine-tune RoBERTa
python src/train_bert.py --train_file data/train_900.csv --epochs 4

# Evaluate
python src/evaluate_bert.py --model_path models/bert_small.pt --test_file data/test_100_manual.csv
```

---

## 📁 Repository Structure
```
policy-rag-content-moderation/
│
├── data/                           # Datasets
│   ├── train_900.csv              # Small-scale training set
│   ├── train_4500.csv             # Large-scale training set
│   ├── test_100_manual.csv        # Manually-validated test set
│   ├── test_500_auto.csv          # Auto-annotated test set
│   └── policies/                  # Content moderation policies
│       ├── fcc_policies.json
│       ├── riaa_policies.json
│       ├── youtube_policies.json
│       ├── spotify_policies.json
│       └── common_sense_policies.json
│
├── src/                           # Source code
│   ├── rag_classifier.py         # RAG system implementation
│   ├── bert_classifier.py        # BERT fine-tuning & inference
│   ├── train_bert.py             # BERT training script
│   ├── evaluate_rag.py           # RAG evaluation
│   ├── evaluate_bert.py          # BERT evaluation
│   ├── annotation_consensus.py    # Model consensus validation
│   ├── data_preprocessing.py      # Dataset preprocessing
│   └── utils.py                   # Helper functions
│
├── models/                        # Trained models
│   ├── bert_small.pt             # BERT trained on 900 songs
│   ├── bert_large.pt             # BERT trained on 4,500 songs
│   └── rag_embeddings/           # Policy embeddings for RAG
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_rag_experiments.ipynb
│   ├── 03_bert_experiments.ipynb
│   └── 04_results_analysis.ipynb
│
├── results/                       # Experimental results
│   ├── figures/                   # Paper figures
│   ├── tables/                    # Result tables
│   └── confusion_matrices/
│
├── paper/                         # Research paper
│   ├── IEEE_Policy_Grounded_RAG.pdf
│   └── supplementary_material.pdf
│
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── LICENSE                        # MIT License
└── README.md                      # This file
```

---

## 📦 Requirements
```
Python >= 3.12
PyTorch >= 2.0
transformers >= 4.35
sentence-transformers >= 2.2
chromadb >= 0.4
scikit-learn >= 1.3
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
seaborn >= 0.12
```

See `requirements.txt` for complete list.

---

## 💾 Dataset

### Dataset Statistics

| Split | Total | Appropriate | Inappropriate |
|-------|-------|-------------|---------------|
| **Small Scale** | | | |
| Training | 900 | 454 (50.4%) | 446 (49.6%) |
| Test | 100 | 44 (44.0%) | 56 (56.0%) |
| **Large Scale** | | | |
| Training | 4,500 | 2,741 (60.9%) | 1,759 (39.1%) |
| Test | 500 | 250 (50.0%) | 250 (50.0%) |

### Content Policies

Our RAG system uses **40 content moderation policies** from 5 authoritative sources:
- FCC Broadcasting Standards (8 policies)
- RIAA Parental Advisory Guidelines (8 policies)
- YouTube Community Guidelines (8 policies)
- Spotify Content Policies (8 policies)
- Common Sense Media Ratings (8 policies)

### Data Format
```csv
song_id,title,artist,year,genre,lyrics,label
1,"Song Title","Artist Name",2020,"Pop","[lyrics text]","appropriate"
```

---

## 🔬 Reproducing Results

### Experiment 1: Small-Scale Comparison (High-Quality Data)
```bash
# Train BERT-Small
python src/train_bert.py \
    --train_file data/train_900.csv \
    --epochs 4 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --output_dir models/bert_small

# Evaluate both systems
python src/evaluate_rag.py --test_file data/test_100_manual.csv
python src/evaluate_bert.py --model_path models/bert_small.pt --test_file data/test_100_manual.csv
```

**Expected Results:**
- RAG: 92.0% accuracy, 0.92 F1
- BERT-Small: 83.0% accuracy, 0.82 F1

### Experiment 2: Large-Scale Comparison
```bash
# Train BERT-Large
python src/train_bert.py \
    --train_file data/train_4500.csv \
    --epochs 3 \
    --batch_size 8 \
    --output_dir models/bert_large

# Evaluate
python src/evaluate_bert.py --model_path models/bert_large.pt --test_file data/test_500_auto.csv
```

**Expected Results:**
- BERT-Large: 89.8% accuracy, 0.90 F1
- RAG: 84.8% accuracy, 0.85 F1

### Experiment 3: Annotation Quality Analysis
```bash
# Identify consensus errors
python src/annotation_consensus.py \
    --test_file data/test_100_manual.csv \
    --rag_predictions results/rag_predictions.csv \
    --bert_predictions results/bert_predictions.csv
```

---

## 📈 System Architectures

### RAG System Pipeline
```
Song Lyrics → Sentence Transformer (MPNet) → ChromaDB Retrieval
                                               ↓
                                    Top-3 Relevant Policies
                                               ↓
                                    FLAN-T5-base Generation
                                               ↓
                            Classification + Explanation
```

### BERT System Pipeline
```
Song Lyrics → RoBERTa Tokenizer → 12 Transformer Layers
                                        ↓
                                   [CLS] Token
                                        ↓
                              Classification Head
                                        ↓
                              Binary Prediction
```

---

## 🎯 Practical Usage Examples

### Example 1: Classify New Lyrics
```python
from src.rag_classifier import RAGClassifier

# Initialize classifier
rag = RAGClassifier(
    policy_path="data/policies/",
    model_name="google/flan-t5-base"
)

# Classify lyrics
lyrics = """
I love this beautiful world,
Dancing under the stars tonight
"""

result = rag.classify(lyrics)
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Explanation: {result['explanation']}")
```

### Example 2: Batch Processing
```python
import pandas as pd
from src.bert_classifier import BERTClassifier

# Load model
bert = BERTClassifier.from_pretrained("models/bert_large.pt")

# Load songs
df = pd.read_csv("data/new_songs.csv")

# Classify
predictions = bert.predict_batch(df['lyrics'].tolist())
df['predicted_label'] = predictions

df.to_csv("results/classified_songs.csv", index=False)
```

---

## 📊 Performance Comparison

### Accuracy vs Training Data Size

| Training Size | BERT Accuracy | RAG Accuracy |
|---------------|---------------|--------------|
| 100 | 68.2% | 92.0%* |
| 500 | 82.5% | 92.0%* |
| 900 | 85.4% | 92.0%* |
| 1,500 | 87.9% | ~86%** |
| 3,000 | 90.1% | ~85%** |
| 4,500 | 92.4% | 84.8%** |

*Manual validation, **Auto-annotation

### Inference Speed

| System | Time/Song | 100 Songs | 100K Songs |
|--------|-----------|-----------|------------|
| Keyword Baseline | 0.001s | 0.1s | 1.7 min |
| BERT | 0.067s | 6.7s | 111 min |
| RAG | 0.91s | 91s | 25.3 hrs |

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Citation

If you use this code or dataset in your research, please cite:
```bibtex
@inproceedings{doshi2024policy,
  title={Policy-Grounded Retrieval-Augmented Generation for Explainable Song Lyrics Content Moderation},
  author={Doshi, Arya and Gaikwad, Shravani and Gunje, Suhani},
  booktitle={Conference Name},
  year={2024},
  organization={IEEE}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Hugging Face** for Transformers library
- **Chroma** for vector database
- **Sentence Transformers** for embedding models
- **Genius API** for song lyrics data
- Policy sources: FCC, RIAA, YouTube, Spotify, Common Sense Media

---

## 📧 Contact

**Arya Doshi** - [arya.doshi22@vit.edu](mailto:arya.doshi22@vit.edu)

**Project Link:** [https://github.com/your-username/policy-rag-content-moderation](https://github.com/your-username/policy-rag-content-moderation)

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/policy-rag-content-moderation&type=Date)](https://star-history.com/#your-username/policy-rag-content-moderation&Date)

---

