# Benchmarking Named Entity Recognition in the Portuguese Language

## About

This project presents a comprehensive comparative study of **Named Entity Recognition (NER)** in Portuguese, evaluating state-of-the-art approaches for information extraction. We investigate the effectiveness of fine-tuning technique across the BERT transformer-based model and large language models (LLMs) such as QWEN on diverse Portuguese datasets. The study employs iterative stratification for dataset partitioning, ensuring balanced distribution of entity types across folds to provide robust cross-validation and reliable performance estimation.

## Overview

The project encompasses:

- **Multiple Models**: BERT and QWEN with different scales
- **Multiple Approaches**: Fine-tuning learning strategies
- **Multiple Datasets**: CachacaNER, HAREM (First, Second, Mini), leNER, Paramopama, and UlyssesNER-BR
- **Comprehensive Evaluation**: Cross-validation 

### Key Features

- Evaluation of domain-specific and general-domain datasets
- Cross-validation partitions for robust performance estimation
- Pre-processing pipelines for each dataset
- Detailed error analysis and performance metrics
- Support for distributed training and optimization (Optuna)

## Project Structure

```
.
├── BERT/
│   ├── BERT_errors/          # Error analysis for BERT models
│   └── FINE_TUNING/          # Fine-tuning notebooks and experiments
├── LLMs/
│   ├── qwen3.5/    #  Fine-tuning implementation and error analysis
│   
├── Partitions/
│   ├── Database/             # Dataset files and partitions
│   │   ├── cachacaNER/
│   │   ├── HAREM/
│   │   ├── leNER/
│   │   ├── Paramopama/
│   │   └── UlyssesNER-BR/
│   ├── Graphics_partitions/  # Visualizations of dataset partitions
│   └── PRE_PROCESSING/       # Data preprocessing notebooks
└── requirements.txt          # Project dependencies
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `transformers` - Hugging Face transformers library
- `datasets` - Hugging Face datasets library
- `torch` - PyTorch deep learning framework
- `conllu` - CoNLL-U format support
- Jupyter and Optuna for experiments


## Citation

If you use this project in your research, please cite our work:

```bibtex
@article{[YourProjectYear],
    author={[Julia Ribeiro and Denilson Alves Pereira]},
    title={{Benchmarking Named Entity Recognition in the Portuguese Language}},
    journal={[Journal of the Brazilian Computer Society]},
    year={[2026]},
    doi={[DOI]},
    url={[URL]},
    note={[submitted]}
}
```

---
