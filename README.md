# LLM Comparison Project

This repository contains a Jupyter notebook that compares the performance of different language models for sentiment analysis on the IMDb movie reviews dataset.

## Project Overview

The project evaluates and compares the performance of the following models:
- DistilBERT (base model)
- Fine-tuned DistilBERT
- GPT-2 (base model)
- Logistic Regression with TF-IDF features (traditional ML approach)

The models are compared based on their accuracy, precision, recall, F1-score, and confusion matrices.

## Dataset

The project uses the IMDb movie reviews dataset, which consists of 50,000 movie reviews labeled as either positive or negative sentiment.

## Implementation Details

### Data Preprocessing
- Text cleaning: Removing HTML tags and special characters
- Lowercasing text
- Splitting into train/test sets (80/20 split)
- Tokenization for transformer models

### Models Implementation
1. **DistilBERT Base Model**: Using the pre-trained model without fine-tuning
2. **Fine-tuned DistilBERT**: Fine-tuning the pre-trained model on the IMDb dataset
3. **GPT-2 Base Model**: Using the pre-trained model for sentiment classification
4. **Logistic Regression**: Traditional ML approach using TF-IDF vectorization

### Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- Confusion matrices
- Training time comparison

## Key Findings

The analysis shows that:
- Fine-tuned DistilBERT performs best with approximately 90% accuracy
- Logistic Regression with TF-IDF features performs surprisingly well (about 89% accuracy)
- Base models (DistilBERT and GPT-2) without fine-tuning perform poorly (around 50% accuracy)
- Traditional machine learning approaches can still be competitive for sentiment analysis tasks

## Requirements

The notebook requires the following libraries:
- PyTorch
- Transformers (Hugging Face)
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Usage

1. Open the `LLM_Comparison_Project.ipynb` in Google Colab or Jupyter Notebook
2. Execute the cells sequentially to reproduce the analysis

## Visualizations

The notebook includes several visualizations:
- Confusion matrices for all models
- Performance comparison charts
- Accuracy and F1-score comparisons

## Conclusion

The project demonstrates that while transformer-based models can achieve superior performance when fine-tuned, traditional machine learning approaches can still be competitive for certain NLP tasks. The choice of model should consider the trade-off between performance, computational resources, and time constraints.