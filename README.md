

# Trump Tweet Device Classification

This repository contains analysis and code for classifying Donald Trump’s tweets by the device used (Android vs. iPhone), based on the hypothesis that different devices may reflect distinct writing patterns, temporal behaviors, or contextual usage.

---

## Problem Definition and Dataset

* **Objective**: Predict whether a tweet was posted from an Android device or an iPhone.
* **Dataset**: Collection of Donald Trump’s tweets, each labeled with the posting device and timestamp.

---

## Data Preprocessing Pipeline

1. **Text Normalization**: Convert all text to lowercase and handle HTML entities (e.g., `&amp;`, `&lt;`).
2. **URL & Mention Removal**: Strip out Twitter URLs (`t.co` links) and user mentions (`@username`).
3. **Hashtag Processing**: Remove the `#` symbol but retain the hashtag text.
4. **Contraction Expansion**: Standardize contractions (e.g., `won’t` → `will not`, `can’t` → `cannot`).
5. **Character Normalization**: Collapse repeated characters (e.g., `sooooo` → `so`).
6. **Tokenization & Lemmatization**: Use NLTK’s WordNetLemmatizer.
7. **Stopword Removal**: Eliminate common English stopwords.
8. **Feature Filtering**: Keep only alphabetic tokens with more than two characters.

---

## Feature Engineering

### Temporal Features (9 dimensions)

* `hour_of_day`
* `day_of_week`
* `day_of_month`
* `month`
* `year`
* `is_weekend` (boolean)
* `is_business_hours` (boolean)
* `is_late_night` (boolean)
* `season` & `time_period` (morning/afternoon/evening/night)

### Text Features

* **TF-IDF Vectorization**: 1‑ and 2‑gram models for traditional ML algorithms.
* **Stylometric Features**: 25+ handcrafted features (e.g., punctuation counts, capitalization patterns, Trump‑specific phrases).
* **Raw Text**: Minimal preprocessing when fine‑tuning transformer models like DistilBERT.

---

## Model Evaluation

| Model                   | F1 Score | Key Driver(s)                                        |
| ----------------------- | -------- | ---------------------------------------------------- |
| **DistilBERT**          | 0.862    | Raw text + contextual embeddings                     |
| **Feed‑Forward NN**     | 0.816    | Dense TF‑IDF + temporal features (non‑linear combos) |
| **Random Forest**       | 0.806    | Stylometric + temporal features (ensemble learning)  |
| **Logistic Regression** | 0.693    | Clean TF‑IDF (linear relationships)                  |
| **SVM**                 | 0.688    | Linear kernel; complex kernels added little benefit  |

**Preprocessing Choices**

* Classical ML: Aggressive cleaning to reduce TF‑IDF sparsity.
* Random Forest: Moderate cleaning + stylometric features.
* DistilBERT: Light cleanup, preserving original punctuation & casing.

**Feature Insights**

* Temporal signals (e.g., `hour_of_day`) rank among top RF features.
* Stylometric features account for \~60% of RF importance.
* Contextual embeddings implicitly capture style and content.

**Model‑Specific Learnings**

* Linear vs. non‑linear learners: trees and NNs benefit from richer feature interactions.
* Early stopping prevents overfitting in FFNN and transformers.
* DistilBERT outperforms classical methods without extensive feature engineering.

---

## Cross‑Validation & Hyperparameter Tuning

* **Strategy**: 5‑fold stratified cross‑validation to maintain balanced class distribution.
* **Grid Search** on key hyperparameters for each algorithm:

### Logistic Regression

* `C`: 1.0, 2.0
* `solver`: liblinear, lbfgs
* `penalty`: l2

### Support Vector Machine

* `kernel`: linear, rbf, poly
* `C`: 0.1, 1.0, 10.0
* `gamma`: scale, auto, 0.001, 0.01
* `degree`: 2, 3

### Feed‑Forward Neural Network

* `hidden_sizes`: (512, 256, 128), (256, 128)
* `dropout_rate`: 0.3, 0.5
* `learning_rate`: 0.001, 0.0005
* `weight_decay`: 0.0, 0.01
* `batch_size`: 32, 64

### Random Forest

* `n_estimators`: 100, 200
* `max_depth`: 10, 20
* `min_samples_split`: 2
* `min_samples_leaf`: 1, 2
* `max_features`: sqrt

### DistilBERT

* `num_train_epochs`: 2, 3
* `per_device_train_batch_size`: 8, 16
* `learning_rate`: 2e-5, 5e-5
* `weight_decay`: 0.0, 0.01
* `warmup_steps`: 0, 500

---

## Optimization Approaches

* **Grid Search**: Systematic hyperparameter exploration for SVM and RF.
* **Early Stopping**: FFNN and DistilBERT to avoid overfitting.
* **Fixed Parameters**: Default settings for LR to maintain interpretability and efficiency.

---

## Practical Takeaways

1. Use raw text when fine‑tuning transformers; apply aggressive cleaning for classical models.
2. Incorporate simple temporal features—they consistently boost performance.
3. Stylometric feature engineering is valuable when deep models are not available.
4. Fine‑tuned DistilBERT yields the best results; for cost‑effectiveness, Random Forest with stylometry + temporal features achieves >0.80 F1.
