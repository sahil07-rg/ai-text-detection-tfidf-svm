# AI Text Detection using TF-IDF + Linear SVM

A baseline Machine Learning model for detecting AI-generated vs Human-written text using TF-IDF feature extraction and Linear Support Vector Machine (SVM).

This project was built as a learning exercise to understand end-to-end ML pipeline development including preprocessing, feature engineering, training, cross-validation, evaluation, and model persistence.

---

## 📌 Problem Statement

Build a text classification model to classify text into:
- `ai`
- `human`
- `post_edited_ai`

---

## 📊 Dataset

- Source: Kaggle AI Human Detection Dataset
- Total Samples: 686
- Classes: 3
- Highly limited dataset (not production ready)

Due to small dataset size, this model is intended for educational purposes only.

---

## ⚙️ ML Pipeline

### 1. Text Preprocessing
- Lowercasing
- URL removal
- Number removal
- Special character cleaning
- Whitespace normalization

### 2. Feature Engineering
- TF-IDF Vectorizer
- N-gram range: (1,4)
- Max features: 20,000
- English stopwords removal
- Sublinear TF scaling
- min_df = 2

### 3. Model
- LinearSVC
- Custom class weights to handle imbalance:
  - ai: 1
  - human: 3
  - post_edited_ai: 2

### 4. Validation
- 5-fold Cross Validation

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Cross-Validation Accuracy | 0.6641 |
| Test Accuracy | 0.73 |

⚠️ Due to small dataset size (686 samples), results are unstable and not production reliable.

---

## 🧠 Key Learnings

- Text preprocessing using regex
- TF-IDF vectorization
- Class imbalance handling
- Cross-validation for robustness
- Model serialization using pickle
- End-to-end ML pipeline design

---

## 🚀 How to Run

### 1. Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-text-detection-tfidf-svm.git
cd ai-text-detection-tfidf-svm

```
### 2. Install dependencies

```bash
pip install -r requirements.txt

```

### 3. Run training notebook

```bash

notebooks/training.ipynb

```

### 4. Example prediction

```bash

print(predict_text("......"))

```

📦 Model Files

	•	final_ai_text_model.pkl
  
	•	tfidf_vectorizer.pkl
  

⚠️ Disclaimer

This is a small-scale academic project built for learning purposes.
It is NOT suitable for real-world AI content detection.


👨‍💻 Author

Sahil Kumar

Robotics & Artificial Intelligence

Sir M Visvesvaraya Institute of technology


