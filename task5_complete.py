"""
TASK 5: Consumer Complaint Text Classification
Author: Utpal Raj

This script performs multi-class text classification on consumer complaint dataset
Steps:
1. Exploratory Data Analysis and Feature Engineering
2. Text Pre-Processing
3. Selection of Multi Classification Models
4. Comparison of Model Performance
5. Model Evaluation
6. Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Categories mapping
CATEGORIES = {
    0: 'Credit reporting, repair, or other',
    1: 'Debt collection',
    2: 'Consumer Loan',
    3: 'Mortgage',
}

print("=" * 80)
print("TASK 5: CONSUMER COMPLAINT TEXT CLASSIFICATION")
print("Author: Utpal Raj")
print("=" * 80)
print()

# ============================================================================
# STEP 1: EXPLORATORY DATA ANALYSIS AND FEATURE ENGINEERING
# ============================================================================
print("STEP 1: EXPLORATORY DATA ANALYSIS AND FEATURE ENGINEERING")
print("-" * 80)

# Load data
df = pd.read_csv('data-science/sample_complaints.csv')
print(f"Dataset loaded: {len(df)} samples")
print()

print("Dataset Info:")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print()

print("First 5 samples:")
print(df.head())
print()

print("Class Distribution:")
class_counts = df['label'].value_counts().sort_index()
for label, count in class_counts.items():
    category = CATEGORIES.get(label, 'Unknown')
    print(f"  {label} - {category}: {count} samples ({count/len(df)*100:.1f}%)")
print()

print("Text Statistics:")
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
print(f"  Average text length: {df['text_length'].mean():.1f} characters")
print(f"  Average word count: {df['word_count'].mean():.1f} words")
print(f"  Min length: {df['text_length'].min()} characters")
print(f"  Max length: {df['text_length'].max()} characters")
print()

# ============================================================================
# STEP 2: TEXT PRE-PROCESSING
# ============================================================================
print("STEP 2: TEXT PRE-PROCESSING")
print("-" * 80)
print("Pre-processing steps:")
print("  - Lowercase conversion")
print("  - TF-IDF Vectorization (max_features=20000)")
print("  - N-gram range: (1, 2) - unigrams and bigrams")
print("  - Removing stop words")
print()

# ============================================================================
# STEP 3: SELECTION OF MULTI CLASSIFICATION MODELS
# ============================================================================
print("STEP 3: SELECTION OF MULTI CLASSIFICATION MODELS")
print("-" * 80)
print("Models selected for comparison:")
print("  1. Logistic Regression - Linear model with regularization")
print("  2. Multinomial Naive Bayes - Probabilistic classifier")
print("  3. Linear SVM - Support Vector Machine with linear kernel")
print("  4. Random Forest - Ensemble decision tree method")
print()

# Split data
min_per_class = df.groupby('label').size().min()
stratify = df['label'] if min_per_class >= 2 else None
test_size = 0.2 if len(df) >= 40 else 0.5

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], 
    test_size=test_size, 
    random_state=42, 
    stratify=stratify
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print()

# ============================================================================
# STEP 4: COMPARISON OF MODEL PERFORMANCE
# ============================================================================
print("STEP 4: COMPARISON OF MODEL PERFORMANCE")
print("-" * 80)

# Define models
models = {
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Multinomial NB': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english')),
        ('clf', MultinomialNB())
    ]),
    'Linear SVM': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english')),
        ('clf', LinearSVC(random_state=42, max_iter=1000))
    ]),
    'Random Forest': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
}

results = {}

print("Training and evaluating models...")
print()

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print()

print("Model Performance Comparison:")
print("-" * 50)
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:25s}: {acc:.4f} ({acc*100:.2f}%)")
print()

# Select best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"Best Model: {best_model_name} (Accuracy: {results[best_model_name]:.4f})")
print()

# ============================================================================
# STEP 5: MODEL EVALUATION
# ============================================================================
print("STEP 5: MODEL EVALUATION")
print("-" * 80)
print(f"Detailed evaluation of best model: {best_model_name}")
print()

y_pred = best_model.predict(X_test)

print("Classification Report:")
print("-" * 50)
report = classification_report(
    y_test, 
    y_pred, 
    target_names=[CATEGORIES[i] for i in sorted(CATEGORIES.keys())],
    digits=3
)
print(report)

print("Confusion Matrix:")
print("-" * 50)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print()

# Calculate per-class metrics
print("Per-Class Performance:")
print("-" * 50)
for label in sorted(CATEGORIES.keys()):
    if label in y_test.values:
        mask = y_test == label
        class_acc = accuracy_score(y_test[mask], y_pred[mask]) if mask.sum() > 0 else 0
        print(f"  Class {label} ({CATEGORIES[label]})")
        print(f"    Samples in test: {mask.sum()}")
        print(f"    Accuracy: {class_acc:.3f}")
print()

# ============================================================================
# STEP 6: PREDICTION
# ============================================================================
print("STEP 6: PREDICTION ON NEW DATA")
print("-" * 80)
print("Testing model on example complaints:")
print()

examples = [
    'They reported incorrect credit card payments to the credit bureau',
    'Debt collector keeps calling me about a payday loan',
    'I need help with my personal consumer loan application',
    'My mortgage payment was misapplied and I was charged late fees',
    'Credit report has errors and wrong information about my account',
    'Collection agency is harassing me with constant phone calls'
]

for i, text in enumerate(examples, 1):
    label = int(best_model.predict([text])[0])
    category = CATEGORIES.get(label, 'Unknown')
    print(f"{i}. Text: \"{text}\"")
    print(f"   Predicted: Class {label} - {category}")
    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("TASK 5 COMPLETED SUCCESSFULLY")
print("=" * 80)
print()
print("Summary:")
print(f"  Dataset: {len(df)} samples across {len(CATEGORIES)} categories")
print(f"  Best Model: {best_model_name}")
print(f"  Best Accuracy: {results[best_model_name]:.4f} ({results[best_model_name]*100:.2f}%)")
print(f"  Models Compared: {len(models)}")
print()
print("All steps completed:")
print("  ✓ 1. Exploratory Data Analysis and Feature Engineering")
print("  ✓ 2. Text Pre-Processing")
print("  ✓ 3. Selection of Multi Classification Models")
print("  ✓ 4. Comparison of Model Performance")
print("  ✓ 5. Model Evaluation")
print("  ✓ 6. Prediction on New Data")
print()
print("Author: Utpal Raj")
print("=" * 80)
