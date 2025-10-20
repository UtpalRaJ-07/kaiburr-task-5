import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Categories mapping (subset)
CATEGORIES = {
    0: 'Credit reporting, repair, or other',
    1: 'Debt collection',
    2: 'Consumer Loan',
    3: 'Mortgage',
}


def load_sample():
    # expects columns: text, label (numeric 0..3)
    return pd.read_csv('data-science/sample_complaints.csv')


def build_model():
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2)) ),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    return pipe


def main():
    df = load_sample()
    n = len(df)
    # Ensure enough samples per class for stratify; else fall back to non-stratified and larger test split
    min_per_class = df.groupby('label').size().min()
    stratify = df['label'] if min_per_class >= 2 else None
    test_size = 0.2
    if n < 40:
        # make test set large enough when dataset is tiny
        test_size = max(0.5, 4 / max(n, 1))
        test_size = min(test_size, 0.8)
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=test_size, random_state=42, stratify=stratify
    )

    model = build_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds, digits=3))

    # demo prediction
    examples = [
        'They reported incorrect credit card payments to the credit bureau',
        'Debt collector keeps calling me about a payday loan',
        'My mortgage payment was misapplied and I was charged fees'
    ]
    for text in examples:
        label = int(model.predict([text])[0])
        print(f'"{text}" => {label} ({CATEGORIES.get(label)})')


if __name__ == '__main__':
    main()
