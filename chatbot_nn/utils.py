from sklearn.feature_extraction.text import CountVectorizer

# bag-of-words model to encode phrases
def encode_phrases(phrases):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(phrases).toarray()
    return X, vectorizer
