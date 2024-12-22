import nltk
import numpy as np
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    preprocessed_sentences = []
    for sentence in sentences:
        words = [word.lower() for word in nltk.word_tokenize(sentence) if word.isalnum() and word.lower() not in stop_words]
        preprocessed_sentences.append(' '.join(words))
    return preprocessed_sentences

def build_similarity_matrix(sentences):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix


def generate_summary(text, num_sentences):
  
    sentences = sent_tokenize(text)
    preprocessed_sentences = preprocess_text(text)

  
    similarity_matrix = build_similarity_matrix(preprocessed_sentences)

    sentence_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(sentence_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    summary = " ".join([ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))])
    return summary

if __name__ == "__main__":
    text = input("Enter the text to summarize: ")
    num_sentences = int(input("Enter the number of sentences for the summary: "))
    summary = generate_summary(text, num_sentences)
    print("\nSummary:")
    print(summary)





