import spacy
from gensim.models import Word2Vec
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter


NUM_CLUSTERS = 25

# funcion para normalizar el oraciones


def normalize_sentence(span):
    # sacamos stopwords y signos de puntuación
    words = [t for t in span if not t.is_punct | t.is_stop]
    # nos quedamos con tokens alfabéticos y con largo considerable
    lexical_tokens = [t for t in words if len(t.orth_) > 3 and
                      t.orth_.isalpha()]

    # removemos pronombres
    cleaned_lemmas = [tok.lemma_.lower()
                      for tok in lexical_tokens if tok.pos_ != 'PRON']

    return cleaned_lemmas


def show_results(vocabulary, model):
    # Show results
    c = Counter(sorted(model.labels_))
    print("\nTotal clusters:", len(c))
    for cluster in c:
        print("Cluster#", cluster, " - Total words:", c[cluster])

    # Show top terms and words per cluster
    print("Top words per cluster:")
    print()

    keysVocab = list(vocabulary.keys())
    for n in range(len(c)):
        print("Cluster %d" % n)
        print("Words:", end='')
        word_indexs = [i for i, x in enumerate(list(model.labels_)) if x == n]
        for i in word_indexs:
            print(' %s' % keysVocab[i], end=',')
        print()
        print()

    print()


# corpus de texto
with open("spanish_billion_words_00") as f:
    raw_text = f.read()
raw_text = raw_text[:len(raw_text) // 75]


nlp = spacy.load("es_core_news_sm")


doc = nlp(raw_text)
lemmatized_sentences = []

for span in doc.sents:
    lemmatized_sentences.append(normalize_sentence(span))


# entrenamos un modelo de word embeddings neuronales
model = Word2Vec(lemmatized_sentences, min_count=1)
vocabulary = model.wv.key_to_index

# vectores que conseguimos del modelo
vectors = []
for word in vocabulary:
    vectors.append(model.wv[word])

matrix = np.array(vectors)
print("Matrix shape:", matrix.shape)

# normalizamos matriz y reducimos dimensionalidad quitando columnas con poca varianza

matrix_normed = matrix / matrix.max(axis=0)

variances = np.square(matrix_normed).mean(axis=0) - \
    np.square(matrix_normed.mean(axis=0))
VarianzaMin = 0.001
red_matrix = np.delete(matrix_normed, np.where(
    variances < VarianzaMin), axis=1)


# Utilizamos el algoritmo de K-means de scikit-learn
k_means_model = KMeans(n_clusters=NUM_CLUSTERS)
k_means_model.fit(red_matrix)


show_results(vocabulary, k_means_model)
