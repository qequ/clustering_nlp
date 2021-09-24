import spacy
from gensim.models import Word2Vec
import numpy as np

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
