import spacy


# corpus de texto
with open("lavoz.txt") as f:
    raw_text = f.read()
raw_text = raw_text[:len(raw_text) // 65]

# spacy nlp
nlp = spacy.load("es_core_news_sm")

# funcion para normalizar el texto


def normalize(text):
    doc = nlp(text)
    # sacamos stopwords y signos de puntuación
    words = [t for t in doc if not t.is_punct | t.is_stop]
    # nos quedamos con tokens alfabéticos y con largo considerable
    lexical_tokens = [t for t in words if len(t.orth_) > 3 and
                      t.orth_.isalpha()]

    # removemos lemmas repetidos y pronombres
    cleaned_lemmas = list(set([tok.lemma_.lower()
                          for tok in lexical_tokens if tok.pos_ != 'PRON']))
    return cleaned_lemmas


tokens = normalize(raw_text)

print(tokens)
