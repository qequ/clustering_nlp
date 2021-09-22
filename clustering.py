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
    words = [t.orth_ for t in doc if not t.is_punct | t.is_stop]
    lexical_tokens = [t.lower() for t in words if len(t) > 3 and
                      t.isalpha()]
    return lexical_tokens


tokens = normalize(raw_text)

print(tokens)
