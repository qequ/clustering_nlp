#import spacy

# probando nlp con es_core_nwews_sm
#nlp = spacy.load("es_core_news_sm")

import spacy
from spacy.lang.es.examples import sentences

with open("spanish_billion_words/spanish_billion_words_00") as f:
    raw_text = f.read()


nlp = spacy.load("es_core_news_sm")


doc = nlp(raw_text)
print(doc.text)
for token in doc:
    print(token.text, token.pos_, token.dep_)

