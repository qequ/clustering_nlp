# Text Mining - Clustering
Alumno: Alvaro Frias Garay

# Objetivo 
Encontrar grupos de palabras similares en un corpus de texto

# Detalles técnicos
se utilizó [el corpus SBWCE de Cristian Cardellino](https://crscardellino.ar/resources/nlp/2016/02/06/spanish-billion-words-corpus-and-embeddings.html) y las siguientes tecnologías:

* Spacy
* Gensim
* scikit-learn

# Procedimiento

## Preprocesamiento del corpus
Se procesaron las oraciones agrupadas como tokens y se les dio el siguiente tratamiento;

* Se removieron stopwords y signos de puntuación.
* Se quitaron tokens no alfabéticos y de un largo de palabra menor a 3.
* Se removieron pronombres.

## Vectorización

Se utilizaron _Word Embeddings Neuronales_, Word2Vec, para crear vectores de palabras a partir del corpus dado.

## Tratamiento de la matriz de Word2Vec
Se la normalizó y se quitaron dimensiones con poca varianza

## Clustering
Se utilizó el algoritmo de K-means tomando una _ventana_ de 5, una frecuencia mínima de 5 y un número de clusters de 25.

## Resultados
 A continuación una muestra de palabras agrupadas en clusters

```
Cluster 0
Words: mediodía, perdóname, bárbara, apresuré, visitarlas, dimir, créanmir, bendiga, vedado, mencioné, retornar, jurar, vieja, opción

```

```
Cluster 1
Words: seguro, ocurrir, vivir, habitación, mujer, bessie, noche, deseo, amigo, recibir, mano, puerta, opinión, hija, asunto, quedar, hijo, dar, pequeño, cuarto, forma, ser, atención, mirada, aspecto,
```
```
Cluster 2
Words: acaso, contestar, rato, noticia, lamentar, alegrar, amable, faltar, vuelta, comprender, criado, satisfacción, separar, sonrisa
```

```
Cluster 3
Words: elinor, señora, marianne, haber, hermana, casa, sentir, madre, deber, hacer,
```