import re

import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from bs4 import BeautifulSoup

from variables import analyzer, lemmatized_sw


# функция, очистка от html разметки
def clean_html_bs4(text_data):
    soup = BeautifulSoup(text_data, 'lxml')
    return soup.get_text()


# функция, очистка от мусора, нормализация и лемматизация
def tokenize(text, stopwords, need_lemmatize=False):
    result = []
    sentences = [item for item in sent_tokenize(str(text))]

    for sentence in sentences:
        text = sentence.lower()
        text = clean_html_bs4(text)
        text = re.sub(r"\s+", ' ', text)

        tokens = [item for item in word_tokenize(text)]
        tokens = [re.sub("[^а-яА-Яa-zA-Z]", ' ', item) for item in tokens]

        if need_lemmatize:
            tokens = [analyzer.parse(token)[0].normal_form for token in tokens if token not in stopwords  and ' ' not in token and len(token) > 2]
            tokens = [token for token in tokens if token not in lemmatized_sw]
        tokens = [re.sub(r"ё", "е", token) for token in tokens]
        result.extend(tokens)
    return result


# функция, возвращает векторизированное предложение
def get_vector(model, sentence, vector_size=100):
    sentence_vector = []

    if len(sentence) == 0:
        # Пустые предложения заполним их одним  словом
        token_vector = np.zeros(vector_size)
        sentence_vector.append(token_vector)
    else:
        for token in sentence:
            try:
                token_vector = model.wv[token]
            except KeyError as e:
                print('here')
                # Случай неизвестного слова
                token_vector = np.zeros(vector_size)
            finally:
                sentence_vector.append(token_vector)

    return np.mean(sentence_vector, axis=0)


# функция, возвращает векторизированные выборки
def vectorize(model, texts, vector_size=300):
    texts_vectorized = np.zeros((len(texts), vector_size))
    for index, sentence in enumerate(texts):
        texts_vectorized[index] = get_vector(model, sentence, vector_size)

    return texts_vectorized


# функция, косинусная метрика сходства векторов
def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))