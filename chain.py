import json
import pandas as pd

from preprocessing import *
from variables import *


class Chain:
    def __init__(self, input_cv_path: str):
        columns = ['title', 'city', 'area', 'desired_wage', 'education_level', 'languages', 'skills', 'about']
        df_cv = pd.read_csv(input_cv_path)[columns]

        line = ' '.join([str(item) for item in df_cv.values[0]]).lower()
        cv_doc = tokenize(line, stopwords=sw, need_lemmatize=True)
        self.cv_emb = vectorize(fasttext, [cv_doc], vector_size=300)[0]

    def infer(self):
        with open('./database/cluster_centers.json') as json_file:
            cluster_centers = json.load(json_file)

        cos_dist = 0
        cv_label = 0
        for label, center in cluster_centers.items():
            dist = cosine_similarity(self.cv_emb, center)
            if dist > cos_dist:
                cos_dist = dist
                cv_label = label

        with open('./database/cluster_samples.json', encoding='utf-8') as json_file:
            cluster_samples = json.load(json_file)

        return cluster_samples[str(cv_label)][:10]


if __name__ == '__main__':
    chain = Chain(input_cv_path='./test_vacs_cvs/resumes.csv')
    out = chain.infer()
    print(out)