from fuzzywuzzy import fuzz
import distance
from collections import defaultdict
import networkx as nx
from keras.models import model_from_json
import re
import pandas as pd
import numpy as np
import time
from keras.preprocessing.sequence import pad_sequences
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import json
import pickle
from multiprocessing.pool import ThreadPool


class SimilarityPredictor(object):
    def __init__(self):
        self.WNL = WordNetLemmatizer()
        self.SAFE_DIV = 0.0001
        self.STOP_WORDS = stopwords.words("english")
        self.NB_CORES = 10
        self.FREQ_UPPER_BOUND = 100
        self.NEIGHBOR_UPPER_BOUND = 5
        self.MODEL_JSON = "model.json"
        self.NUM_MODELS = 10
        self.TRAIN_TARGET_MEAN = 0.37
        self.TEST_TARGET_MEAN = 0.16
        self.MAX_SEQUENCE_LENGTH = 30
        self.MIN_WORD_OCCURRENCE = 100
        self.REPLACE_WORD = "memento"
        self.EMBEDDING_DIM = 300
        self.NUM_FOLDS = 10
        self.BATCH_SIZE = 512
        self.EMBEDDING_FILE = "F:\\Thesis_Code\\data\\glove.840B.300d.txt"

        print("Loading top words")
        with open("top_words.pkl", "rb") as f:
            self.top_words = pickle.load(f)

        print("Loading tokenizer")
        with open("tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)

        print("Loading word embeddings")
        self.word_embedding_matrix = np.load(open("word_embedding_matrix.npy", 'rb'))

        print("Loading models")
        self.models = []
        for i in range(0, self.NUM_MODELS):
            with open(self.MODEL_JSON, "r") as f:
                model = model_from_json(json.load(f))
                model.load_weights("best_model" + str(i) + ".h5")
                model._make_predict_function()
                self.models.append(model)

        self.pool = ThreadPool(processes=self.NUM_MODELS)

    def _cutter(self, word):
        if len(word) < 4:
            return word
        return self.WNL.lemmatize(self.WNL.lemmatize(word, "n"), "v")

    def _preprocess(self, string):
        string = string.lower().replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'") \
            .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
            .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
            .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
            .replace("he's", "he is").replace("she's", "she is").replace("'s", " own") \
            .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ") \
            .replace("€", " euro ").replace("'ll", " will").replace("=", " equal ").replace("+", " plus ")
        string = re.sub('[“”\(\'…\)\!\^\"\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', string)
        string = re.sub(r"([0-9]+)000000", r"\1m", string)
        string = re.sub(r"([0-9]+)000", r"\1k", string)
        string = ' '.join([self._cutter(w) for w in string.split()])
        return string

    def _get_embedding(self):
        embeddings_index = {}
        f = open(self.EMBEDDING_FILE, encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            if len(values) == self.EMBEDDING_DIM + 1 and word in self.top_words:
                coefs = np.asarray(values[1:], dtype="float32")
                embeddings_index[word] = coefs
        f.close()
        return embeddings_index

    def _is_numeric(self, s):
        return any(i.isdigit() for i in s)

    def _prepare(self, q):
        new_q = []
        surplus_q = []
        numbers_q = []
        new_memento = True
        for w in q.split()[::-1]:
            if w in self.top_words:
                new_q = [w] + new_q
                new_memento = True
            elif w not in self.STOP_WORDS:
                if new_memento:
                    new_q = ["memento"] + new_q
                    new_memento = False
                if self._is_numeric(w):
                    numbers_q = [w] + numbers_q
                else:
                    surplus_q = [w] + surplus_q
            else:
                new_memento = True
            if len(new_q) == self.MAX_SEQUENCE_LENGTH:
                break
        new_q = " ".join(new_q)
        return new_q, set(surplus_q), set(numbers_q)

    def _extract_features(self, df):
        q1s = np.array([""] * len(df), dtype=object)
        q2s = np.array([""] * len(df), dtype=object)
        features = np.zeros((len(df), 4))

        for i, (q1, q2) in enumerate(list(zip(df["question1"], df["question2"]))):
            q1s[i], surplus1, numbers1 = self._prepare(q1)
            q2s[i], surplus2, numbers2 = self._prepare(q2)
            features[i, 0] = len(surplus1.intersection(surplus2))
            features[i, 1] = len(surplus1.union(surplus2))
            features[i, 2] = len(numbers1.intersection(numbers2))
            features[i, 3] = len(numbers1.union(numbers2))

        return q1s, q2s, features

    def _get_token_features(self, q1, q2):
        token_features = [0.0] * 10

        q1_tokens = q1.split()
        q2_tokens = q2.split()

        if len(q1_tokens) == 0 or len(q2_tokens) == 0:
            return token_features

        q1_words = set([word for word in q1_tokens if word not in self.STOP_WORDS])
        q2_words = set([word for word in q2_tokens if word not in self.STOP_WORDS])

        q1_stops = set([word for word in q1_tokens if word in self.STOP_WORDS])
        q2_stops = set([word for word in q2_tokens if word in self.STOP_WORDS])

        common_word_count = len(q1_words.intersection(q2_words))
        common_stop_count = len(q1_stops.intersection(q2_stops))
        common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

        token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + self.SAFE_DIV)
        token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + self.SAFE_DIV)
        token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + self.SAFE_DIV)
        token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + self.SAFE_DIV)
        token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + self.SAFE_DIV)
        token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + self.SAFE_DIV)
        token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
        token_features[7] = int(q1_tokens[0] == q2_tokens[0])
        token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
        token_features[9] = (len(q1_tokens) + len(q2_tokens)) / 2
        return token_features

    def _get_longest_substr_ratio(self, a, b):
        strs = list(distance.lcsubstrings(a, b))
        if len(strs) == 0:
            return 0
        else:
            return len(strs[0]) / (min(len(a), len(b)) + 1)

    def _extract_features1(self, df):
        df["question1"] = df["question1"].fillna("").apply(self._preprocess)
        df["question2"] = df["question2"].fillna("").apply(self._preprocess)

        token_features = df.apply(lambda x: self._get_token_features(x["question1"], x["question2"]), axis=1)
        df["cwc_min"] = list(map(lambda x: x[0], token_features))
        df["cwc_max"] = list(map(lambda x: x[1], token_features))
        df["csc_min"] = list(map(lambda x: x[2], token_features))
        df["csc_max"] = list(map(lambda x: x[3], token_features))
        df["ctc_min"] = list(map(lambda x: x[4], token_features))
        df["ctc_max"] = list(map(lambda x: x[5], token_features))
        df["last_word_eq"] = list(map(lambda x: x[6], token_features))
        df["first_word_eq"] = list(map(lambda x: x[7], token_features))
        df["abs_len_diff"] = list(map(lambda x: x[8], token_features))
        df["mean_len"] = list(map(lambda x: x[9], token_features))

        df["token_set_ratio"] = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
        df["token_sort_ratio"] = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
        df["fuzz_ratio"] = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
        df["fuzz_partial_ratio"] = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
        df["longest_substr_ratio"] = df.apply(lambda x: self._get_longest_substr_ratio(x["question1"], x["question2"]),
                                              axis=1)
        return df

    def _create_question_hash(self, train_df):
        train_qs = np.dstack([train_df["question1"], train_df["question2"]]).flatten()
        all_qs = train_qs
        all_qs = pd.DataFrame(all_qs)[0].drop_duplicates()
        all_qs.reset_index(inplace=True, drop=True)
        question_dict = pd.Series(all_qs.index.values, index=all_qs.values).to_dict()
        return question_dict

    def _get_hash(self, df, hash_dict):
        df["qid1"] = df["question1"].map(hash_dict)
        df["qid2"] = df["question2"].map(hash_dict)
        return df.drop(["question1", "question2"], axis=1)

    def _get_kcore_dict(self, df):
        g = nx.Graph()
        g.add_nodes_from(df.qid1)
        edges = list(df[["qid1", "qid2"]].to_records(index=False))
        g.add_edges_from(edges)
        g.remove_edges_from(g.selfloop_edges())

        df_output = pd.DataFrame(data=g.nodes(), columns=["qid"])
        df_output["kcore"] = 0
        for k in range(2, self.NB_CORES + 1):
            ck = nx.k_core(g, k=k).nodes()
            df_output.ix[df_output.qid.isin(ck), "kcore"] = k

        return df_output.to_dict()["kcore"]

    def _get_kcore_features(self, df, kcore_dict):
        df["kcore1"] = df["qid1"].apply(lambda x: kcore_dict[x])
        df["kcore2"] = df["qid2"].apply(lambda x: kcore_dict[x])
        return df

    def _convert_to_minmax(self, df, col):
        sorted_features = np.sort(np.vstack([df[col + "1"], df[col + "2"]]).T)
        df["min_" + col] = sorted_features[:, 0]
        df["max_" + col] = sorted_features[:, 1]
        return df.drop([col + "1", col + "2"], axis=1)

    def _get_neighbors(self, train_df):
        neighbors = defaultdict(set)
        for df in [train_df]:
            for q1, q2 in zip(df["qid1"], df["qid2"]):
                neighbors[q1].add(q2)
                neighbors[q2].add(q1)
        return neighbors

    def _get_neighbor_features(self, df, neighbors):
        common_nc = df.apply(lambda x: len(neighbors[x.qid1].intersection(neighbors[x.qid2])), axis=1)
        min_nc = df.apply(lambda x: min(len(neighbors[x.qid1]), len(neighbors[x.qid2])), axis=1)
        df["common_neighbor_ratio"] = common_nc / min_nc
        df["common_neighbor_count"] = common_nc.apply(lambda x: min(x, self.NEIGHBOR_UPPER_BOUND))
        return df

    def _get_freq_features(self, df, frequency_map):
        df["freq1"] = df["qid1"].map(lambda x: min(frequency_map[x], self.FREQ_UPPER_BOUND))
        df["freq2"] = df["qid2"].map(lambda x: min(frequency_map[x], self.FREQ_UPPER_BOUND))
        return df

    def similarity(self, sequence1, sequence2):
        df = pd.DataFrame({"question1":sequence1, "question2":sequence2})
        nlp_features = self._extract_features1(df.copy())
        nlp_features.drop(["question1", "question2"], axis=1, inplace=True)

        question_dict = self._create_question_hash(df.copy())
        train_df = self._get_hash(df, question_dict)
        kcore_dict = self._get_kcore_dict(train_df)
        train_df = self._get_kcore_features(train_df, kcore_dict)
        train_df = self._convert_to_minmax(train_df, "kcore")
        neighbors = self._get_neighbors(train_df)
        train_df = self._get_neighbor_features(train_df, neighbors)
        frequency_map = dict(zip(*np.unique(np.vstack((train_df["qid1"], train_df["qid2"])), return_counts=True)))
        train_df = self._get_freq_features(train_df, frequency_map)
        train_df = self._convert_to_minmax(train_df, "freq")
        cols = ["min_kcore", "max_kcore", "common_neighbor_count", "common_neighbor_ratio", "min_freq", "max_freq"]
        non_nlp_features = train_df.loc[:, cols]

        q1s_train, q2s_train, train_q_features = self._extract_features(df.copy())

        features_train = np.hstack((train_q_features, nlp_features, non_nlp_features))

        def doPred(model, q1s_train, q2s_train, features_train):
            data_1 = pad_sequences(self.tokenizer.texts_to_sequences(q1s_train), maxlen=self.MAX_SEQUENCE_LENGTH)
            data_2 = pad_sequences(self.tokenizer.texts_to_sequences(q2s_train), maxlen=self.MAX_SEQUENCE_LENGTH)
            pred = model.predict([data_1, data_2, features_train])[0][0]
            return pred

        promises = [self.pool.apply_async(doPred, (model, q1s_train, q2s_train, features_train,)) for model in
                    self.models]
        results = np.array([promise.get(timeout=1) for promise in promises])
        avg = results.mean()
        a = self.TEST_TARGET_MEAN / self.TRAIN_TARGET_MEAN
        b = (1 - self.TEST_TARGET_MEAN) / (1 - self.TRAIN_TARGET_MEAN)
        similarity = (lambda x: a * x / (a * x + b * (1 - x)))(avg)
        return similarity


if __name__ == "__main__":

    predictor = SimilarityPredictor()

    stop = False

    while not stop:

        question1 = input('Enter the first question: ')
        question2 = input('Enter the second question: ')

        if question1 == "" or question2 == "":
            stop = True
            print("Stopping")
            continue
        t_start = time.time()
        similarity = predictor.similarity([question1], [question2])
        print("time took: {}\tSimilarity: {}".format(time.time() - t_start, similarity))
