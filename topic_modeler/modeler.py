import pandas as pd
import re
import numpy as np
import math
import torch


class Modeler():
    def __init__(self, sentence_series, topic_series):
        self.sentence_series = sentence_series
        self.topic_series = topic_series
        self.feature_matrix = None
        self.word_mapping = None
        self.topic_to_index = None
        self.index_to_topic = None
        self.train_targets = None

        self.build_model()

    def build_model(self):
        print("Converting to lowercase")
        self.convert_series_to_lowercase()
        print("Word to index")
        self.word_to_index()
        print("Building feature matrix")
        self.build_feature_matrix()
        print("Setting topic index")
        self.set_topic_index()
        print("Building train output list")
        self.build_train_output_list()
        print("Build Complete")

    def convert_series_to_lowercase(self):
        for i in self.sentence_series.index:
            curr_sent = self.sentence_series.at[i]
            self.sentence_series.at[i] = curr_sent.lower()

    def get_tokens_from_sentence(self, sentence):
        return re.sub("[^\w]", " ",  sentence).split()

    def word_to_index(self, threshold=1):
        word_mapping = {}
        index = 0

        for i in self.sentence_series.index:
            sentence = self.sentence_series.at[i]
            currWordList = self.get_tokens_from_sentence(sentence)

            for word in currWordList:

                if word not in word_mapping:
                    word_mapping[word] = [-1, 0]

                word_mapping[word][1] += 1

                if word_mapping[word][0] == -1:
                    if word_mapping[word][1] >= threshold:
                        word_mapping[word][0] = index
                        index += 1

        final_mapping = {}
        for word in word_mapping:
            if word_mapping[word][0] != -1:
                final_mapping[word] = word_mapping[word][0]

        self.word_mapping = final_mapping

    def build_feature_matrix(self, threshold=1):

        M = len(self.sentence_series)
        N = len(self.word_mapping)

        feature_matrix = np.zeros((M, N))

        i = 0

        for k in self.sentence_series.index:
            word_count = {}
            sentence = self.sentence_series.at[k]
            currWordList = self.get_tokens_from_sentence(sentence)

            for word in currWordList:
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

                l2_norm = math.sqrt(
                    sum([value ** 2 for key, value in word_count.items()]))

            for word, count in word_count.items():
                if word in self.word_mapping:
                    feature_matrix[i, self.word_mapping[word]] = count/l2_norm

            i += 1

        self.feature_matrix = torch.from_numpy(feature_matrix)
        self.feature_matrix = self.feature_matrix.float()

    def set_topic_index(self):
        topics = self.topic_series.unique()

        topic_dict = {}
        index = 0

        for topic in topics:
            topic_dict[topic] = index
            index += 1

        self.topic_to_index = topic_dict

        self.index_to_topic = {value: key for key,
                               value in self.topic_to_index.items()}

    def build_train_output_list(self):
        M = len(self.topic_series)

        train_output_list = np.zeros(M, dtype=int)

        i = 0

        for k in self.topic_series.index:
            topic = self.topic_series.at[k]
            train_output_list[i] = self.topic_to_index[topic]
            i += 1

        self.train_targets = torch.from_numpy(train_output_list)

    def get_feature_matrix(self, sentence_series):

        M = len(sentence_series)
        N = len(self.word_mapping)

        feature_matrix = np.zeros((M, N))

        i = 0

        for k in sentence_series.index:
            word_count = {}
            sentence = sentence_series.at[k]
            currWordList = self.get_tokens_from_sentence(sentence)

            for word in currWordList:
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

                l2_norm = math.sqrt(
                    sum([value ** 2 for key, value in word_count.items()]))

            for word, count in word_count.items():
                if word in self.word_mapping:
                    feature_matrix[i, self.word_mapping[word]] = count/l2_norm

            i += 1

        return feature_matrix
