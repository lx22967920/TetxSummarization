# Coding: utf-8
# Time  : 2019/8/13
# Author: Li Xiang
# @Email: 22967920@qq.com
"""
基于MMR的无监督的多文本摘要。
Usage:
mmr = MMR(word2vec_path="D:/project/TextRank_Word2Vec/sgns.renmin.bigram-char", dim=300, stopwords_path="D:/project/TextRank/stopwords.txt")
summary =mmr.mmr_summary(text)
"""
import re
import jieba
import numpy as np


class MMR(object):
    def __init__(self, word2vec_path, dim, stopwords_path):
        """
        Args:
            word2vec_path: 预训练词向量路径
            dim:  预训练词向量维度
            stopwords_path:  停止词的路径
        """
        self.word2vec_path = word2vec_path
        self.dim = dim
        self.stopwords_path = stopwords_path
        self.word_2vec = self.load_word2vec(self.word2vec_path)
        with open(self.stopwords_path, encoding="utf-8") as f:
            self.stopwords = f.readlines()
            self.stopwords = [word.replace("\n", "") for word in self.stopwords]

    def get_sentence(self, text):
        """

        Args:
            text: 输入文本

        Returns:
            分词后的句子列表
        """
        line_break = re.compile('[\r\n]')
        delimiter = re.compile('[。？！；]')
        sentences = []
        for line in line_break.split(text):
            line = line.strip()
            if not line:
                continue
            for sent in delimiter.split(line):
                sent = sent.strip()
                if not sent:
                    continue
                sentences.append(sent)
        return sentences

    def cal_score(self, doc):
        """

        Args:
            doc: 句子列表，元素为未分词的句子

        Returns:
            每个句子得分
        """
        doc_set = set(doc)
        scores = {}
        for sent1 in doc_set:
            temp_sent = doc_set - set([sent1])  # 将本句从集合中剔除
            tmp_score = 0.0
            for sent2 in temp_sent:  # 计算本句与其他所有句子的相似度,相加求平均
                tmp_score += self.calculate_cos_similarity(sent1, sent2)
            scores[sent1] = tmp_score / (len(doc) - 1)
        return scores

    def mmr_summary(self, doc, n=5):
        """
        Args:
            doc: 文本
            n: 抽取的句子个数

        Returns:
            摘要
        """
        alpha = 0.7
        summary = []
        doc_list = self.get_sentence(doc)  # 对文档进行分句
        if len(doc_list) <= n:
            print("The text is too short.")
            return
        scores = self.cal_score(doc=doc_list)  # 计算每个句子的分数scores
        while n > 0:
            mmr_result = {}
            for sen in scores:
                mmr_result[sen] = alpha * scores[sen] - (1.0 - alpha) * self.cal_max_similarity(sen, summary)
            selected = max(mmr_result, key=mmr_result.get)
            if selected not in summary:
                summary.append(selected)
            n -= 1
        summary = '。'.join(summary) + "。"
        return summary

    def cal_max_similarity(self, sentence, doc):
        """
        计算sentence与doc中所有句子的相似度，取相似度的最大值
        Args:
            sentence: 句子
            doc: 多个句子列表

        Returns:
            最大相似度
        """
        if doc == []:
            return 0
        max_sim = 0
        for sent in doc:
            tmp_sim = self.calculate_cos_similarity(sentence, sent)
            if tmp_sim > max_sim:
                max_sim = tmp_sim
        return max_sim

    def sentence_vector(self, sen):
        """
        构建句子向量
        Args:
            sen: 分词后的句子

        Returns:
            句向量
        """
        zero = np.zeros(self.dim)
        vec = np.array([self.word_2vec.get(word, zero) for word in sen])
        vec = np.sum(vec, axis=0) / len(vec)
        return vec

    def calculate_cos_similarity(self, sen1, sen2):
        sen1 = self.sentence_vector(self.tokenize_and_rmstopwords(sen1))
        sen2 = self.sentence_vector(self.tokenize_and_rmstopwords(sen2))
        fraction = np.dot(sen1, sen2)
        denominator = (np.linalg.norm(sen1) * (np.linalg.norm(sen2))) + 0.0001
        return fraction / denominator

    def tokenize_and_rmstopwords(self, sentence):
        """
        Args:
            sentence: 句子

        Returns:
            分词，去除停用词
        """
        if sentence != "":
            return list(word for word in list(jieba.cut(sentence)) if word not in self.stopwords)
        else:
            return

    @staticmethod
    def load_word2vec(path):
        """加载预训练word2vec向量
        """
        with open(path, 'r', encoding='utf-8') as f:
            f.readline()
            word_2vec = dict()
            for line in f:
                line_ = line.split(' ')
                key_ = line_[0]
                val_ = np.asanyarray(list(map(float, line_[1:-1])))
                word_2vec[key_] = val_
        return word_2vec