# Coding: utf-8
# Time  : 2019/8/13
# Author: Li Xiang
# @Email: 22967920@qq.com
"""
基于TextRank的无监督的单文本摘要。
Usage:
textrank = TextRank_Word2vec(word2vec_path="sgns.renmin.bigram-char")
top_n_summary = textrank.summary(text, n=3)
"""
import numpy as np
import re
import jieba
import itertools
import pandas as pd
import time
from tqdm import tqdm


class TextRank_Word2vec(object):
    """基于TextRank的文本摘要。
    Word2Vec获得词向量，求和取平均得到句向量，句子相似度由句向量的余弦相似度确定
    Attributes:
        word2vec_path: 预训练的词向量路径
        dim: 预训练词向量的维度，默认为300
    """
    def __init__(self, word2vec_path, dim=300):
        self.word2vec_path = word2vec_path
        self.dim = dim
        try:
            self.word_vec = self.load_word2vec(self.word2vec_path)
        except:
            raise

    def sentence_vector(self, sen):
        """构建句子向量
        Args:
            sen: 分词后的句子
            dim: 词向量的维度
        Returns:
            vec: 句向量
        """
        zero = np.zeros(self.dim)
        vec = np.array([self.word_vec.get(word, zero) for word in sen])
        vec = np.sum(vec, axis=0) / len(vec)
        return vec

    def creat_graph(self, doc_sents):
        """构建句子相似矩阵
        Args:
            doc_sents:分词后的句子列表
        Returns:
            weight_graph: 相似度矩阵
        """
        num = len(doc_sents)
        weight_graph = np.zeros([num, num])
        doc_vec = [self.sentence_vector(i) for i in doc_sents]
        for i, j in itertools.product(range(num), repeat=2):
            if i > j:
                weight_graph[i][j] = self.calculate_similarity(doc_vec[i], doc_vec[j])
        weight_graph += weight_graph.T
        return weight_graph

    def weighted_pagerank(self, weight_graph):
        """迭代计算各个句子的得分
        Args：
            weight_graph: 相似度矩阵，即text_rank中的得分矩阵
        Returns:
            scores: 句子最终得分
        """
        scores = np.zeros(len(weight_graph)) + 0.5  # 初始化句子得分为0.5
        old_scores = np.zeros(len(weight_graph))

        n = 0
        while self.different(scores, old_scores) or n > 200:  # 最大迭代次数200
            old_scores[:][:] = scores[:][:]
            for i in range(len(weight_graph)):
                scores[i] = self.calculate_score(weight_graph, scores, i)
            n += 1
        return scores

    def summary(self, doc, n=3):
        """返回top-n的句子
        Args:
            doc: 原始输入文本
            n: 前n个句子，默认为3
        Returns:
            top:前n个句子
        """
        doc = self.get_sentences(doc)
        if len(doc) <= n:
            print("The text is too short.")
            return '。'.join(doc) + "。"
        doc_sent = [jieba.lcut(sent) for sent in doc]
        similarity_graph = self.creat_graph(doc_sent)
        similarity_graph = np.nan_to_num(similarity_graph)
        scores = self.weighted_pagerank(similarity_graph)
        top_n = np.argpartition(scores, -n)[-n:]  # 取出最大的n个得分索引
        top_n.sort()
        top_n_summary = []
        for m in range(n):
            top_n_summary.append(doc[m])
        top_n_summary = '。'.join(top_n_summary) + "。"
        return top_n_summary

    def criterion(self, result, title):
        """
        计算抽取出来的摘要与标题的相似度
        """
        similarity = 0.0
        result = self.get_sentences(result)
        title = self.sentence_vector(title)
        for sen in result:
            sen = self.sentence_vector(sen)
            similarity += self.calculate_similarity(sen, title)
        return similarity / len(result)

    def get_sentences(self, doc):
        """按照。？！；进行分句
        Args:
            doc： 字符串序列
        Returns:
            sentence: 分句后的句子列表

        """
        line_break = re.compile('[\r\n]')
        delimiter = re.compile('[。？！；]')
        sentences = []
        for line in line_break.split(doc):
            line = line.strip()
            if not line:
                continue
            for sent in delimiter.split(line):
                sent = sent.strip()
                if not sent:
                    continue
                sentences.append(sent)
        return sentences

    @staticmethod
    def different(scores, old_scores):
        """用于判断前后得分是否变化，小于阈值则认为收敛
        Args:
            scores:新得分
            old_scores:上一轮迭代得分
        Returns:True or False
        """
        if np.max(np.fabs(scores - old_scores)) >= 0.0001:
            return True
        return False

    @staticmethod
    def calculate_score(weight_graph, scores, i):
        """计算指定句子得分，更新相似矩阵
        Args:
            weight_graph:相似矩阵
            scores:每个句子的得分
            i:索引
        Returns:
            weight_graph:相似矩阵
        """
        d = 0.85

        fraction = weight_graph[:][i] * scores[:]
        denominator = np.sum(weight_graph, axis=0)
        # 防止分母为0
        denominator = np.nan_to_num(denominator) + 0.0001
        added_score = np.sum(fraction / denominator)

        weight_graph = (1 - d) + d * added_score
        return weight_graph

    @staticmethod
    def calculate_similarity(sen1, sen2):
        """计算两向量余弦相似度
        """
        fraction = np.dot(sen1, sen2)
        denominator = (np.linalg.norm(sen1) * (np.linalg.norm(sen2))) + 0.0001
        return fraction / denominator

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
