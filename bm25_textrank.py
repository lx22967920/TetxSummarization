# Coding: utf-8
# Time  : 2019/8/15
# Author: Li Xiang
# @Email: 22967920@qq.com
# coding: utf-8
# time: 2019/7/24
"""
基于TextRank的无监督的单文本摘要。
Usage:
summ = TextRankSummary()
print(summ.get_summary(text))
"""

from summary.textrank import TextRank
from utils import utils
import jieba


class TextRankSummary(object):
    @staticmethod
    def get_summary(texts, n=3):
        texts = utils.get_sentences(texts)
        doc_sents = [jieba.lcut(i) for i in texts]
        rank = TextRank(doc_sents)
        rank.text_rank()
        results = []
        for j in range(len(texts)):
            if j in rank.top_index(n):
                results.append(texts[j])
        summary = "。".join(results) + "。"
        return summary
