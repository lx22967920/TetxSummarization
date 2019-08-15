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

from textrank import TextRank
import jieba
import re


class TextRankSummary(object):
    def get_sentences(self, doc):
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

    def get_summary(self, texts, n=3):
        texts = self.get_sentences(texts)
        doc_sents = [jieba.lcut(i) for i in texts]
        rank = TextRank(doc_sents)
        rank.text_rank()
        results = []
        for j in range(len(texts)):
            if j in rank.top_index(n):
                results.append(texts[j])
        summary = "。".join(results) + "。"
        return summary
