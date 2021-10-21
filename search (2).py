#!/usr/bin/env python
# coding: utf-8

# In[37]:


from typing import List
import pickle
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    NamesExtractor,
    Doc
)


# In[38]:


segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
names_extractor = NamesExtractor(morph_vocab)


# In[39]:


class Word:
    token: str
    lemma: str
    pos: str

    def __init__(self, token: str, lemma: str, pos: str) -> None:
        self.token = token
        self.lemma = lemma if lemma is not None else ''
        self.pos = pos

    def __str__(self) -> str:
        return self.token + ' ' + self.lemma + ' ' + self.pos

    def __repr__(self) -> str:
        return self.token + ' ' + self.lemma + ' ' + self.pos

class Text:
    source: str
    text: str
    words: List[Word]
    current: int

    def __init__(self, source: str, text: str) -> None:
        self.current = -1
        self.source = source
        self.text = text
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.current += 1
        if self.current < len(self.words):
            return self.words[self.current]
        self.current = -1
        raise StopIteration

    def parse_text(self) -> None:
        doc = Doc(self.text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        for t in doc.tokens:
            t.lemmatize(morph_vocab)
        self.words = [Word(t.text.lower(), t.lemma, t.pos) for t in doc.tokens if t.pos != 'PUNCT']

    def __str__(self) -> str:
        return self.source + ' ' + self.text

    def __repr__(self) -> str:
        return self.source + ' ' + self.text


# In[40]:


import numpy as np
import math
from sympy import prime, primerange
from collections import defaultdict
from pymorphy2 import MorphAnalyzer
import json


# In[41]:


class Searcher:
    pos_matrix: np.ndarray
    lemmas_matrix: np.ndarray
    tokens_matrix: np.ndarray

    pos_vocab: dict
    lemmas_vocab: dict
    tokens_vocab: dict

    max_len: int
    prime_ids: np.ndarray

    articles: list

    morph: MorphAnalyzer

    def __init__(self, articles: list) -> None:
        self.articles = articles
        self.max_len = len(max(self.articles, key=lambda x: len(x.words)).words)
        self.morph = MorphAnalyzer()
        self.prime_ids = np.array([list(primerange(1, prime(self.max_len)+1)),
        list(primerange(prime(self.max_len)+1, prime(2 * self.max_len)+1)),
        list(primerange(prime(2 * self.max_len)+1 ,prime(3 * self.max_len)+1))])
#         in order of  [POS, lemma, token]
        self.pos_vocab = {x[1]: x[0] for x in enumerate(list(set([word.pos
        for words in self.articles for word in words])))}
        self.lemmas_vocab = {x[1]: x[0] for x in enumerate(list(set([word.lemma
        for words in self.articles for word in words])))}
        self.tokens_vocab = {x[1]: x[0] for x in enumerate(list(set([word.token
        for words in self.articles for word in words])))}

        self.pos_matrix = np.zeros((len(self.articles), len(self.pos_vocab)))
        self.lemmas_matrix = np.zeros((len(self.articles), len(self.lemmas_vocab)))
        self.tokens_matrix = np.zeros((len(self.articles), len(self.tokens_vocab)))
        for i, sent in enumerate(self.articles):
            for j, t in enumerate(sent):
                self.pos_matrix[i][self.pos_vocab[t.pos]]                += math.log(self.prime_ids[0][j])
                self.lemmas_matrix[i][self.lemmas_vocab[t.lemma]]                += math.log(self.prime_ids[1][j])
                self.tokens_matrix[i][self.tokens_vocab[t.token]]                += math.log(self.prime_ids[2][j])
        
        
    def parse_query(self, query) -> (defaultdict, np.ndarray, np.ndarray, np.ndarray):
        pos_query = np.zeros((len(self.pos_vocab), 1))
        lemmas_query = np.zeros((len(self.lemmas_vocab), 1))
        tokens_query = np.zeros((len(self.tokens_vocab), 1))
        query_status = defaultdict(list)
        toks = query.split()
        if len(toks) == 1 and toks[0].find('+') < 0:
            query_status['simple'] = True
        for i, tok in enumerate(toks):
            split_toks = tok.split('+')
            for s_tok in split_toks:
                if s_tok.startswith('''"'''):
                    query_status['token'].append(i)
                    try:
                        tokens_query[self.tokens_vocab[s_tok.strip('''"''')]][0] = i + 1
                    except KeyError:
                        query_status['invalid'] = 'Token %s is not found' % s_tok
                        return query_status, None, None, None
                elif s_tok in self.pos_vocab.keys():
                    query_status['POS'].append(i)
                    pos_query[self.pos_vocab[s_tok]][0] = i + 1
                else:
                    query_status['lemma'].append(i)
                    ana = self.morph.parse(s_tok)
                    poss_lemmas = set([x.normal_form for x in ana])
                    valid = False
                    for lemma in poss_lemmas:
                        try:
                            lemmas_query[self.lemmas_vocab[lemma]][0] = i + 1
                            valid = True
                        except KeyError:
                            continue
                    if not valid:
                        query_status['invalid'] =                        'Lemmas %s are either not valid POS tags or not found' % ' '.join(poss_lemmas)
                        return query_status, None, None, None
        return query_status, pos_query, lemmas_query, tokens_query
    
    def display_results(self, rel) -> dict:
        result = {}
        if len(rel) == 0:
            return {-1: 'Nothing found'}
        for n, idx in enumerate(rel):
            sent = self.articles[idx]
            result[n] = [sent.source, sent.text]
        return result
        
    
    def process_query(self, query) -> list:
        def _simple_search(query, matrix) -> np.ndarray:
            res = np.argwhere(matrix @ query > 0)
            return res[:, 0]
        
        def _create_kernel(query_status) -> np.ndarray:
            kernel = np.zeros((3, 3))
            for row, term in {0: 'POS', 1: 'lemma', 2: 'token'}.items():
                if query_status[term]:
                    for i in query_status[term]:
                        kernel[row][i] = i+1
            return kernel
        
        def _inverted_prime_convolution(primes, kernel) -> np.ndarray:
            kernel = np.delete(kernel, np.argwhere(np.all(kernel[..., :] == 0, axis=0)), axis=1)
            fin = primes.shape[1] + 1 - kernel.shape[1]
            res = np.ones(fin)
            for i in range(fin):
                for j in range(kernel.shape[1]):
                    res[i] /= (math.pow(primes[0][i + j], kernel[0][j])                    * math.pow(primes[1][i + j], kernel[1][j])                    * math.pow(primes[2][i + j], kernel[2][j]))
            return res
        
        def _find_integers(prime_mapping) -> np.ndarray:
            e = math.pow(10, -15)
            mask = np.vectorize(lambda x: abs(x - np.round(x)) < e)(prime_mapping)
            return np.nonzero(np.sum(mask, axis=1))[0]
        
        query_status, pos_query, lemmas_query, tokens_query = self.parse_query(query)
        if query_status['invalid']:
            return {-1: query_status['invalid']}
        if query_status['simple']:
            if query_status['POS']:
                rel = _simple_search(pos_query, self.pos_matrix)
            elif query_status['lemma']:
                rel = _simple_search(lemmas_query, self.lemmas_matrix)
            else:
                rel = _simple_search(tokens_query, self.tokens_matrix)
        else:
            char_vec = np.ones(len(self.articles))
            kernel = _create_kernel(query_status)
            convolution = _inverted_prime_convolution(self.prime_ids, kernel)
            if query_status['POS']:
                char_vec *= np.around(np.exp(self.pos_matrix @ pos_query)).flatten()
            if query_status['lemma']:
                char_vec *= np.around(np.exp(self.lemmas_matrix @ lemmas_query)).flatten()
            if query_status['token']:
                char_vec *= np.around(np.exp(self.tokens_matrix @ tokens_query)).flatten()
            prime_mapping = char_vec.reshape((-1, 1)) @ convolution.reshape((1, -1))
            rel = _find_integers(prime_mapping)
        return json.dumps(self.display_results(rel), ensure_ascii=False).encode('utf8')

        


# In[42]:


with open("articles_parsed_final", "rb") as f:
        a = pickle.load(f)


# In[43]:


searcher = Searcher(a)


# In[44]:


with open('corpus', 'wb') as f:
    pickle.dump(searcher, f)


# In[ ]:




