#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:08:34 2019

@author: du
"""
import pickle
import spacy

def preprocess(doc_set):
    """
    
    Input: document_list
    Purporse: preprocess text (tokenize, remove words that are shorted than two characters ,lemmatizer, removing stopwords, and stemming) Spacy
    Output: preprocessed_text 
    """
    nlp = spacy.load('en_core_web_md',disable = ['parser','tagger','ner']) # remove three tasks so as to speed up the precess 
    nlp.max_length = 1100000
    texts = [] 
    for caption in doc_set:
        doc = nlp(caption)
        lemmanized_list = []
        lemmanized_phrase = ""
        for token in doc:
            if not token.is_punct and not token.is_stop and not token.is_oov: # check is token is not punctutation stop word and in the nlp vocab
                lemmanized_list.append(token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.text) 
        lemmanized_phrase = ' '.join(lemmanized_list)
        texts.append(lemmanized_phrase)
    return(texts)
            


