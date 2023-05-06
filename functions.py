import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import nltk
import wordcloud
import gensim
import seaborn as sns
import re
import collections
import langdetect
import spacy
import textblob
import transformers
import rouge
import lime
import shap
import difflib
import tensorflow
from tensorflow import keras
from sklearn import preprocessing, model_selection, feature_extraction, feature_selection, metrics, manifold, naive_bayes, pipeline
from keras import callbacks, models, layers, preprocessing as kprocessing
from keras import backend as K

###  TEXT ANALYSIS   ###
'''
Plot univariate and bivariate distributions.
'''
def plot_distributions(dtf, x, max_cat=20, top=None, y=None, bins=None, figsize=(10,5)):
    ## univariate
    if y is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(x, fontsize=15)
        ### categorical
        if dtf[x].nunique() <= max_cat:
            if top is None:
                dtf[x].reset_index().groupby(x).count().sort_values(by="index").plot(kind="barh", legend=False, ax=ax).grid(axis='x')
            else:   
                dtf[x].reset_index().groupby(x).count().sort_values(by="index").tail(top).plot(kind="barh", legend=False, ax=ax).grid(axis='x')
            ax.set(ylabel=None)
        ### numerical
        else:
            sns.distplot(dtf[x], hist=True, kde=True, kde_kws={"shade":True}, ax=ax)
            ax.grid(True)
            ax.set(xlabel=None, yticklabels=[], yticks=[])

    ## bivariate
    else:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=figsize)
        fig.suptitle(x, fontsize=15)
        for i in dtf[y].unique():
            sns.distplot(dtf[dtf[y]==i][x], hist=True, kde=False, bins=bins, hist_kws={"alpha":0.8}, axlabel="", ax=ax[0])
            sns.distplot(dtf[dtf[y]==i][x], hist=False, kde=True, kde_kws={"shade":True}, axlabel="", ax=ax[1])
        ax[0].set(title="histogram")
        ax[0].grid(True)
        ax[0].legend(dtf[y].unique())
        ax[1].set(title="density")
        ax[1].grid(True)
    plt.show()

'''
Compute different text length metrics.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
:return
    dtf: input dataframe with 2 new columns
'''

def add_text_length(data, column):
    dtf = data.copy()
    dtf['word_count'] = dtf[column].apply(lambda x: len(nltk.word_tokenize(str(x))))
    dtf['char_count'] = dtf[column].apply(lambda x: len(word) for word in nltk.word_tokenize(str(x)))
    dtf['sentence_count'] = dtf[column].apply(lambda x: len(nltk.sent_tokenize(str(x))))
    dtf['avg_word_length'] = dtf['char_count']/dtf['word_count']
    dtf['avg_sentence_length'] = dtf['word_count'] / dtf['sentence_count']
    print(dtf[['char_count','word_count','sentence_count','avg_word_length','avg_sentence_length']].describe().T[['min','mean','max']])
    return dtf

'''
Creates a list of stopwords.
:parameter
    :param lst_langs: list - ["english", "italian"]
    :param lst_add_words: list - list of new stopwords to add
    :param lst_keep_words: list - list words to keep (exclude from stopwords)
:return
    stop_words: list of stop words
''' 

def create_stopwords(lst_langs=['english'], lst_add_words=[], lst_keep_words=[]):
    lst_stopwords = set()
    for lang in lst_langs:
        lst_stopwords = lst_stopwords.union(set(nltk.corpus.stopwords.words(lang)))
    lst_stopwords = lst_stopwords.union(lst_add_words)
    lst_stopwords = list(set(lst_stopwords)-set(lst_keep_words))
    return sorted(list(set(lst_stopwords)))


'''
Adds a column of preprocessed text.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
:return
    : input dataframe with two new columns
'''

def add_preprocessed_text(data, column, lst_regex=None, punkt=False, lower=False,slang=False, lst_stopwords=None, stemm=False, lemm=False, remove_na=True):
    dtf = data.copy()

    ## apply preprocess
    dtf = dtf[pd.notnull(dtf[column])]
    dtf[column+'_clean'] = dtf[column].apply(lambda x: utils_preprocess_text(x,lst_regex,punkt,lower,slang,lst_stopwords,stemm,lemm))

    ##residuals
    dtf['check']=dtf[column+"_clean"].apply(lambda x: len(x))
    if dtf['check'].min()==0:
        print("---found NAs---")
        print(dtf[[column,column+"_clean"]][dtf['check']==0].head())
        if remove_na is True:
            dtf = dtf[dtf['check']>0]

    return dtf.drop("check", axis=1)


def word_freq(corpus, ngrams=[1,2,3], top=10, figsize=(10,7)):
    lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))
    ngrams = [ngrams] if type(ngrams) is int else ngrams
    
    ## calculate
    dtf_freq = pd.DataFrame()
    for n in ngrams:
        dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, n))
        dtf_n = pd.DataFrame(dic_words_freq.most_common(), columns=["word","freq"])
        dtf_n["ngrams"] = n
        dtf_freq = dtf_freq.append(dtf_n)
    dtf_freq["word"] = dtf_freq["word"].apply(lambda x: " ".join(string for string in x) )
    dtf_freq = dtf_freq.sort_values(["ngrams","freq"], ascending=[True,False])
    
    ## plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x="freq", y="word", hue="ngrams", dodge=False, ax=ax,
                data=dtf_freq.groupby('ngrams')["ngrams","freq","word"].head(top))
    ax.set(xlabel=None, ylabel=None, title="Most frequent words")
    ax.grid(axis="x")
    plt.show()
    return dtf_freq