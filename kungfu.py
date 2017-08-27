# coding:utf-8
import nltk
import re, os
import string
import enchant
import pandas as pd
from contractions import CONTRACTION_MAP
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from pattern.en import suggest
from concurrent import futures
wnl = WordNetLemmatizer()
porter_stemmer = PorterStemmer()


def to_excel(df, i):
    df['old_text'] = df['text']
    for idx, text in enumerate(df['text']):
        text_ = english_only(text)
        print u'原始数据：', text_
        new_text = main(text_)
        print u'加入新的表格的', ','.join(new_text) if new_text else ' '
        df.ix[idx, "text"] = ','.join(new_text) if isinstance(new_text, list) else ' '
    df.to_excel('new_kungfu'+i+'.xlsx')



def tokenize_text(text):
    text = re.sub(r'kung\s*fu', 'kungfu', text)
    text = re.sub(r'(wing\s*tsun)|(wing\s*chun)', 'wingchun', text)
    sentences = nltk.sent_tokenize(text)
    word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences]

    return word_tokens

def remove_characters_before_tokenization(sentence, keep_apostrophes=False):
    sentence = sentence.strip()
    if keep_apostrophes:
        filtered_sentence = re.sub(r"(\w+.\w+.com/\S+)|(@\w+)|(\d+\w*)|(rt)|((\S+/\S+)+)", "", sentence)
        PATTERN = r'[?|$|&|*|￥|%|@|(|)|~|.|,|:|;|[|]|“|”|/|=|_|!|\'|-|]' # add other characters here to
        filtered_sentence1 = re.sub(PATTERN, r'', filtered_sentence)
        result = ' '.join(filtered_sentence1.split())
    else:
        PATTERN = r'[^a-zA-Z0-9 ]' # only extract alpha-numeric characters
        result = re.sub(PATTERN, r'', sentence)

    return result

def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])

    return filtered_tokens

def expand_contractions(sentence, contraction_mapping):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence


def remove_stopwords(tokens):
    stopword_list = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens

def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word
    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens


def check_english(word):
    d = enchant.Dict("en_US")
    return d.check(word)

def english_only(data):
    english_only = ''.join(x for x in data if ord(x) < 128)

    return english_only

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return ''

# 调用enchant，检测出为写错的单词或者不是英文单词，再进行纠正
def spelling_correct(token_list):
    res_correct = []
    for word in token_list:
        if not check_english(word) :
            word_correct = max(dict(suggest(word)), key=lambda k: dict(suggest(word))[k])
            res_correct.append(word_correct)
        else:
            res_correct.append(word)

    return res_correct

# 基本做到同义词转换，"film"," cinema",都可转换成"movie"
def getSynonyms(name):
    res_name = wordnet.synsets(name)[0].lemmas()[0].name() if wordnet.synsets(name) else name
    new_res_name = wordnet.synsets(res_name)[0].lemmas()[0].name() if wordnet.synsets(res_name) else res_name
    return new_res_name

# 同义转换名词
def collect_sysname(res_postag):
    collection = []
    for tag in res_postag:
        if tag[1].startswith('N'):
            collection.append((getSynonyms(tag[0]), tag[1]))
        else:
            collection.append((tag[0], tag[1]))

    return collection

def main(corpus):
    expanded_corpus = expand_contractions(corpus, CONTRACTION_MAP)
    cleaned_corpus = remove_characters_before_tokenization(expanded_corpus, keep_apostrophes=True)
    res_tokennize = tokenize_text(cleaned_corpus.decode('utf-8'))[0]
    res_tokennized = remove_repeated_characters(res_tokennize)
    res_spelling_correct = spelling_correct(res_tokennized)
    tweet = remove_stopwords(res_spelling_correct)
    res_postag = nltk.pos_tag(tweet)
    res_textag = collect_sysname(res_postag)
    res_lemmatize = [wnl.lemmatize(tag[0], get_wordnet_pos(tag[1])).encode('utf-8').lower() for tag in res_textag if
                     get_wordnet_pos(tag[1]) and 3 < len(tag[0]) < 20 ]

    return res_lemmatize


if __name__ == '__main__':
    dir_name = '/Users/zhangxin/Desktop/Twitter数据/'
    files = os.listdir(dir_name)
    n = 1
    # 电脑内存大的话也可多开几个并发
    file_list = [list(files[i:i + 4]) for i in range(0, len(files), 4)]

    for ix, i in enumerate(file_list):
        df1 = pd.read_excel(dir_name+i[0])
        df2 = pd.read_excel(dir_name+i[1])
        df3 = pd.read_excel(dir_name+i[2])
        df3 = pd.read_excel(dir_name+i[3])

        with futures.ThreadPoolExecutor(max_workers=4) as excuter:
            excuter.submit(to_excel, df1, i[0])
            excuter.submit(to_excel, df2, i[1])
            excuter.submit(to_excel, df2, i[2])
            excuter.submit(to_excel, df2, i[3])

    # print getSynonyms('motorbike')
    # print wordnet.synsets('minibike')[0].lemmas()[0].name()


