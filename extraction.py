import pyconll
import pandas as pd
import string
import math
import multiprocessing
from manualValidation import *

# extraction functions
def asciiFilter(row):
    '''
    This function delete rows containing letters in clf_form, clf_gov1_form, clf_gov2_form columns
    '''
    if len(set(list(row['clf_form'])).intersection(set(string.printable)))>0:
        return True
    elif len(set(list(row['clf_gov1_form'])).intersection(set(string.printable)))>0:
        return True
    elif len(set(list(row['clf_gov2_form'])).intersection(set(string.printable)))>0:
        return True
    else:
        return False
    
def puncFilter(row):
    '''
    This function delete rows containing punctuation in the phrase
    '''
    punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    if bool(set(row['phrase']) & set(punc)):
        return True
    elif bool(set(row['phrase']) & set(string.punctuation)):
        return True
    else:
        return False

    
def nounPhrase(file_path):
    '''
    This function extract nominal phrase, sent[element1(e.g.numerals) : element2(noun)]excluding any element
    containing letters.
    '''
    f = pyconll.load_from_file(file_path)
    clf_noun_element_ls = []
    clf_phrase = {}

    for sent in f:
        for token in sent:
            if token.deprel == 'mark:clf':
                clf_phrase = {}
                tk_hd = str(int(token.head))
                clf_phrase['clf_form']= token.form
                clf_phrase['clf_id']= token.id
                clf_phrase['clf_pos']= token.xpos
                clf_phrase['clf_gov1_form']= sent[tk_hd].form
                clf_phrase['clf_gov1_id'] = sent[tk_hd].id
                clf_phrase['clf_gov1_pos'] = sent[tk_hd].xpos
                if sent[tk_hd].head == '0':
                    clf_phrase['clf_gov2_form'] = ''
                    clf_phrase['clf_gov2_id'] = ''
                    clf_phrase['clf_gov2_pos'] = '' 
                else:
                    clf_phrase['clf_gov2_form'] = sent[str(sent[tk_hd].head)].form
                    clf_phrase['clf_gov2_id'] = sent[str(sent[tk_hd].head)].id
                    clf_phrase['clf_gov2_pos'] = sent[str(sent[tk_hd].head)].xpos
                    clf_phrase['phrase']=''
                    
                    if int(sent[tk_hd].id) < int(sent[str(sent[tk_hd].head)].id): # try to get the entire nominal phrases
                        for idx in range(int(sent[tk_hd].id)-1, int(sent[str(sent[tk_hd].head)].id)):
                            clf_phrase['phrase'] += sent[idx].form   
                    else:
                        for idx in range(int(sent[str(sent[tk_hd].head)].id)-1, int(sent[tk_hd].id)+1):
                            clf_phrase['phrase'] += sent[idx].form
                            
                clf_noun_element_ls.append(clf_phrase)
    df = pd.DataFrame(clf_noun_element_ls)
    df = df[df.clf_gov2_pos=='NN']      #remove the heads are not nouns
    df = df[df.clf_pos=='M']            #remove the part of speech that is not measurewords
    df = df[~df.apply(asciiFilter, axis=1)] # remove letters
    df = df[~df.apply(puncFilter, axis=1)] # remove punc
    df = df[df.apply(lambda x: bool(set(x.clf_form) & set(x.phrase)), axis=1)] # remove nominal phrases does not contain clf, miss parsed
    
    return df

def nounPhraseP(list_file_paths, cores = 12):
    """
    parallel run of nounPhrase
    """
    with multiprocessing.Pool(cores) as pool: 
        list_df = pool.map(nounPhrase, list_file_paths)
    df = pd.concat(list_df)
    return df

def man_label1(row):
    if row['clf_form'] in man_sortal:
        return 'sortal_classifier'
    elif row['clf_form'] in man_measure:
        return 'mensural_classifier'

def man_label2(row):
    if row['clf_form'] in man_sortal:
        return 'sortal_classifier'
    elif row['clf_form'] in man_measure1:
        return 'measurement'
    elif row['clf_form'] in man_measure2:
        return 'currency'
    else:
        return 'quantity'


def calculate_entropy(type_counter):
    total_count = sum(type_counter.values())
    result = 0.0
    
    for key in type_counter:
        prob = type_counter[key] * 1.0/total_count
        result += (-prob*math.log(prob,2))
    return result

def calculate_normalized_entropy(noun_on_classifier_counters):
    '''
    calculate the conditional entropy of nouns over clfs normalized by number of clfs  
    '''
    num_clf_total = len(noun_on_classifier_counters)
    result = 0.0
    for clf, noun_counter in noun_on_classifier_counters.items():
        result += calculate_entropy(noun_counter)
    result /= num_clf_total
    return result

def mutual_information(noun_counter,noun_on_classifier_counters):
    '''
    This function calculates the mutual information between I(N;C)=H(N)-H(N|C)
    '''
    return calculate_entropy(noun_counter)-calculate_normalized_entropy(noun_on_classifier_counters)


def calculate_conditional_entropy(noun_clf_counter):
    '''
    calculate the conditional entropy of nouns given a certain clf   
    based on the formular on Page 27 from Dye_2017.
    '''
    cond_n_exact_clf = {}
    for key1 in noun_clf_counter.keys():
        total_count = sum(noun_clf_counter[key1].values())
        result = 0.0
        for item in noun_clf_counter[key1]:
            prob = noun_clf_counter[key1][item]*1.0/total_count
            result += (-prob*math.log(prob,2))
        cond_n_exact_clf[key1] = result
    return cond_n_exact_clf
         
        