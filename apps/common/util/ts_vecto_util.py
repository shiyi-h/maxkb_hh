# coding=utf-8
"""
    @project: maxkb
    @Author：hh
    @file： ts_vecto_util.py
    @date：2024/4/16 15:26
    @desc:
"""
import re
import uuid
from typing import List
import math
import jieba
import jieba.posseg
from jieba import analyse
from copy import deepcopy

from common.util.split_model import group_by


jieba_word_list_cache = [chr(item) for item in range(38, 84)]

for jieba_word in jieba_word_list_cache:
    jieba.add_word('#' + jieba_word + '#')
# r"(?i)\b(?:https?|ftp|tcp|file)://[^\s]+\b",
# 某些不分词数据
# r'"([^"]*)"'
word_pattern_list = [r"v\d+.\d+.\d+",
                     r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}"]

remove_chars = '\n , :\'<>！@#￥%……&*（）!@#$%^&*()： ；，/"./'

jieba_remove_flag_list = ['x', 'w']

stop_words = set(["请问", "您", "你", "我", "他", "是", "的", "就", "有", "于", "及", "即", "在", 
                      "为", "最", "有", "从", "以", "了", "将", "与", "吗", "吧", "中", "#", "什么", 
                      "怎么", "哪个", "哪些", "啥", "相关", "都", "多少"])

original_freq = deepcopy(jieba.dt.FREQ)


RE_PAT = [re.compile(r'!\[[^\]]*\]\(/api/image/[a-zA-Z0-9]{8}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{12}\)'),
        re.compile(r'\[[^\]]*\]\(/api/longtext/[a-zA-Z0-9]{8}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{12}\)'),
        re.compile(r"</?(table|td|caption|tr|th)( [^<>]*)?>"),
        re.compile(r' {2,}')]
    
def re_content(text):
    if isinstance(text, str):
        for pat in RE_PAT:
            text = pat.sub(' ', text)
        return text 
    else:
        text = [re_content(ti) for ti in text]
        return text

def get_word_list(text: str):
    result = []
    for pattern in word_pattern_list:
        word_list = re.findall(pattern, text)
        for child_list in word_list:
            for word in child_list if isinstance(child_list, tuple) else [child_list]:
                # 不能有: 所以再使用: 进行分割
                if word.__contains__(':'):
                    item_list = word.split(":")
                    for w in item_list:
                        result.append(w)
                else:
                    result.append(word)
    return result


def replace_word(word_dict, text: str):
    for key in word_dict:
        text = re.sub('(?<!#)' + word_dict[key] + '(?!#)', key, text)
    return text


def get_word_key(text: str, use_word_list):
    j_word = next((j for j in jieba_word_list_cache if j not in text and all(j not in used for used in use_word_list)),
                  None)
    if j_word:
        return j_word
    j_word = str(uuid.uuid1())
    jieba.add_word(j_word)
    return j_word


def to_word_dict(word_list: List, text: str):
    word_dict = {}
    for word in word_list:
        key = get_word_key(text, set(word_dict))
        word_dict['#' + key + '#'] = word
    return word_dict


def get_key_by_word_dict(key, word_dict):
    v = word_dict.get(key)
    if v is None:
        return key
    return v

def token_merge(tks):
    def oneTerm(t): return len(t) == 1 or re.match(r"[0-9a-z]{1,2}$", t)
    res = []
    i = 0 
    while i<len(tks):
        if oneTerm(tks[i]):
            s1 = res[-1] if res else ''
            s2 = tks[i+1] if i+1<len(tks) else ''
            ts = []
            if s1 and s2:
                ts.append([s1+tks[i]+s2, 2, 0])
            elif s1:
                ts.append([s1+tks[i], 1, 0])
            elif s2:
                ts.append([tks[i]+s2, 2, 1])
            merge = False
            for t, _i, _f in ts:
                if t in original_freq:
                    if _f==0:
                        res[-1] = t 
                    else:
                        res.append(t)
                    i += _i
                    merge = True
                    break
            if not merge:
                res.append(tks[i])
                i+=1
        else:
            res.append(tks[i])
            i+=1
    return res


def weight_norm(dic):
    norm = math.sqrt(sum(map(lambda x:math.pow(x,2), dic.values())))
    for k in dic:
        dic[k] = dic[k]/norm
    return dic

def pretoken(txt, num=False, stpwd=True, withWeight=False):
    # resume freq 
    jieba.dt.FREQ=original_freq
    
    for s in stop_words:
        txt = txt.replace(s, f' {s} ')
    tks = '#@#'.join([s.strip().lower() for s in jieba.lcut(txt, HMM=True, use_paddle=True) if s.strip()])
    tks = tks.replace('#@#_#@#','_').split('#@#')
    
    tks = [s for s in token_merge(tks) if not (stpwd and s in stop_words) and not (re.match(r"[0-9]$", s) and not num)]
    for tk in tks:
        jieba.add_word(tk)
    
    if withWeight:
        d1 = weight_norm(dict(jieba.analyse.textrank(txt, topK=None, withWeight=True)))
        d2 = weight_norm(dict(jieba.analyse.extract_tags(txt, topK=None, withWeight=True)))
        for k in d2:
            if k in d1:
                d1[k] = max(d1[k], d2[k])
            else:
                d1[k] = d2[k]
        res = d1.copy()
        tks = [tk for tk in tks if tk in d1]
        for t1, t2 in zip(tks[:-1], tks[1:]):
            if t1 in d1 and t2 in d1:
                res[f'"{t1} {t2}"'] = (d1[t1]+d1[t2])/2
        return res
    else:
        keywords = set(jieba.analyse.textrank(txt, topK=None)+jieba.analyse.extract_tags(txt, topK=None))
        tks = [tk for tk in tks if tk in keywords]
        return tks


def to_ts_vector(text: str):
    # 获取不分词的数据
    word_list = get_word_list(text)
    # 获取关键词关系
    word_dict = to_word_dict(word_list, text)
    # 替换字符串
    text = replace_word(word_dict, text)
    # 分词
    result = pretoken(text, num=True)
    result_ = [{'word': get_key_by_word_dict(result[index], word_dict), 'index': index} for index in
               range(len(result))]
    result_group = group_by(result_, lambda r: r['word'])
    return " ".join(
        [f"{key.lower()}:{','.join([str(item['index'] + 1) for item in result_group[key]][:20])}" for key in
         result_group if
         not remove_chars.__contains__(key) and len(key.strip()) >= 0])
    
def to_search_vector(text: str):
    # 获取不分词的数据
    word_list = get_word_list(text)
    # 获取关键词关系
    word_dict = to_word_dict(word_list, text)
    # 替换字符串
    text = replace_word(word_dict, text)
    # 分词
    result = pretoken(text, num=True)
    return ' '.join([key for key in result if not remove_chars.__contains__(key) and len(key.strip()) >= 0])


def to_query(text: str):
    # 获取不分词的数据
    word_list = get_word_list(text)
    # 获取关键词关系
    word_dict = to_word_dict(word_list, text)
    # 替换字符串
    text = replace_word(word_dict, text)
    extract_result = pretoken(text, num=True, withWeight=True)
    result = []
    for k,v in extract_result.items():
        result.append(f'title_vector:{k}^{v*2:.4f}')
        result.append(f'content_vector:{k}^{v:.4f}')
    result = ' OR '.join(result)
    # result = "|".join([get_key_by_word_dict(word, word_dict) for word in extract_result if
    #                    (not remove_chars.__contains__(word)) and (not word in stop_words)])
    # 删除词库
    for word in word_list:
        jieba.del_word(word)
    return result

