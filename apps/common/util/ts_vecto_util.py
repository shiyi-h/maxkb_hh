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
                      "怎么", "哪个", "哪些", "啥", "相关", "都"])

original_freq = deepcopy(jieba.dt.FREQ)


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

def pretoken(txt, num=False, stpwd=True):
    stop_words = set(["请问", "您", "你", "我", "他", "是", "的", "就", "有", "于", "及", "即", "在", 
                      "为", "最", "有", "从", "以", "了", "将", "与", "吗", "吧", "中", "#", "什么", 
                      "怎么", "哪个", "哪些", "啥", "相关", "都"])
    rewt = [
    ]
    for p, r in rewt:
        txt = re.sub(p, r, txt)
    keywords = set(jieba.analyse.textrank(txt, topK=None)+jieba.analyse.extract_tags(txt, topK=None))
    for key in keywords:
        if key not in stop_words:
            jieba.add_word(key)
    res = []
    for tk in jieba.lcut(txt, HMM=True, use_paddle=True):
        if (stpwd and tk in stop_words) or (
                re.match(r"[0-9]$", tk) and not num):
            continue
        if tk in keywords:
            res.append(tk)
    # resume freq
    jieba.dt.FREQ = original_freq
    return res


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


def to_query(text: str):
    # 获取不分词的数据
    word_list = get_word_list(text)
    # 获取关键词关系
    word_dict = to_word_dict(word_list, text)
    # 替换字符串
    text = replace_word(word_dict, text)
    extract_tags = set(jieba.analyse.textrank(text, topK=10)+jieba.analyse.extract_tags(text, topK=10))
    result = "|".join([get_key_by_word_dict(word, word_dict) for word in extract_tags if
                       (not remove_chars.__contains__(word)) and (not word in stop_words)])
    # 删除词库
    for word in word_list:
        jieba.del_word(word)
    return result

