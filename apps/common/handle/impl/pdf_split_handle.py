# coding=utf-8
"""
    @project: maxkb
    @Author：hh
    @file： pdf_split_handle.py
    @date：2024/5/15 18:19
    @desc:
"""
import re
from typing import List

from .pdf_parser import RAGFlowPdfParser, CommonPdfParser
from common.handle.base_split_handle import BaseSplitHandle
from common.util.split_model import SplitModel

default_pattern_list = [re.compile('(?<=^)# .*|(?<=\\n)# .*'),
                        re.compile('(?<=\\n)(?<!#)## (?!#).*|(?<=^)(?<!#)## (?!#).*'),
                        re.compile("(?<=\\n)(?<!#)### (?!#).*|(?<=^)(?<!#)### (?!#).*"),
                        re.compile("(?<=\\n)(?<!#)#### (?!#).*|(?<=^)(?<!#)#### (?!#).*"),
                        re.compile("(?<=\\n)(?<!#)##### (?!#).*|(?<=^)(?<!#)##### (?!#).*"),
                        re.compile("(?<=\\n)(?<!#)###### (?!#).*|(?<=^)(?<!#)###### (?!#).*"),
                        re.compile("(?<!\n)\n\n+")]

def fix_engs(t):
    engs = re.findall(r"[a-zA-Z0-9]{2,}", t)
    if engs:
        for en in engs:
            if re.search(r"[a-zA-Z]", t):
                half = len(en)//2
                if len(en)/2==half and en[0]==en[1] and en[-1]==en[-2] and \
                    ((half%2==0 and en[half]==en[half+1]) or (half%2!=0 and en[half]==en[half-1])):
                    t = t.replace(en,en[::2])
    return t


def tokenize_presentation(res):
    pn_dict = set()
    new = []
    for d in res:
        if d['page_number'] not in pn_dict:
            pn_dict.add(d['page_number'])
            new.append([])
        text = re.sub('[\.。….]{3,}', '...', fix_engs(d['text'])).rstrip('\n')
        new[-1].append(text)
    res = ['\n'.join(lst) for lst in new]
    return res

class OCRPdf(RAGFlowPdfParser):
    def __call__(self, filename, binary=None, from_page=0,
                 to_page=100000, zoomin=3, callback=None, img_list=[]):
        callback = callback_func
        callback(msg="OCR is running...")
        self.__images__(
            filename if not binary else binary,
            zoomin,
            from_page,
            to_page,
            callback
        )
        callback(msg="OCR finished")

        self._layouts_rec(zoomin)
        callback(0.63, "Layout analysis finished.")
        self._table_transformer_job(zoomin)
        callback(0.65, "Table analysis finished.")
        self._text_merge()
        callback(0.67, "Text merging finished")
        res, img_list = self._extract_table_figure(True, zoomin, True, img_list)
        #self._naive_vertical_merge()
        #self._concat_downward()
        #self._filter_forpages()
        return res, img_list

def callback_func(prog=None, msg=""):
    print(msg)

class PdfSplitHandle(BaseSplitHandle):
    def handle(self, file, pattern_list: List, with_filter: bool, limit: int, get_buffer,save_image, save_longtext):
        try:
            if pattern_list and ('ocr' in pattern_list or 'OCR' in pattern_list):
                pdf = OCRPdf()
            else:
                pdf = CommonPdfParser()
            img_list = []
            buffer = get_buffer(file)
            res, img_list = pdf(file.name, buffer, img_list=img_list)
            content = tokenize_presentation(res)
            if limit:
                content = '\n'.join(content)
                if pattern_list is not None and len(pattern_list) > 0:
                    split_model = SplitModel(pattern_list, with_filter, limit)
                else:
                    split_model = SplitModel(default_pattern_list, with_filter=with_filter, limit=limit)
                content = split_model.parse(content)
            else:
                content = [{"title": '','content': con} for con in content]
            if img_list:
                save_image(img_list)
        except BaseException as e:
            return {'name': file.name,
                    'content': []}
        return {'name': file.name,
                'content': content
                }

    def support(self, file, get_buffer):
        file_name: str = file.name.lower()
        if file_name.endswith(".pdf"):
            return True
        return False
