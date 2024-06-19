# coding=utf-8
"""
    @project: maxkb
    @Author：hh
    @file： text_split_handle.py
    @date：2024/5/13 
    @desc:
"""
from typing import List
from common.handle.base_split_handle import BaseSplitHandle
from .ppt_parser import RAGFlowPptParser



class PptSplitHandle(BaseSplitHandle):
    def handle(self, file, pattern_list: List, with_filter: bool, limit: int, get_buffer, 
               save_image, save_longtext):
        try:
            buffer = get_buffer(file)
            ppt = RAGFlowPptParser()
            txts, image_list = ppt(file.name, buffer)
            content = [{"title": '','content': txt} for txt in txts]
            if len(image_list)>0:
                save_image(image_list)
        except BaseException as e:
            return {'name': file.name,
                    'content': []}
        return {'name': file.name,
                'content': content
                }

    def support(self, file, get_buffer):
        file_name: str = file.name.lower()
        if file_name.endswith(".ppt") or file_name.endswith('.pptx'):
            return True
        return False