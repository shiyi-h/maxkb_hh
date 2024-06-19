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
from .excel_parser import RAGFlowExcelParser



class ExcelSplitHandle(BaseSplitHandle):
    def handle(self, file, pattern_list: List, with_filter: bool, limit: int, get_buffer, 
               save_image, save_longtext):
        try:
            if pattern_list and ('markdown' in pattern_list or 'md' in pattern_list or 'mk' in pattern_list):
                table_type = 'markdown'
            else:
                table_type = 'html'
            buffer = get_buffer(file)
            excel = RAGFlowExcelParser()
            tbs, sheetnames, longtexts = excel(file.name, buffer, limit, 256, table_type)
            content = [{"title": sn,'content': tb} for tb,sn in zip(tbs, sheetnames)]
            if len(longtexts)>0:
                save_longtext(longtexts)
        except BaseException as e:
            return {'name': file.name,
                    'content': []}
        return {'name': file.name,
                'content': content
                }

    def support(self, file, get_buffer):
        file_name: str = file.name.lower()
        if file_name.endswith(".xlsx") or file_name.endswith('.xls'):
            return True
        return False