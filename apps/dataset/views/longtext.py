# coding=utf-8
"""
    @project: maxkb
    @Author：hh
    @file： longtext.py
    @date：2024/5/14 
    @desc:
"""
from drf_yasg.utils import swagger_auto_schema
from rest_framework.decorators import action
from rest_framework.views import APIView
from rest_framework.views import Request

from dataset.models import LongText as LongTextModel
from django.db.models import QuerySet
from django.http import HttpResponse




class LongText(APIView):
    @action(methods=['GET'], detail=False)
    @swagger_auto_schema(operation_summary="获取长文本",
                            operation_id="获取长文本",
                            tags=["长文本"])
    def get(self, request: Request, text_id: str):
        longtext = QuerySet(LongTextModel).filter(id=text_id).first()
        if longtext is None:
            md_format_str = ''
        else:
            content = longtext.text.replace('<br>', '\n')
            md_format_str = f"""{content}"""
        return HttpResponse(md_format_str, status=200, content_type="text/markdown; charset=utf-8")
