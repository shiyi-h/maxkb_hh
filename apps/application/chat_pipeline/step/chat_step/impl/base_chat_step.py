# coding=utf-8
"""
    @project: maxkb
    @Author：hh
    @file： base_chat_step.py
    @date：2024/1/9 18:25
    @desc: 对话step Base实现
"""
import json
import logging
import time
import traceback
import uuid
from typing import List
import re 

from django.db.models import QuerySet
from dataset.models import LongText
from django.http import StreamingHttpResponse
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage
from langchain.schema.messages import HumanMessage, AIMessage
from langchain_core.messages import AIMessageChunk

from application.chat_pipeline.I_base_chat_pipeline import ParagraphPipelineModel
from application.chat_pipeline.pipeline_manage import PipelineManage
from application.chat_pipeline.step.chat_step.i_chat_step import IChatStep, PostResponseHandler
from application.models.api_key_model import ApplicationPublicAccessClient
from common.constants.authentication_type import AuthenticationType
from common.response import result


def add_access_num(client_id=None, client_type=None):
    if client_type == AuthenticationType.APPLICATION_ACCESS_TOKEN.value:
        application_public_access_client = QuerySet(ApplicationPublicAccessClient).filter(id=client_id).first()
        if application_public_access_client is not None:
            application_public_access_client.access_num = application_public_access_client.access_num + 1
            application_public_access_client.intraday_access_num = application_public_access_client.intraday_access_num + 1
            application_public_access_client.save()

def write_context(step, manage, request_token, response_token, all_text):
    step.context['message_tokens'] = request_token
    step.context['answer_tokens'] = response_token
    current_time = time.time()
    step.context['answer_text'] = all_text
    step.context['run_time'] = current_time - step.context['start_time']
    manage.context['run_time'] = current_time - manage.context['start_time']
    manage.context['message_tokens'] = manage.context['message_tokens'] + request_token
    manage.context['answer_tokens'] = manage.context['answer_tokens'] + response_token


def get_long_text(text_id):
    longtext = QuerySet(LongText).filter(id=text_id).first()
    if longtext is None:
        return ''
    title = '展开查看内容'
    content = longtext.text
    content = re.sub('\n\s*\n', '\n', content)
    content = content.replace('\n', '<br>')
    md_format_str = f"""<details><summary>{title}</summary><code>{content}</code></details>"""
    return md_format_str

LONGTEXT_FIND = re.compile(r"/?api/longtext/([a-zA-Z0-9]{8}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{12})[\)\]]?")
LONGTEXT_SEARCH = re.compile(r"/?api/longtext/[a-zA-Z0-9]{8}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{12}[\)\]]?")

def convert_unicode_to_string(input_str):
    unicode_pattern = re.compile(r'(\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})')
    
    match = unicode_pattern.search(input_str)
    def to_char(match_obj):
            code_point = int(match_obj.group(), 16)  
            return chr(code_point)  
    if match:
        converted_str = unicode_pattern.sub(to_char, input_str)
        return converted_str
    else:
        return input_str

def event_content(response,
                  chat_id,
                  chat_record_id,
                  paragraph_list: List[ParagraphPipelineModel],
                  post_response_handler: PostResponseHandler,
                  manage,
                  step,
                  chat_model,
                  message_list: List[BaseMessage],
                  problem_text: str,
                  padding_problem_text: str = None,
                  client_id=None, client_type=None,
                  is_ai_chat: bool = None):
    response_text = ''
    all_text = ''
    long_text_map = {}
    long_text_num = 0
    code_str_num = 0
    long_text_show = ''
    try:
        for chunk in response:
            cont = chunk.content
            if '`' in cont:
                code_str_num += 1
            if long_text_show:
                if '`' in cont:
                    cont = cont.replace('`','`'+long_text_show)
                long_text_show = ''
            response_text += convert_unicode_to_string(cont) 
            ltids = re.findall(LONGTEXT_FIND, response_text)
            if ltids:
                if len(ltids)>long_text_num:
                    ltid = ltids[-1]
                    if ltid in long_text_map:
                        lt = long_text_map.get(ltid)
                    else:
                        long_text_map[ltid] = get_long_text(ltid)
                        lt = long_text_map.get(ltid)
                    long_text_num += 1
                    if code_str_num%2==0 and '`' in cont:
                        with open("/opt/maxkb/app/test.txt",'w') as f:
                            print(f"{time.ctime()} code_str_num%2==0 {lt}", file=f)
                        idx = cont.rfind('`')
                        cont = cont[:idx+1]+lt+cont[idx+1:]
                    else:
                        if code_str_num%2==1:
                            long_text_show = lt
                            with open("/opt/maxkb/app/test.txt",'w') as f:
                                print(f"{time.ctime()} long_text_show {lt}", file=f)
                        else:
                            with open("/opt/maxkb/app/test.txt",'w') as f:
                                print(f"{time.ctime()} no_long_text_show {lt}", file=f)
                            idx = cont.rfind(ltid)+len(ltid)
                            if len(cont)>idx:
                                if cont[idx] in [')', '）', '"' , '’', '”', '>', "》"]:
                                    idx = idx+1
                            cont = cont[:idx]+lt+cont[idx:]
            cont_ex = cont
            if '<details>' in cont_ex:
                lid = cont_ex.find("<code>")+len('<code>')
                rid = cont_ex.rfind('</code>')
                cont_detail = cont_ex[lid:rid]
                if len(cont_detail)>4096:
                    cont_ex = cont_ex.replace(cont_detail, cont_detail[:4042]+'...')
            all_text += cont_ex
            yield 'data: ' + json.dumps({'chat_id': str(chat_id), 'id': str(chat_record_id), 'operate': True,
                                         'content': cont, 'is_end': False}) + "\n\n"

        # 获取token
        if is_ai_chat:
            try:
                request_token = chat_model.get_num_tokens_from_messages(message_list)
                response_token = chat_model.get_num_tokens(response_text)
            except Exception as e:
                request_token = 0
                response_token = 0
        else:
            request_token = 0
            response_token = 0
        step.context['message_tokens'] = request_token
        step.context['answer_tokens'] = response_token
        current_time = time.time()
        step.context['answer_text'] = all_text
        step.context['run_time'] = current_time - step.context['start_time']
        manage.context['run_time'] = current_time - manage.context['start_time']
        manage.context['message_tokens'] = manage.context['message_tokens'] + request_token
        manage.context['answer_tokens'] = manage.context['answer_tokens'] + response_token
        post_response_handler.handler(chat_id, chat_record_id, paragraph_list, problem_text,
                                      all_text, manage, step, padding_problem_text, client_id)
        yield 'data: ' + json.dumps({'chat_id': str(chat_id), 'id': str(chat_record_id), 'operate': True,
                                     'content': '', 'is_end': True}) + "\n\n"
        add_access_num(client_id, client_type)
    except Exception as e:
        logging.getLogger("max_kb_error").error(f'{str(e)}:{traceback.format_exc()}')
        all_text = '异常' + str(e)
        write_context(step, manage, 0, 0, all_text)
        post_response_handler.handler(chat_id, chat_record_id, paragraph_list, problem_text,
                                      all_text, manage, step, padding_problem_text, client_id)
        add_access_num(client_id, client_type)
        yield 'data: ' + json.dumps({'chat_id': str(chat_id), 'id': str(chat_record_id), 'operate': True,
                                     'content': all_text, 'is_end': True}) + "\n\n"


class BaseChatStep(IChatStep):
    def execute(self, message_list: List[BaseMessage],
                chat_id,
                problem_text,
                post_response_handler: PostResponseHandler,
                chat_model: BaseChatModel = None,
                paragraph_list=None,
                manage: PipelineManage = None,
                padding_problem_text: str = None,
                stream: bool = True,
                client_id=None, client_type=None,
                no_references_setting=None,
                **kwargs):
        if stream:
            return self.execute_stream(message_list, chat_id, problem_text, post_response_handler, chat_model,
                                       paragraph_list,
                                       manage, padding_problem_text, client_id, client_type, no_references_setting)
        else:
            return self.execute_block(message_list, chat_id, problem_text, post_response_handler, chat_model,
                                      paragraph_list,
                                      manage, padding_problem_text, client_id, client_type, no_references_setting)

    def get_details(self, manage, **kwargs):
        return {
            'step_type': 'chat_step',
            'run_time': self.context['run_time'],
            'model_id': str(manage.context['model_id']),
            'message_list': self.reset_message_list(self.context['step_args'].get('message_list'),
                                                    self.context['answer_text']),
            'message_tokens': self.context['message_tokens'],
            'answer_tokens': self.context['answer_tokens'],
            'cost': 0,
        }

    @staticmethod
    def reset_message_list(message_list: List[BaseMessage], answer_text):
        result = [{'role': 'user' if isinstance(message, HumanMessage) else 'ai', 'content': message.content} for
                  message
                  in
                  message_list]
        result.append({'role': 'ai', 'content': answer_text})
        return result

    @staticmethod
    def get_stream_result(message_list: List[BaseMessage],
                          chat_model: BaseChatModel = None,
                          paragraph_list=None,
                          no_references_setting=None):
        if paragraph_list is None:
            paragraph_list = []
        directly_return_chunk_list = [AIMessageChunk(content=paragraph.content)
                                      for paragraph in paragraph_list if (
                                              paragraph.hit_handling_method == 'directly_return' and paragraph.similarity >= paragraph.directly_return_similarity)]
        if directly_return_chunk_list is not None and len(directly_return_chunk_list) > 0:
            return iter(directly_return_chunk_list), False
        elif len(paragraph_list) == 0 and no_references_setting.get(
                'status') == 'designated_answer':
            return iter([AIMessageChunk(content=no_references_setting.get('value'))]), False
        if chat_model is None:
            return iter([AIMessageChunk('抱歉，没有配置 AI 模型，无法优化引用分段，请先去应用中设置 AI 模型。')]), False
        else:
            return chat_model.stream(message_list), True

    def execute_stream(self, message_list: List[BaseMessage],
                       chat_id,
                       problem_text,
                       post_response_handler: PostResponseHandler,
                       chat_model: BaseChatModel = None,
                       paragraph_list=None,
                       manage: PipelineManage = None,
                       padding_problem_text: str = None,
                       client_id=None, client_type=None,
                       no_references_setting=None):
        chat_result, is_ai_chat = self.get_stream_result(message_list, chat_model, paragraph_list,
                                                         no_references_setting)
        chat_record_id = uuid.uuid1()
        r = StreamingHttpResponse(
            streaming_content=event_content(chat_result, chat_id, chat_record_id, paragraph_list,
                                            post_response_handler, manage, self, chat_model, message_list, problem_text,
                                            padding_problem_text, client_id, client_type, is_ai_chat),
            content_type='text/event-stream;charset=utf-8')

        r['Cache-Control'] = 'no-cache'
        return r

    @staticmethod
    def get_block_result(message_list: List[BaseMessage],
                         chat_model: BaseChatModel = None,
                         paragraph_list=None,
                         no_references_setting=None):
        if paragraph_list is None:
            paragraph_list = []

        directly_return_chunk_list = [AIMessage(content=paragraph.content)
                                      for paragraph in paragraph_list if
                                      paragraph.hit_handling_method == 'directly_return']
        if directly_return_chunk_list is not None and len(directly_return_chunk_list) > 0:
            return directly_return_chunk_list[0], False
        elif len(paragraph_list) == 0 and no_references_setting.get(
                'status') == 'designated_answer':
            return AIMessage(no_references_setting.get('value')), False
        if chat_model is None:
            return AIMessage('抱歉，没有配置 AI 模型，无法优化引用分段，请先去应用中设置 AI 模型。'), False
        else:
            return chat_model.invoke(message_list), True

    def execute_block(self, message_list: List[BaseMessage],
                      chat_id,
                      problem_text,
                      post_response_handler: PostResponseHandler,
                      chat_model: BaseChatModel = None,
                      paragraph_list=None,
                      manage: PipelineManage = None,
                      padding_problem_text: str = None,
                      client_id=None, client_type=None, no_references_setting=None):
        chat_record_id = uuid.uuid1()
        # 调用模型
        try:
            chat_result, is_ai_chat = self.get_block_result(message_list, chat_model, paragraph_list,
                                                            no_references_setting)
            if is_ai_chat:
                request_token = chat_model.get_num_tokens_from_messages(message_list)
                response_token = chat_model.get_num_tokens(chat_result.content)
            else:
                request_token = 0
                response_token = 0
            self.context['message_tokens'] = request_token
            self.context['answer_tokens'] = response_token
            current_time = time.time()
            
            cont = chat_result.content
            long_text_map = {}
            ltapis = re.findall(LONGTEXT_FIND, cont)
            if ltapis:
                for ltapi in ltapis:
                    if ltapi in long_text_map:
                        continue
                    ltid = re.search(LONGTEXT_SEARCH, ltapi).group(1)
                    long_text_map[ltapi] = get_long_text(ltid)
                    cont = cont.replace(ltapi, ltapi+long_text_map[ltapi])
            if '<detail>' in cont:
                pattern = re.compile(r'<code>(.*?)</code>', re.DOTALL)
                cont_details = pattern.findall(cont)
                for cd in cont_details:
                    if len(cd)>4096:
                        cont = cont.replace(cd, cd[:4042]+'...')
        
            self.context['answer_text'] = cont
            self.context['run_time'] = current_time - self.context['start_time']
            manage.context['run_time'] = current_time - manage.context['start_time']
            manage.context['message_tokens'] = manage.context['message_tokens'] + request_token
            manage.context['answer_tokens'] = manage.context['answer_tokens'] + response_token
            post_response_handler.handler(chat_id, chat_record_id, paragraph_list, problem_text,
                                        cont, manage, self, padding_problem_text, client_id)
            add_access_num(client_id, client_type)
        
            return result.success({'chat_id': str(chat_id), 'id': str(chat_record_id), 'operate': True,
                                'content': cont, 'is_end': True})
        except Exception as e:
            all_text = '异常' + str(e)
            write_context(self, manage, 0, 0, all_text)
            post_response_handler.handler(chat_id, chat_record_id, paragraph_list, problem_text,
                                          all_text, manage, self, padding_problem_text, client_id)
            add_access_num(client_id, client_type)
            return result.success({'chat_id': str(chat_id), 'id': str(chat_record_id), 'operate': True,
                                   'content': all_text, 'is_end': True})
