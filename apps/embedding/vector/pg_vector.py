# coding=utf-8
"""
    @project: maxkb
    @Author：hh
    @file： pg_vector.py
    @date：2023/10/19 15:28
    @desc:
"""
import json
import os
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List
import re

from django.db.models import QuerySet
from langchain_community.embeddings import HuggingFaceEmbeddings
from dataset.models import Paragraph, Status
from common.config.embedding_config import EmbeddingModel
from common.db.search import generate_sql_by_query_dict
from common.db.sql_execute import select_list
from common.util.file_util import get_file_content
from common.util.ts_vecto_util import to_search_vector, to_query, re_content
from common.util.rsa_util import rsa_long_decrypt
from embedding.models import Embedding, SourceType, SearchMode
from embedding.vector.base_vector import BaseVectorStore
from embedding.llm.raptor import RecursiveAbstractiveProcessing4TreeOrganizedRetrieval as Raptor
from setting.models import Model
from setting.models_provider.constants.model_provider_constants import ModelProvideConstants
from smartdoc.conf import PROJECT_DIR


class PGVector(BaseVectorStore):

    def delete_by_source_ids(self, source_ids: List[str], source_type: str):
        QuerySet(Embedding).filter(source_id__in=source_ids, source_type=source_type).delete()

    def update_by_source_ids(self, source_ids: List[str], instance: Dict):
        QuerySet(Embedding).filter(source_id__in=source_ids).update(**instance)

    def embed_documents(self, text_list: List[str]):
        text_list = self.re_content(text_list)
        embedding = EmbeddingModel.get_embedding_model()
        return embedding.embed_documents(text_list)

    def embed_query(self, text: str):
        text = self.re_content(text)
        embedding = EmbeddingModel.get_embedding_model()
        return embedding.embed_query(text)

    def vector_is_create(self) -> bool:
        # 项目启动默认是创建好的 不需要再创建
        return True

    def vector_create(self):
        return True

    def _save(self, text, source_type: SourceType, dataset_id: str, document_id: str, paragraph_id: str, source_id: str,
              is_active: bool,
              embedding: HuggingFaceEmbeddings):
        return_long = 'mix'
        text = re_content(text)
        if hasattr(embedding, 'long_pooling'):
            text_embedding = embedding.embed_query(text, return_long=return_long)
            if return_long=='pooling':
                text_embedding = [text_embedding]
        else:
            text_embedding = [embedding.embed_query(text)]
        for emb in text_embedding:
            embedding = Embedding(id=uuid.uuid1(),
                                dataset_id=dataset_id,
                                document_id=document_id,
                                is_active=is_active,
                                paragraph_id=paragraph_id,
                                source_id=source_id,
                                embedding=emb,
                                source_type=source_type)
            embedding.save()
        return True

    def _batch_save(self, text_list: List[Dict], embedding: HuggingFaceEmbeddings, raptor_model):
        texts = [re_content(row.get('text')) for row in text_list]
        return_long = 'mix'
        if hasattr(embedding, 'long_pooling'):
            embeddings = embedding.embed_documents(texts, return_long=return_long)
        else:
            embeddings = embedding.embed_documents(texts)
        extra = []
        if raptor_model:
            model = QuerySet(Model).filter(id=raptor_model).first()
            chat_model = ModelProvideConstants[model.provider].value.get_model(model.model_type, model.model_name,
                                                                               json.loads(
                                                                                   rsa_long_decrypt(model.credential)))
            raptor = Raptor(chat_model, embedding)
            chunks = [(text_list[i].get('text'), embeddings[i]) for i in range(len(text_list))]
            result = raptor(chunks, 10086)
            for tid, embd in result['table']:
                if isinstance(embeddings[tid][0], list):
                    embeddings[tid].append(embd)
                else:
                    embeddings[tid] = [embd, embeddings[tid]]
            document_id = text_list[0].get('document_id')
            dataset_id = text_list[0].get('dataset_id')
            for text, embd in result['text']:
                extra.append({'id':uuid.uuid1(), 'content': text, 'embd':embd, 
                              'document_id':document_id, 'dataset_id':dataset_id})
        embedding_list = []
        for index in range(0, len(text_list)):
            embs = embeddings[index]
            embs = embs if isinstance(embs[0], list) else [embs]
            for emb in embs:
                embedding_list.append(Embedding(id=uuid.uuid1(),
                                                document_id=text_list[index].get('document_id'),
                                                paragraph_id=text_list[index].get('paragraph_id'),
                                                dataset_id=text_list[index].get('dataset_id'),
                                                is_active=text_list[index].get('is_active', True),
                                                source_id=text_list[index].get('source_id'),
                                                source_type=text_list[index].get('source_type'),
                                                embedding=emb))
        paragraph_list = []
        for d in extra:
            paragraph_list.append(Paragraph(id=d['id'], document_id=d['document_id'], content=d.get("content"),
                                  dataset_id=d['dataset_id'], title='', title_vector='',
                                  content_vector=to_search_vector(d.get("content")), 
                                  status=Status.raptor))
            embedding_list.append(Embedding(id=uuid.uuid1(),
                                            document_id=d['document_id'],
                                            paragraph_id=d['id'],
                                            dataset_id=d.get('dataset_id'),
                                            is_active=True,
                                            source_id=d['id'],
                                            source_type=1,
                                            embedding=d['embd']))
        if paragraph_list:
            QuerySet(Paragraph).bulk_create(paragraph_list)
        QuerySet(Embedding).bulk_create(embedding_list) if len(embedding_list) > 0 else None
        return True

    def hit_test(self, query_text, dataset_id_list: list[str], exclude_document_id_list: list[str], top_number: int,
                 similarity: float,
                 search_mode: SearchMode,
                 embedding: HuggingFaceEmbeddings):
        if dataset_id_list is None or len(dataset_id_list) == 0:
            return []
        query_text = re_content(query_text)
        exclude_dict = {}
        embedding_query = embedding.embed_query(query_text)
        query_set = QuerySet(Embedding).filter(dataset_id__in=dataset_id_list, is_active=True)
        if exclude_document_id_list is not None and len(exclude_document_id_list) > 0:
            exclude_dict.__setitem__('document_id__in', exclude_document_id_list)
        query_set = query_set.exclude(**exclude_dict)
        for search_handle in search_handle_list:
            if search_handle.support(search_mode):
                return search_handle.handle(query_set, query_text, embedding_query, top_number, similarity, search_mode)

    def query(self, query_text: str, query_embedding: List[float], dataset_id_list: list[str],
              exclude_document_id_list: list[str],
              exclude_paragraph_list: list[str], is_active: bool, top_n: int, similarity: float,
              search_mode: SearchMode):
        query_text = re_content(query_text)
        exclude_dict = {}
        if dataset_id_list is None or len(dataset_id_list) == 0:
            return []
        query_set = QuerySet(Embedding).filter(dataset_id__in=dataset_id_list, is_active=is_active)
        if exclude_document_id_list is not None and len(exclude_document_id_list) > 0:
            exclude_dict.__setitem__('document_id__in', exclude_document_id_list)
        if exclude_paragraph_list is not None and len(exclude_paragraph_list) > 0:
            exclude_dict.__setitem__('paragraph_id__in', exclude_paragraph_list)
        query_set = query_set.exclude(**exclude_dict)
        for search_handle in search_handle_list:
            if search_handle.support(search_mode):
                return search_handle.handle(query_set, query_text, query_embedding, top_n, similarity, search_mode)

    def update_by_source_id(self, source_id: str, instance: Dict):
        QuerySet(Embedding).filter(source_id=source_id).update(**instance)

    def update_by_paragraph_id(self, paragraph_id: str, instance: Dict):
        QuerySet(Embedding).filter(paragraph_id=paragraph_id).update(**instance)

    def update_by_paragraph_ids(self, paragraph_id: str, instance: Dict):
        QuerySet(Embedding).filter(paragraph_id__in=paragraph_id).update(**instance)

    def delete_by_dataset_id(self, dataset_id: str):
        QuerySet(Embedding).filter(dataset_id=dataset_id).delete()

    def delete_by_dataset_id_list(self, dataset_id_list: List[str]):
        QuerySet(Embedding).filter(dataset_id__in=dataset_id_list).delete()

    def delete_by_document_id(self, document_id: str):
        QuerySet(Embedding).filter(document_id=document_id).delete()
        return True

    def delete_bu_document_id_list(self, document_id_list: List[str]):
        return QuerySet(Embedding).filter(document_id__in=document_id_list).delete()

    def delete_by_source_id(self, source_id: str, source_type: str):
        QuerySet(Embedding).filter(source_id=source_id, source_type=source_type).delete()
        return True

    def delete_by_paragraph_id(self, paragraph_id: str):
        QuerySet(Embedding).filter(paragraph_id=paragraph_id).delete()

    def delete_by_paragraph_ids(self, paragraph_ids: List[str]):
        QuerySet(Embedding).filter(paragraph_id__in=paragraph_ids).delete()


class ISearch(ABC):
    @abstractmethod
    def support(self, search_mode: SearchMode):
        pass

    @abstractmethod
    def handle(self, query_set, query_text, query_embedding, top_number: int,
               similarity: float, search_mode: SearchMode):
        pass


class EmbeddingSearch(ISearch):
    def handle(self,
               query_set,
               query_text,
               query_embedding,
               top_number: int,
               similarity: float,
               search_mode: SearchMode):
        exec_sql, exec_params = generate_sql_by_query_dict({'embedding_query': query_set},
                                                           select_string=get_file_content(
                                                               os.path.join(PROJECT_DIR, "apps", "embedding", 'sql',
                                                                            'embedding_search.sql')),
                                                           with_table_name=True)
        embedding_model = select_list(exec_sql,
                                      [json.dumps(query_embedding), *exec_params, similarity, top_number])
        return embedding_model

    def support(self, search_mode: SearchMode):
        return search_mode.value == SearchMode.embedding.value


class KeywordsSearch(ISearch):
    def handle(self,
               query_set,
               query_text,
               query_embedding,
               top_number: int,
               similarity: float,
               search_mode: SearchMode):
        exec_sql, exec_params = generate_sql_by_query_dict({'keywords_query': query_set},
                                                           select_string=get_file_content(
                                                               os.path.join(PROJECT_DIR, "apps", "embedding", 'sql',
                                                                            'keywords_search.sql')),
                                                           with_table_name=False)
        embedding_model = select_list(exec_sql,
                                      [to_query(query_text), *exec_params, similarity, top_number])
        return embedding_model

    def support(self, search_mode: SearchMode):
        return search_mode.value == SearchMode.keywords.value


class BlendSearch(ISearch):
    def handle(self,
               query_set,
               query_text,
               query_embedding,
               top_number: int,
               similarity: float,
               search_mode: SearchMode):
        exec_sql, exec_params = generate_sql_by_query_dict({'embedding_query': query_set},
                                                           select_string=get_file_content(
                                                               os.path.join(PROJECT_DIR, "apps", "embedding", 'sql',
                                                                            'blend_search.sql')),
                                                           with_table_name=False)
        embedding_model = select_list(exec_sql,
                                      [json.dumps(query_embedding), *exec_params, to_query(query_text), *exec_params, 
                                       similarity, top_number])
        return embedding_model

    def support(self, search_mode: SearchMode):
        return search_mode.value == SearchMode.blend.value


search_handle_list = [EmbeddingSearch(), KeywordsSearch(), BlendSearch()]
