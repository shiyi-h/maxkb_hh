#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import re
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, ALL_COMPLETED, wait
from threading import Lock
from typing import Tuple
import umap
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
import tiktoken
from langchain.schema import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from smartdoc.const import CONFIG
import time

max_kb = logging.getLogger("max_kb")

DEFALUT_PROMPT = """请用中文总结以下段落。小心数字，不要编造。段落如下：
      {content}
以上就是你需要总结的内容。"""

DEFALUT_PROMPT_LIMIT = """请用中文总结以下段落。段落如下：
{content}
要求：
- 小心数字，不要编造。
- 字数限制在{limit}内。"""

DEFALUT_TABLE_PROMPT = """请总结以下表格：
{content}"""

encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
def truncate_text(string: str, max_len: int) -> int:
    """Returns truncated text if the length of text exceed max_len."""
    return encoder.decode(encoder.encode(string)[:max_len])

def summary_text(chat_model, string: str, max_len: int):
    if len(max_len)<10:
        return encoder.decode(encoder.encode(string)[:max_len])
    msg = [HumanMessage(content=DEFALUT_PROMPT_LIMIT.format(content=string, limit=max_len))]
    res = chat_model.invoke(msg).content 
    return res
    

class RecursiveAbstractiveProcessing4TreeOrganizedRetrieval:
    def __init__(self, llm_model: BaseChatModel, embd_model):
        self._max_cluster = CONFIG.get('RAPTOR_MAX_CLUSTER', 64)
        self._llm_model = llm_model
        self._embd_model = embd_model
        self._threshold = CONFIG.get('RAPTOR_THRESHOLD', 0.1)
        self._prompt = DEFALUT_PROMPT
        self._max_token = CONFIG.get('RAPTOR_MAX_TOKEN', 256)
        self._defalut_max_len = CONFIG.get('RAPTOR_DEFALUT_MAX_LEN', 20000)
        self.set_params(temperature=0.3)
        max_kb.info(f'使用Raptor，threshold:{self._threshold}，max_cluster:{self._max_cluster}')
        
    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self._llm_model, k):
                setattr(self._llm_model, k, v)

    def _get_optimal_clusters(self, embeddings: np.ndarray, random_state:int):
        max_clusters = min(self._max_cluster, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        optimal_clusters = n_clusters[np.argmin(bics)]
        return optimal_clusters
    
    def _table_check(self, content):
        if '<table>' in content and '</table>' in content:
            return True 
        if len(content.split('|'))>=6:
            return True 
        return False

    def __call__(self, chunks: Tuple[str, np.ndarray], random_state):
        tables = []
        i = 0
        while i<len(chunks):
            if self._table_check(chunks[i][0]):
                tables.append([i, chunks.pop(i)[0]])
            else:
                i+=1
        
        summary_dict = {'text':[], 'table':[]}
        
        def summarize(ck_idx, lock, add_cnts, compress='truncate'):
            nonlocal chunks
            try:
                texts = [chunks[i][0] for i in ck_idx]
                if compress:
                    if hasattr(self._llm_model, 'max_len'):
                        len_per_chunk = int((self._llm_model.max_len - self._max_token)/len(texts))
                    else:
                        len_per_chunk = int(self._defalut_max_len/len(texts))
                    if compress=='truncate':
                        cluster_content = "\n".join([truncate_text(t, max(1, len_per_chunk)) for t in texts])
                    else:
                        cluster_content = "\n".join([summary_text(self._llm_model, t, max(1, len_per_chunk)) 
                                                    for t in texts])
                else:
                    cluster_content = "\n".join(texts)
                msg_list = [HumanMessage(content=self._prompt.format(content=cluster_content))]
                time.sleep(0.5)
                cnt = self._llm_model.invoke(msg_list).content
                with lock:
                    add_cnts.append(cnt)
                max_kb.info(f"Cluster SUM（layer:{len(layers)}, chunks:{len(texts)}, len:{len(cluster_content)}）: {cnt}")
            except Exception as e:
                max_kb.info(e)
                traceback.print_stack(e)
                return e
            
        def summarize_table(idx, lock, tids, cnts):
            nonlocal tables
            try:
                tid, table = tables[idx]
                msg_list = [HumanMessage(content=DEFALUT_TABLE_PROMPT.format(content=table))]
                time.sleep(0.5)
                cnt = self._llm_model.invoke(msg_list).content
                with lock:
                    tids.append(tid)
                    cnts.append(cnt)
                    max_kb.info(f'Table SUM（process:{len(tids)}/{len(tables)}, len:{len(table)}）: {cnt}')
            except Exception as e:
                max_kb.info(e)
                traceback.print_stack(e)
                return e
        
        def embed_and_append(cnts):
            nonlocal chunks
            if hasattr(self._embd_model, 'long_pooling'):
                embds = self._embd_model.embed_documents(cnts, return_long='pooling')
            else:
                embds = self._embd_model.embed_documents(cnts)
            for cnt, embd in zip(cnts, embds):
                chunks.append((cnt, embd))
            
        # table 
        if tables:
            self.set_params(max_tokens=self._defalut_max_len)
            max_kb.info(f'Cluster tables: {len(tables)}')
            lock = Lock()
            with ThreadPoolExecutor(max_workers=12) as executor:
                threads = []
                tids, cnts = [], []
                for idx in range(len(tables)):
                    threads.append(executor.submit(summarize_table, idx, lock, tids, cnts))
                wait(threads, return_when=ALL_COMPLETED)
                max_kb.info([t.result() for t in threads])
            if hasattr(self._embd_model, 'long_pooling'):
                embds = self._embd_model.embed_documents(cnts, return_long='pooling')
            else:
                embds = self._embd_model.embed_documents(cnts)
            for tid, cnt, embd in zip(tids, cnts, embds):
                summary_dict['table'].append([tid, embd])
                if chunks:
                    #not excel
                    chunks.append((cnt, embd))
                    
        
        # text 
        self.set_params(max_tokens=self._max_token)
        layers = [(0, len(chunks))]
        start, end = 0, len(chunks)
        if len(chunks) <= 1: return summary_dict
        
        labels = []
        while end - start > 1:
            embeddings = []
            for _, embd in chunks[start: end]:
                if isinstance(embd, list):
                    embd = np.array(embd)
                elif isinstance(embd, torch.Tensor):
                    embd = embd.numpy()
                if len(embd.shape)>1:
                    embd = embd[0]
                embeddings.append(embd)
            if len(embeddings) == 2:
                add_cnts = []
                max_kb.info("Cluster {}th layer: {} -> {}".format(len(layers), end-start, 1))
                summarize([start, start+1], Lock(), add_cnts)
                embed_and_append(add_cnts)
                labels.extend([0,0])
                layers.append((end, len(chunks)))
                start = end
                end = len(chunks)
                continue

            n_neighbors = int((len(embeddings) - 1) ** 0.8)
            reduced_embeddings = umap.UMAP(
                n_neighbors=max(2, n_neighbors), n_components=min(12, len(embeddings)-2), metric="cosine"
            ).fit_transform(embeddings)
            n_clusters = self._get_optimal_clusters(reduced_embeddings, random_state)
            if n_clusters == 1:
                lbls = [[0] for _ in range(len(reduced_embeddings))]
            else:
                gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
                gm.fit(reduced_embeddings)
                probs = gm.predict_proba(reduced_embeddings)
                lbls = [np.where(prob > self._threshold)[0] for prob in probs]
            max_kb.info("Cluster {}th layer: {} -> {}".format(len(layers), end-start, n_clusters))
            lock = Lock()
            single_cnt = 0
            with ThreadPoolExecutor(max_workers=12) as executor:
                threads = []
                add_cnts = []
                for c in range(n_clusters):
                    ck_idx = [i+start for i in range(len(lbls)) if c in lbls[i]]
                    if len(ck_idx)==1 and len(layers)>1:
                        single_cnt += 1
                    else:
                        threads.append(executor.submit(summarize, ck_idx, lock, add_cnts))
                if len(threads):
                    wait(threads, return_when=ALL_COMPLETED)
                    max_kb.info([t.result() for t in threads])
            if add_cnts:
                embed_and_append(add_cnts)
            n_clusters -= single_cnt
            assert len(chunks) - end == n_clusters, "{} vs. {}".format(len(chunks) - end, n_clusters)
            labels.extend(lbls)
            layers.append((end, len(chunks)))
            start = end
            end = len(chunks)
        
        summary_dict['text'] = chunks[layers[0][1]:]
        return summary_dict

