'''
@Description: 
@Author: hh
@LastEditTime: 2024-05-22 17:04:23
@LastEditors: hh
'''
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
import torch
import os
from tqdm import tqdm
from numpy import ndarray
from typing import List, Dict, Tuple, Type, Union
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

def logger_wrapper(name='EmbeddingModel'):
    logging.basicConfig(format='%(asctime)s - [%(levelname)s] -%(name)s->>>    %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(name)
    return logger

logger = logger_wrapper('EmbeddingModel')

DEFAULT_QUERY_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："

class MixEmbedding():
    def __init__(
            self,
            model_name: str='maidalun1020/bce-embedding-base_v1',
            cache_folder: str=None,
            pooler: str='cls',
            use_fp16: bool=False,
            device: str=None,
            long_pooling: str='weight',
            **kwargs
        ):
        try:
            if cache_folder:
                model_name = model_name.split('/')[-1]
                model_name_or_path = os.path.join(cache_folder, model_name)
            else:
                model_name_or_path = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
            self.model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        except Exception as e:
            model_name_or_path = os.path.join('InfiniFlow', model_name.split('/')[-1])
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
            self.model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        logger.info(f"Loading from `{model_name_or_path}`.")

        assert pooler in ['cls', 'mean'], f"`pooler` should be in ['cls', 'mean']. 'cls' is recommended!"
        self.pooler = pooler
        
        num_gpus = torch.cuda.device_count()
        if device is None:
            self.device = "cuda" if num_gpus > 0 else "cpu"
        else:
            self.device = 'cuda:{}'.format(int(device)) if device.isdigit() else device
        
        if self.device == "cpu":
            self.num_gpus = 0
        elif self.device == "mps":
            self.num_gpus = 0
        elif self.device.startswith('cuda:') and num_gpus > 0:
            self.num_gpus = 1
        elif self.device == "cuda":
            self.num_gpus = num_gpus
        else:
            raise ValueError("Please input valid device: 'cpu', 'mps', 'cuda', 'cuda:0', '0' !")

        if use_fp16:
            self.model.half()

        self.model.eval()
        self.model = self.model.to(self.device)
        self.long_pooling = long_pooling
        self.model_name = model_name

        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        # special input 
        self.pad_input = self.tokenizer(self.tokenizer.pad_token, add_special_tokens=False)
        self.pad_input['attention_mask'] = [0]
        self.cls_input = self.tokenizer(self.tokenizer.cls_token, add_special_tokens=False)
        self.sep_input = self.tokenizer(self.tokenizer.sep_token, add_special_tokens=False)
        
        logger.info(f"Execute device: {self.device};\t gpu num: {self.num_gpus};\t use fp16: {use_fp16};\t embedding pooling type: {self.pooler};\t trust remote code: {kwargs.get('trust_remote_code', False)}")

    def embedding(self, inputs_on_device):
        if isinstance(inputs_on_device, list):
            return torch.cat([self.embedding(input) for input in inputs_on_device])
        else:
            outputs = self.model(**inputs_on_device, return_dict=True)
            if self.pooler == "cls":
                embeddings = outputs.last_hidden_state[:, 0]
            elif self.pooler == "mean":
                attention_mask = inputs_on_device['attention_mask']
                last_hidden = outputs.last_hidden_state
                embeddings = (last_hidden * attention_mask.unsqueeze(-1).float()).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            else:
                raise NotImplementedError
            return embeddings
        
    def moving_average_2d(self, data, alpha, initial_value=None, keepdim=True):
        time_steps, features = data.shape
        if initial_value is None:
            initial_value = data[0]  
        elif initial_value.numel() != features:
            raise ValueError("initial_value 的尺寸必须与data的特征数匹配")
        ema_value = initial_value
        for i in range(time_steps):
            ema_value = (1 - alpha) * data[i] + alpha * ema_value
        if keepdim:
            ema_value = ema_value.unsqueeze(0)
        return ema_value

    def weight_average_2d(self, data, keepdim=True):
        time_steps, _ = data.shape
        weight = [i+1 for i in range(time_steps)][::-1]
        weight = torch.tensor(weight).to(data.device)/sum(weight)
        weight.unsqueeze_(-1)
        data = data*weight
        data = data.sum(dim=0, keepdim=keepdim)
        return data 
    
    def encode_long_pooling(self, sentence_batch, max_length, 
                            query_instr_input, batch_size, return_long, normalize_to_unit, return_type):
        inputs = {}
        batch_idx = []
        idx = 0
        if query_instr_input is None:
            query_length = 0
        else:
            query_length = len(query_instr_input['input_ids'])
        
        input = self.tokenizer(sentence_batch, max_length=None, add_special_tokens=False)
        for k,v in input.items():
            if k not in inputs:
                inputs[k] = []
            for vi in v:
                if len(vi)+query_length+2>max_length:
                    b_idx = []
                    for i in range(0, len(vi), max_length-2-query_length):
                        vii = vi[i:i+max_length-2-query_length]
                        if len(vii)<10:
                            continue 
                        if query_instr_input is None:
                            vii = self.cls_input[k]+vii+self.sep_input[k]
                        else:
                            vii = self.cls_input[k]+query_instr_input[k]+vii+self.sep_input[k]
                        if len(vii)<max_length:
                            vii = vii+self.pad_input[k]*(max_length-len(vii))
                        inputs[k].append(vii)
                        if k=='input_ids':
                            b_idx.append(idx)
                            idx += 1
                    if k=='input_ids':
                        batch_idx.append(b_idx)
                else:
                    if query_instr_input is None:
                        vi = self.cls_input[k]+vi+self.sep_input[k]
                    else:
                        vi = self.cls_input[k]+query_instr_input[k]+vi+self.sep_input[k]
                    if len(vi)<max_length:
                        vi = vi+self.pad_input[k]*(max_length-len(vi))
                    inputs[k].append(vi)
                    if k=='input_ids':
                        batch_idx.append([idx])
                        idx += 1
        inputs_on_device = [{k: torch.tensor(v[i:i+batch_size], device=self.device) for k,v in inputs.items()} 
                            for i in range(0, idx, batch_size)]
        long_embeddings = self.embedding(inputs_on_device)
        if normalize_to_unit:
            long_embeddings = F.normalize(long_embeddings, dim=-1)
        
        embeddings = []
        for bids in batch_idx:
            if len(bids)>1:
                if self.long_pooling=='moving':
                    em = self.moving_average_2d(data=long_embeddings[bids], alpha=0.95, keepdim=True)
                elif self.long_pooling=='weight':
                    em = self.weight_average_2d(long_embeddings[bids], keepdim=True)
                else:
                    em = long_embeddings[bids].sum(dim=0, keepdim=True)/len(bids)
                if return_long.lower()=='mix':
                    em = torch.cat([em, long_embeddings[bids]], dim=0)
            else:
                em = long_embeddings[bids]
            if return_long.lower()=='mix':
                if return_type=='list':
                    em = em.tolist()
                elif return_type=='numpy':
                    em = em.numpy()
                else:
                    em = em.cpu()
            else:
                em = em.cpu()
            embeddings.append(em)
        return embeddings
    
    def encode(
            self,
            sentences: Union[str, List[str]],
            batch_size: int=256,
            max_length: int=512,
            normalize_to_unit: bool=True,
            return_type: str='list',
            enable_tqdm: bool=True,
            query_instruction: str="",
            return_long: str=None,
            **kwargs
        ):
        if self.num_gpus > 1:
            batch_size = batch_size * self.num_gpus
                
        if isinstance(sentences, str):
            sentences = [sentences]
        
        if return_long:
            assert return_long.lower() in ['mix', 'pooling'], f"`return_long` should be in ['mix', 'pooling']."
        
        query_instruction = query_instruction if query_instruction else DEFAULT_QUERY_INSTRUCTION_ZH
        query_instr_input = self.tokenizer(query_instruction, add_special_tokens=False) if query_instruction else None
        
        with torch.no_grad():
            embeddings_collection = []
            for sentence_id in tqdm(range(0, len(sentences), batch_size), desc='Extract embeddings', disable=not enable_tqdm):
                sentence_batch = sentences[sentence_id:sentence_id+batch_size]
                if return_long:
                    embeddings = self.encode_long_pooling(sentence_batch, max_length, query_instr_input, 
                                                          batch_size, return_long, normalize_to_unit, return_type)
                    embeddings_collection.extend(embeddings)
                else:
                    inputs = self.tokenizer(sentence_batch, padding=True, truncation=True, max_length=max_length, 
                                            return_tensors="pt")
                    inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    embeddings = self.embedding(inputs_on_device)
                    if normalize_to_unit:
                        embeddings = F.normalize(embeddings, dim=-1)
                    embeddings_collection.append(embeddings.cpu())
            if return_long and return_long.lower()=='mix':
                return embeddings_collection
            else:
                embeddings = torch.cat(embeddings_collection)
                if return_type=='list':
                    embeddings = embeddings.tolist()
                elif return_type=='numpy':
                    embeddings = embeddings.numpy()
                return embeddings
    
    def embed_documents(self, texts: list, batch_size=32, return_long=None):
        res = []
        for i in range(0, len(texts), batch_size):
            embds = self.encode(texts[i:i + batch_size], return_type='list', 
                                batch_size=batch_size, return_long=return_long)
            res.extend(embds)
        return res

    def embed_query(self, text, return_long=None):
        embds = self.encode([text], return_type='list', return_long=return_long)
        return embds[0]
    
class XinferenceEmbed(HuggingFaceEmbeddings):
    def __init__(self, model_name="", base_url=""):
        self.client = OpenAI(api_key="xxx", base_url=base_url)
        self.model_name = model_name

    def embed_documents(self, texts: list, batch_size=32):
        res = self.client.embeddings.create(input=texts,
                                            model=self.model_name)
        return [d.embedding.tolist() for d in res.data]

    def embed_query(self, text):
        res = self.client.embeddings.create(input=[text],
                                            model=self.model_name)
        return res.data[0].embedding.tolist()
