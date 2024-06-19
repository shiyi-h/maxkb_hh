'''
@Description: 
@Author: hh
@LastEditTime: 2024-05-22 17:04:23
@LastEditors: hh
'''
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import os
from tqdm import tqdm
from numpy import ndarray
from typing import List, Dict, Tuple, Type, Union
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, XLMRobertaModel

def logger_wrapper(name='EmbeddingModel'):
    logging.basicConfig(format='%(asctime)s - [%(levelname)s] -%(name)s->>>    %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(name)
    return logger

logger = logger_wrapper('EmbeddingModel')

DEFAULT_QUERY_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："

class YoudaoEmbedding():
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
    
    def encode(
            self,
            sentences: Union[str, List[str]],
            batch_size: int=256,
            max_length: int=512,
            normalize_to_unit: bool=True,
            return_list: bool=False,
            return_numpy: bool=True,
            enable_tqdm: bool=True,
            query_instruction: str="",
            long_pooling: bool=False,
            **kwargs
        ):
        if self.num_gpus > 1:
            batch_size = batch_size * self.num_gpus
        
        if isinstance(sentences, str):
            sentences = [sentences]
        
        with torch.no_grad():
            for sentence_id in tqdm(range(0, len(sentences), batch_size), desc='Extract embeddings', disable=not enable_tqdm):
                if isinstance(query_instruction, str) and len(query_instruction) > 0:
                    sentence_batch = [query_instruction+sent for sent in sentences[sentence_id:sentence_id+batch_size]] 
                else:
                    sentence_batch = sentences[sentence_id:sentence_id+batch_size]
                if long_pooling:
                    inputs = {}
                    batch_idx = []
                    idx = 0
                    for sentence in sentence_batch:
                        input = self.tokenizer(sentence, padding=True, truncation=True, max_length=None, 
                                            return_tensors="pt")
                        if input['input_ids'].size(1)>max_length:
                            b_idx = []
                            for k,v in input.items():
                                v = v[:, 1:]
                                for i in range(0, v.size(1), max_length-1):
                                    vi = v[:, i:i+max_length-1]
                                    if vi.size(1)>=batch_size:
                                        vi = F.pad(vi, (1, 0), 'constant', 1 if 'mask' in k else 0)
                                        if vi.size(1)<max_length:
                                            vi = F.pad(vi, (0,max_length-vi.size(1)), 'constant', 0 if 'mask' in k else 1)
                                        if k not in inputs:
                                            inputs[k] = []
                                        inputs[k].append(vi)
                                        if k=='input_ids':
                                            b_idx.append(idx)
                                            idx+=1
                                            
                            batch_idx.append(b_idx)
                        else:
                            for k,v in input.items():
                                if k not in inputs:
                                    inputs[k] = []
                                if v.size(1)<max_length:
                                    v = F.pad(v, (0, max_length-v.size(1)), 'constant', 0 if 'mask' in k else 1)
                                inputs[k].append(v)
                            batch_idx.append([idx])
                            idx+=1
                    inputs_on_device = [{k: torch.cat(v[i:i+batch_size], dim=0).to('cpu') for k,v in inputs.items()} 
                                        for i in range(0, idx, batch_size)]
                    long_embeddings = self.embedding(inputs_on_device)
                    embeddings = []
                    for bids in batch_idx:
                        batch_emb = []
                        if len(bids)>1:
                            if self.long_pooling=='moving':
                                em = self.moving_average_2d(data=long_embeddings[bids], alpha=0.95, keepdim=False)
                            elif self.long_pooling=='weight':
                                em = self.weight_average_2d(long_embeddings[bids], keepdim=False)
                            else:
                                em = long_embeddings[bids].sum(dim=0, keepdim=False)/len(bids)
                            if return_list:
                                batch_emb.append(em.tolist())
                                batch_emb.extend(long_embeddings[bids].tolist())
                            elif return_numpy:
                                batch_emb.append(em.numpy())
                                batch_emb.extend(long_embeddings[bids].numpy())
                            else:
                                batch_emb.append(em)
                                batch_emb.extend(long_embeddings[bids])
                        else:
                            if return_list:
                                batch_emb.append(long_embeddings[bids][0].tolist())
                            elif return_numpy:
                                batch_emb.append(long_embeddings[bids][0].numpy())
                            else:
                                batch_emb.append(long_embeddings[bids][0])
                        embeddings.append(batch_emb)    
                    return embeddings                
                else:
                    embeddings_collection = []
                    inputs = self.tokenizer(sentence_batch, padding=True, truncation=True, max_length=max_length, 
                                            return_tensors="pt")
                    inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
                    embeddings = self.embedding(inputs_on_device)

                    if normalize_to_unit:
                        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                    embeddings_collection.append(embeddings.cpu())
            
                    embeddings = torch.cat(embeddings_collection, dim=0)
        
                    if return_list and not isinstance(embeddings, list):
                        embeddings = embeddings.tolist()
                    elif return_numpy and not isinstance(embeddings, ndarray):
                        embeddings = embeddings.numpy()
                    
                    return embeddings
    
    def embed_documents(self, texts: list, batch_size=32, long_pooling=False):
        res = []
        for i in range(0, len(texts), batch_size):
            embds = self.encode(texts[i:i + batch_size], return_list=True, 
                                batch_size=batch_size, long_pooling=long_pooling)
            res.extend(embds)
        return res

    def embed_query(self, text, long_pooling=False):
        embds = self.encode([text], return_list=True, long_pooling=long_pooling)
        return embds[0]
