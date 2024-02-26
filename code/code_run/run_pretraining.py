from __future__ import absolute_import, division, print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"



import json
import logging
import numpy as np
import torch
from tqdm import tqdm, trange
from torch.distributions import Categorical
import torch.multiprocessing as mp
import sys

import shutil
import copy_util as util
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler
from random import random, shuffle, choice, sample, randint
from random import seed as rd_seed

from roberta_model_file.modeling_roberta import RobertaForMaskedLM, RobertaModel
from roberta_model_file.tokenization_roberta import RobertaTokenizer
from optimizer_pkg.myOptimizer import BertAdam
from transformers.optimization import get_linear_schedule_with_warmup
import argparse
import pickle
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    filename='./run.log')
logger = logging.getLogger(__name__)
logger.info('logger initaled ok')



CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
GRADIENT_CLIP = 25

data_path = './PLM_RL/'

with open(data_path+"roberta_pretrain_data/tokenize_result_triples_with_s_pos.json") as f:
    line = f.readline()
    triple_token_result = json.loads(line)
data_file = open(data_path+"roberta_pretrain_data/total_pretrain_data.txt", 'r', encoding='utf-8')
examples_sentences = data_file.readlines()

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rd_seed(seed)

class OurENRIEDataset(Dataset):
    def __init__(self, args, data_path, max_seq_length, masked_lm_prob,
                 max_predictions_per_seq, tokenizer,
                 data_type='train', min_len=100):
        self.args = args
        self.data_path = data_path
        self.max_seq_length = max_seq_length  
        self.masked_lm_prob = masked_lm_prob  
        self.max_predictions_per_seq = max_predictions_per_seq  
        self.tokenizer = tokenizer
        self.min_len = min_len  
        self.max_num_tokens = max_seq_length - 2  
        self.examples = []  
        self.vocab = tokenizer.get_vocab()
        self.data_type = data_type

        self.__read_data__()
        self.data_sources = os.path.join(self.data_path, '{}_pretrain_data_with_importance_combine_600w.txt'.format('total'))
       
        
    def __getitem__(self, index):
        offset = self.examples[index] 
        
        example = eval(examples_sentences[index].strip())
        tmp_qid_list = []
        tmp_pos_list = []
        for i, qid in enumerate(example['entity_qid']):
            if qid in triple_token_result.keys():
                tmp_qid_list.append(qid)
                tmp_pos_list.append(example['entity_pos'][i])
        if len(tmp_qid_list) > self.args.max_entity_num_threshold:
            tmp_qid_list = tmp_qid_list[:self.args.max_entity_num_threshold]
            tmp_pos_list = tmp_pos_list[:self.args.max_entity_num_threshold]
        example['entity_qid'] = tmp_qid_list
        example['entity_pos'] = tmp_pos_list

        masked_example = self.__get_example__(example)
        feature = self.__get_feature__(masked_example)
        tensor_tuple = self.__feature2tensor__(feature)
        return tensor_tuple

    def __get_example__(self, example):

        token_ids, masked_lm_positions, masked_label_ids, entity_qid, entity_pos, entity_masked_lm_positions_after_duplicate_removal, masked_entity_label_ids, \
        token_masked_lm_positions_after_duplicate_removal, masked_token_label_ids = create_wwm_lm_predictions(self.args,
                                                                                                              example,
                                                                                                              self.masked_lm_prob,
                                                                                                              self.max_predictions_per_seq,
                                                                                                              self.vocab,
                                                                                                              self.tokenizer,
                                                                                                              self.data_type)
        after_filter_entity_qid = []
        batch_qid_triple_list = []
        batch_qid_triple_inputId_list = []
        batch_qid_triple_attMask_list = []
        batch_qid_triple_element_pos_list = []
        batch_qid_entity_importance_list = []
        batch_qid_triples_importance_indices_list = []
        batch_qid_triples_importance_values_list = []
        for qid in entity_qid:
            temp_qid_triple_input_ids = triple_token_result[qid]['input_ids']
            temp_qid_triple_attention_mask = triple_token_result[qid]['attention_mask']
            temp_qid_triple_element_pos = triple_token_result[qid]['element_separater_index']
            t_i = [temp_qid_triple_input_ids[i] for i in example['triples_importance'][qid]['indices']]
            t_a = [temp_qid_triple_attention_mask[i] for i in example['triples_importance'][qid]['indices']]
            t_e = [temp_qid_triple_element_pos[i][:3] for i in example['triples_importance'][qid]['indices']]


            batch_qid_triple_inputId_list.append(t_i)
            batch_qid_triple_attMask_list.append(t_a)
            batch_qid_triple_element_pos_list.append(t_e)
            batch_qid_entity_importance_list.append(example['entity_importance'][qid])
            batch_qid_triples_importance_indices_list.append(example['triples_importance'][qid]['indices'])
            batch_qid_triples_importance_values_list.append(example['triples_importance'][qid]['values'])



        segment_ids = [0] * len(token_ids)  
        example = {
            "token_ids": token_ids,  
            "segment_ids": segment_ids,
            "masked_lm_positions": masked_lm_positions,            
            "masked_label_ids": masked_label_ids,           
            'entity_qid': entity_qid,  
            'entity_pos': entity_pos,  
            'entity_masked_lm_positions_after_duplicate_removal': entity_masked_lm_positions_after_duplicate_removal,
            'masked_entity_label_ids': masked_entity_label_ids,  
            'token_masked_lm_positions_after_duplicate_removal': token_masked_lm_positions_after_duplicate_removal,
            'masked_token_label_ids': masked_token_label_ids,
            'batch_qid_triple_inputId_list': batch_qid_triple_inputId_list,
            'batch_qid_triple_attMask_list': batch_qid_triple_attMask_list,
            'batch_qid_triple_element_pos_list': batch_qid_triple_element_pos_list,
            'batch_qid_entity_importance_list': batch_qid_entity_importance_list,
            'batch_qid_triples_importance_indices_list': batch_qid_triples_importance_indices_list,
            'batch_qid_triples_importance_values_list': batch_qid_triples_importance_values_list,
        }
        return example

    def __get_feature__(self, example):
        max_seq_length = self.max_seq_length
        input_ids = example["token_ids"]
        segment_ids = example["segment_ids"]
        masked_lm_positions = example["masked_lm_positions"]
        masked_label_ids = example["masked_label_ids"]
        entity_masked_lm_positions_after_duplicate_removal = example["entity_masked_lm_positions_after_duplicate_removal"]
        masked_entity_label_ids = example["masked_entity_label_ids"]
        token_masked_lm_positions_after_duplicate_removal = example["token_masked_lm_positions_after_duplicate_removal"]
        masked_token_label_ids = example["masked_token_label_ids"]
        batch_qid_triple_inputId_list = example["batch_qid_triple_inputId_list"]
        batch_qid_triple_attMask_list = example["batch_qid_triple_attMask_list"]
        batch_qid_triple_element_pos_list = example["batch_qid_triple_element_pos_list"]
        batch_qid_entity_importance_list = example["batch_qid_entity_importance_list"]
        batch_qid_triples_importance_indices_list = example["batch_qid_triples_importance_indices_list"]
        batch_qid_triples_importance_values_list = example["batch_qid_triples_importance_values_list"]
        try:
            assert len(input_ids) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
        except AssertionError:
            a = input_ids
            b = segment_ids
            c = max_seq_length
        input_array = np.full(max_seq_length, dtype=int, fill_value=self.tokenizer.convert_tokens_to_ids(['<pad>'])[0])
        input_array[:len(input_ids)] = input_ids

        mask_array = np.zeros(max_seq_length, dtype=int)
        mask_array[:len(input_ids)] = 1

        segment_array = np.zeros(max_seq_length, dtype=int)
        segment_array[:len(segment_ids)] = segment_ids

        lm_label_array_total = np.full(max_seq_length, dtype=int, fill_value=-100)
        lm_label_array_total[masked_lm_positions] = masked_label_ids

        lm_label_array_entity = np.full(max_seq_length, dtype=int, fill_value=-100)
        lm_label_array_entity[entity_masked_lm_positions_after_duplicate_removal] = masked_entity_label_ids

        lm_label_array_token = np.full(max_seq_length, dtype=int, fill_value=-100)
        lm_label_array_token[token_masked_lm_positions_after_duplicate_removal] = masked_token_label_ids

        batch_qid_triple_inputId_np = np.zeros((self.args.max_entity_num_threshold, self.args.max_triple_num_threshold, 15), dtype=int)
        batch_qid_triple_attMask_np = np.zeros((self.args.max_entity_num_threshold, self.args.max_triple_num_threshold, 15), dtype=int)
        batch_qid_triple_element_pos_np = np.zeros((self.args.max_entity_num_threshold, self.args.max_triple_num_threshold, 3), dtype=int)
        batch_qid_entity_importance_np = np.full((self.args.max_entity_num_threshold), dtype=float, fill_value=-100.0)
        batch_qid_triples_importance_indices_np = np.full((self.args.max_entity_num_threshold, self.args.max_triple_num_threshold), dtype=int, fill_value=-100)
        batch_qid_triples_importance_values_np = np.full((self.args.max_entity_num_threshold, self.args.max_triple_num_threshold), dtype=float, fill_value=-100.0)

        entity_pos_sequence = np.zeros((self.args.max_entity_num_threshold, 2), dtype=int)
        math_num_cut_q_qid = np.zeros((self.args.max_entity_num_threshold, 1), dtype=int)

        entity_qid = example['entity_qid']
        entity_pos = example['entity_pos']
        if len(entity_qid) != 0:
            for index, qid in enumerate(entity_qid):
                entity_start_in_context, entity_end_in_context = entity_pos[index]  
                entity_pos_sequence[index] = entity_pos[index]

                if len(batch_qid_triple_inputId_list[index])>self.args.max_triple_num_threshold:
                    batch_qid_triple_inputId_list[index] = batch_qid_triple_inputId_list[index][0:self.args.max_triple_num_threshold]
                if len(batch_qid_triple_attMask_list[index])>self.args.max_triple_num_threshold:
                    batch_qid_triple_attMask_list[index] = batch_qid_triple_attMask_list[index][0:self.args.max_triple_num_threshold]
                if len(batch_qid_triple_element_pos_list[index])>self.args.max_triple_num_threshold:
                    batch_qid_triple_element_pos_list[index] = batch_qid_triple_element_pos_list[index][0:self.args.max_triple_num_threshold]
                if len(batch_qid_entity_importance_list)>self.args.max_triple_num_threshold:
                    batch_qid_entity_importance_list = batch_qid_entity_importance_list[0:self.args.max_triple_num_threshold]
                if len(batch_qid_triples_importance_indices_list[index])>self.args.max_triple_num_threshold:
                    batch_qid_triples_importance_indices_list[index] = batch_qid_triples_importance_indices_list[index][0:self.args.max_triple_num_threshold]
                if len(batch_qid_triples_importance_values_list[index])>self.args.max_triple_num_threshold:
                    batch_qid_triples_importance_values_list[index] = batch_qid_triples_importance_values_list[index][0:self.args.max_triple_num_threshold]

                batch_qid_triple_inputId_np[index][:len(batch_qid_triple_inputId_list[index])] = \
                batch_qid_triple_inputId_list[index]
                batch_qid_triple_attMask_np[index][:len(batch_qid_triple_attMask_list[index])] = \
                batch_qid_triple_attMask_list[index]
                batch_qid_triple_element_pos_np[index][:len(batch_qid_triple_element_pos_list[index])] = \
                    batch_qid_triple_element_pos_list[index]
                batch_qid_entity_importance_np[:len(batch_qid_entity_importance_list)] = \
                    batch_qid_entity_importance_list
                batch_qid_triples_importance_indices_np[index][:len(batch_qid_triples_importance_indices_list[index])] = \
                    batch_qid_triples_importance_indices_list[index]
                batch_qid_triples_importance_values_np[index][:len(batch_qid_triples_importance_values_list[index])] = \
                    batch_qid_triples_importance_values_list[index]

                math_num_cut_q_qid[index] = qid[1:]


        feature = InputFeatures(input_ids=input_array,
                                input_mask=mask_array,
                                segment_ids=segment_array,
                                label_id=lm_label_array_total,
                                lm_label_array_entity=lm_label_array_entity,
                                lm_label_array_token=lm_label_array_token,
                                batch_qid_triple_inputId_np=batch_qid_triple_inputId_np,
                                batch_qid_triple_attMask_np=batch_qid_triple_attMask_np,
                                batch_qid_triple_element_pos_np=batch_qid_triple_element_pos_np,
                                batch_qid_entity_importance_np=batch_qid_entity_importance_np,
                                batch_qid_triples_importance_indices_np=batch_qid_triples_importance_indices_np,
                                batch_qid_triples_importance_values_np=batch_qid_triples_importance_values_np,
                                entity_pos_sequence=entity_pos_sequence,
                                math_num_cut_q_qid=math_num_cut_q_qid)
        return feature

    def __feature2tensor__(self, feature):
        f = feature
        all_input_ids = torch.tensor(f.input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(f.input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(f.segment_ids, dtype=torch.long)
        all_label_ids = torch.tensor(f.label_id, dtype=torch.long)
        lm_label_array_entity = torch.tensor(f.lm_label_array_entity, dtype=torch.long).squeeze(0)
        lm_label_array_token = torch.tensor(f.lm_label_array_token, dtype=torch.long).squeeze(0)

        batch_qid_triple_inputId_tensor = torch.tensor(f.batch_qid_triple_inputId_np, dtype=torch.long)
        batch_qid_triple_attMask_tensor = torch.tensor(f.batch_qid_triple_attMask_np, dtype=torch.long)
        batch_qid_triple_element_pos_tensor = torch.tensor(f.batch_qid_triple_element_pos_np, dtype=torch.long)
        batch_qid_entity_importance_tensor = torch.tensor(f.batch_qid_entity_importance_np, dtype=torch.float)
        batch_qid_triples_importance_indices_tensor = torch.tensor(f.batch_qid_triples_importance_indices_np, dtype=torch.long)
        batch_qid_triples_importance_values_tensor = torch.tensor(f.batch_qid_triples_importance_values_np, dtype=torch.float)

        entity_pos_sequence = torch.tensor(f.entity_pos_sequence, dtype=torch.long)
        math_num_cut_q_qid = torch.tensor(f.math_num_cut_q_qid, dtype=torch.long)

        return all_input_ids, all_input_mask, all_segment_ids, all_label_ids,\
               lm_label_array_entity, lm_label_array_token, \
               batch_qid_triple_inputId_tensor,\
               batch_qid_triple_attMask_tensor,\
               batch_qid_triple_element_pos_tensor,\
               batch_qid_entity_importance_tensor,\
               batch_qid_triples_importance_indices_tensor,\
               batch_qid_triples_importance_values_tensor,\
               entity_pos_sequence, math_num_cut_q_qid

    def __len__(self):
        return len(self.examples)

    def __read_data__(self):
        if self.data_type == 'train':
            with open(data_path+'roberta_pretrain_data/train_sample_data_99%.txt',
                      'r') as file:
                self.examples = tuple([int(item.strip()) for item in file.readlines()])
                logger.info('training samples loaded!')
        else:
            with open(data_path+'roberta_pretrain_data/dev_sample_data_1%.txt', 'r') as file:
                self.examples = tuple([int(item.strip()) for item in file.readlines()])
                logger.info('dev samples loaded!')


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, lm_label_array_entity, lm_label_array_token,
                 batch_qid_triple_inputId_np,
                 batch_qid_triple_attMask_np,
                 batch_qid_triple_element_pos_np,
                 batch_qid_entity_importance_np,
                 batch_qid_triples_importance_indices_np,
                 batch_qid_triples_importance_values_np,
                 entity_pos_sequence,
                 math_num_cut_q_qid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.lm_label_array_entity = lm_label_array_entity,
        self.batch_qid_triple_inputId_np = batch_qid_triple_inputId_np,
        self.batch_qid_triple_attMask_np = batch_qid_triple_attMask_np,
        self.batch_qid_triple_element_pos_np = batch_qid_triple_element_pos_np,
        self.batch_qid_entity_importance_np = batch_qid_entity_importance_np,
        self.batch_qid_triples_importance_indices_np = batch_qid_triples_importance_indices_np,
        self.batch_qid_triples_importance_values_np = batch_qid_triples_importance_values_np,
        self.lm_label_array_token = lm_label_array_token,
        self.entity_pos_sequence = entity_pos_sequence
        self.math_num_cut_q_qid = math_num_cut_q_qid

def isSkipToken(token):
    return token == "[CLS]" or token == "[SEP]" or (not token.isalnum() and len(token) == 1)

def get_reduce_dict(d, n):
    tmp = list(d.items())
    shuffle(tmp)
    tmp = tmp[:n]
    return dict(tmp)

def create_wwm_lm_predictions(args, example, masked_lm_prob, max_predictions_per_seq, vocab, tokenizer, data_type):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    token_ids = example['token_ids'][:126]
    token_ids = [0] + token_ids + [2]

    filtered_out_range_entities_qid = []
    filtered_out_range_entities_pos = []
    for i, entity_pos in enumerate(example['entity_pos']):
        if entity_pos[1] <= 126:
            filtered_out_range_entities_qid.append(example['entity_qid'][i])
            filtered_out_range_entities_pos.append(example['entity_pos'][i])


    entity_qid = filtered_out_range_entities_qid
    entity_pos = filtered_out_range_entities_pos
    temp_tmp = list(zip(entity_pos, entity_qid))  
    tmp = []
    if len(temp_tmp) == 0:
        pass
    elif len(temp_tmp) == 1:
        tmp.append(temp_tmp[0])
    else:
        tmp.append(temp_tmp[0])
        count_pos = temp_tmp[0][0][1]  
        for i in range(1, len(temp_tmp), 1):
            item = temp_tmp[i]
            entityPos, entityQname = item[0], item[1]
            start_ent, end_ent = entityPos[0], entityPos[1]
            if start_ent - count_pos >= 5:
                tmp.append(item)
                count_pos = end_ent
            else:
                continue
    filter_long_distant_pos = []
    for i in range(len(tmp)):
        filter_long_distant_pos.append(tmp[i][0])


    if len(tmp) > 0:

        total_num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(token_ids) * masked_lm_prob))))  

        total_entity_num_to_mask = int(total_num_to_mask * 0.5)
        total_token_num_to_mask = total_num_to_mask - total_entity_num_to_mask

        shuffle(filter_long_distant_pos)
        entity_masked_lm_positions = []
        for select in range(len(filter_long_distant_pos)):
            if len(entity_masked_lm_positions) >= total_entity_num_to_mask:
                break
            else:
                entity_masked_lm_positions.extend(j for j in range(entity_pos[select][0], entity_pos[select][1]))

        if len(entity_masked_lm_positions) < total_entity_num_to_mask: 
            total_token_num_to_mask = total_token_num_to_mask + (
                        total_entity_num_to_mask - len(entity_masked_lm_positions))
        else:
            total_token_num_to_mask = total_num_to_mask - len(entity_masked_lm_positions)

        token_masked_lm_positions = []
        token_mask_list = list(i for i in range(1, len(token_ids) - 1))
        for i in token_mask_list:
            if i in entity_masked_lm_positions:
                token_mask_list.remove(i)

        shuffle(token_mask_list)
        for t in token_mask_list:
            if (total_token_num_to_mask <= 0): break
            if (t not in token_masked_lm_positions):
                token_masked_lm_positions.append(t)
                total_token_num_to_mask -= 1

        both_token_and_entity_masked_lm_positions = token_masked_lm_positions + entity_masked_lm_positions

        masked_total_label_ids = []
        masked_entity_label_ids = []
        masked_token_label_ids = []
        entity_masked_lm_positions_after_duplicate_removal = []
        token_masked_lm_positions_after_duplicate_removal = []
        for pos in both_token_and_entity_masked_lm_positions:
            masked_total_label_ids.append(token_ids[pos])
            if pos in entity_masked_lm_positions:
                masked_entity_label_ids.append(token_ids[pos])
                entity_masked_lm_positions_after_duplicate_removal.append(pos)
            else:
                masked_token_label_ids.append(token_ids[pos])
                token_masked_lm_positions_after_duplicate_removal.append(pos)

            masked_token = 50264  
            if (random() < .2):  
                masked_token = token_ids[pos]
                if (random() < .5):
                    masked_token = randint(3, len(vocab) - 100)
            token_ids[pos] = masked_token

        return token_ids, both_token_and_entity_masked_lm_positions, masked_total_label_ids, entity_qid, entity_pos, \
               entity_masked_lm_positions_after_duplicate_removal, masked_entity_label_ids, \
               token_masked_lm_positions_after_duplicate_removal, masked_token_label_ids

    else:
        total_num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(token_ids) * masked_lm_prob))))  
        token_masked_lm_positions = []
        token_mask_list = list(i for i in range(1, len(token_ids) - 1))

        shuffle(token_mask_list)
        for t in token_mask_list:
            if (total_num_to_mask <= 0): break
            if (t not in token_masked_lm_positions):
                token_masked_lm_positions.append(t)
                total_num_to_mask -= 1

        masked_label_ids = []
        for pos in token_masked_lm_positions:
            masked_label_ids.append(token_ids[pos])
            masked_token = 50264
            if (random() < .2):
                masked_token = token_ids[pos]
                if (random() < .5):
                    masked_token = randint(3, len(vocab) - 100)
            token_ids[pos] = masked_token

        entity_masked_lm_positions_after_duplicate_removal = token_masked_lm_positions
        masked_entity_label_ids = masked_label_ids
        token_masked_lm_positions_after_duplicate_removal = token_masked_lm_positions
        masked_token_label_ids = masked_label_ids
        return token_ids, token_masked_lm_positions, masked_label_ids, entity_qid, entity_pos, \
               entity_masked_lm_positions_after_duplicate_removal, masked_entity_label_ids, \
               token_masked_lm_positions_after_duplicate_removal, masked_token_label_ids


def reduce_tensor(tensor, ws=2):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= ws
    return rt


def evaluate(args, model, eval_dataloader, device, epoch, train_loss, best_loss):
    torch.cuda.empty_cache()
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    if (args.local_rank >= 0):
        torch.distributed.barrier()
    if (args.local_rank != -1):
        eval_dataloader.sampler.set_epoch(args.seed)
    for batch in tqdm(eval_dataloader, desc='Evaluation') if args.local_rank <= 0 else eval_dataloader:
        batch0 = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids, lm_label_array_entity, lm_label_array_token, \
            batch_qid_triple_inputId_tensor, batch_qid_triple_attMask_tensor, entity_pos_sequence, math_num_cut_q_qid = batch0
            outputs = model(
                input_ids=all_input_ids,
                attention_mask=all_input_mask,
                token_type_ids=all_segment_ids,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                labels=all_label_ids,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=False
            )
            loss = outputs[0]
          
            if (args.local_rank >= 0):
                loss = reduce_tensor(loss, dist.get_world_size())
        eval_loss += loss.mean().item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    if best_loss > eval_loss:
        best_loss = eval_loss
        # save best model
        if args.fp16:
            if (args.local_rank <= 0):
                logger.info('**************************************************************************')
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.output_dir, "best_pytorch_model.bin")
                logger.info('Saving best Model into {}'.format(output_model_file))
                torch.save(model_to_save.state_dict(), output_model_file)
                logger.info('Saving best Model Done!')
                logger.info('**************************************************************************')
        else:
       
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, "best_pytorch_model.bin")
            if (args.local_rank <= 0):
                logger.info('**************************************************************************')
                logger.info('Saving best Model into {}'.format(output_model_file))
                torch.save(model_to_save.state_dict(), output_model_file)
                logger.info('Saving best Model Done!')
                logger.info('**************************************************************************')

    if (args.local_rank <= 0):
        logger.info(
            "============================ -epoch %d -train_loss %.4f -eval_loss %.4f -best_loss %.4f\n" % (epoch,
                                                                                                           train_loss,
                                                                                                           eval_loss,
                                                                                                           best_loss))
    torch.cuda.empty_cache()
    return best_loss


def retrieval_triple(math_num_cut_q_qid, result_dict):
    
    triple_collection_batch = []
    for i in math_num_cut_q_qid:
        tmp = []
        for j in i:
            key = 'Q{}'.format(j)
            
            if key not in result_dict.keys():
                triple_list_single_qid = ['0'] * 12
            else:
                triple_list_single_qid = result_dict[key]
                n = []
                for no, m in enumerate(triple_list_single_qid):
                    m = ' '.join(m)
                    n.append(m)
                triple_list_single_qid = n
                len_triple_list_single_qid = len(triple_list_single_qid)
                if len_triple_list_single_qid < 12:
                    triple_list_single_qid[(len_triple_list_single_qid + 1):12] = ['0'] * (
                                12 - len_triple_list_single_qid)
                else:
                    triple_list_single_qid = triple_list_single_qid[:12]
            
            tmp.append(triple_list_single_qid)
        triple_collection_batch.append(tmp)
    return triple_collection_batch


def get_batch_triple_hidden(batch_qid_triple_inputId_tensor, batch_qid_triple_attMask_tensor, model, batchsize, max_entity_num_threshold, max_triple_num_threshold):
    
    batch_qid_triple_inputId_tensor1 = batch_qid_triple_inputId_tensor.reshape(batchsize * max_entity_num_threshold * max_triple_num_threshold, 15)
    batch_qid_triple_attMask_tensor1 = batch_qid_triple_attMask_tensor.reshape(batchsize * max_entity_num_threshold * max_triple_num_threshold, 15)

    output = model(
        input_ids=batch_qid_triple_inputId_tensor1,
        attention_mask=batch_qid_triple_attMask_tensor1,
        token_type_ids=torch.zeros(batchsize * max_entity_num_threshold * max_triple_num_threshold, 15).long().cuda(),
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,
        
        after_triple_enhanced_batch=None
    )
 
    total_qid_triple_embeddings_bs_7_5_15_768 = (output[1]).reshape(batchsize, max_entity_num_threshold, max_triple_num_threshold, 15, 768)
    total_qid_triple_embeddings_bs_7_5_768 = torch.div(output[1].sum(1), 15.0).reshape(batchsize, max_entity_num_threshold, max_triple_num_threshold, 768)

    return total_qid_triple_embeddings_bs_7_5_768, total_qid_triple_embeddings_bs_7_5_15_768


def get_masked_triple_batch(triple_collection_batch):
    s = []
    for a, i in enumerate(triple_collection_batch):
        z = []
        for b, j in enumerate(i):
            y = []
            for c, m in enumerate(j):
                if m != '0':
                    x = 1
                    print(m)
                else:
                    x = 0
                y.append(x)
            z.append(y)
        a = 0
        s.append(z)
    triple_collection_batch = torch.tensor(s)
    return triple_collection_batch



def get_full_spans(sequence_tensor: torch.Tensor, span_indices: torch.LongTensor) -> torch.Tensor:

    span_starts, span_ends = span_indices.split(1, dim=-1)  
    span_widths = span_ends - span_starts
    max_batch_span_width = span_widths.max().item() + 1  
    max_span_range_indices = util.get_range_vector(
        max_batch_span_width, util.get_device_of(sequence_tensor)
    ).view(1, 1, -1)

  
    span_mask = (max_span_range_indices <= span_widths).float()
    raw_span_indices = (span_starts + max_span_range_indices)
    final_span_indices = (raw_span_indices *
                          (raw_span_indices < sequence_tensor.size(1)).float())

    final_span_indices = (torch.nn.functional.relu(final_span_indices)).long() 
    final_span_indices_masked = (final_span_indices * span_mask).long()
    span_embeddings = util.batched_index_select(
        sequence_tensor, final_span_indices
    )

    return span_embeddings, span_mask, final_span_indices_masked

def classify_element_start_end_pos(elements_indices,batch_size,max_entity_num_threshold, max_triple_num_threshold):
    first_element_end, second_element_end, third_element_end = elements_indices.split(1,dim=-1)
    first_element_start = torch.zeros_like(first_element_end)

    second_element_start = (first_element_end + 1)
    second_element_start = torch.where(second_element_start==1, 0, second_element_start)

    third_element_start= second_element_end + 1
    third_element_start= torch.where(third_element_start==1, 0, third_element_start)

    first_span = torch.cat((first_element_start,first_element_end),-1)
    second_span = torch.cat((second_element_start, second_element_end), -1)
    third_span = torch.cat((third_element_start, third_element_end), -1)
    three_element_span = torch.stack((first_span, second_span, third_span), -1).reshape(batch_size,max_entity_num_threshold,max_triple_num_threshold,3,2).reshape(batch_size*max_entity_num_threshold*max_triple_num_threshold,3,2)

    return three_element_span


def after_calculate_similarity_logits(logits, span_embeddings, span_mask, batch_size, max_entity_num_threshold, max_triple_num_threshold):

    logits = logits.sum(3)/15.0
    logits_elements = ( span_embeddings*(span_mask.unsqueeze(-1)) ).sum(2) 
    logits_elements = logits_elements / logits_elements.size(2)
    logits_elements = logits_elements.reshape(batch_size, max_entity_num_threshold, max_triple_num_threshold, 3, 768)

    logits_expand = logits.unsqueeze(3).expand(batch_size, max_entity_num_threshold, max_triple_num_threshold, 3, 768)
    cos_similarity = torch.nn.functional.cosine_similarity(logits_expand, logits_elements,dim=-1)
    cos_similarity = torch.softmax(cos_similarity,dim=-1).unsqueeze(-1)

    sub_logits = (cos_similarity * logits_elements).sum(3)

    logits_triple_4_7_5_768 = (logits + sub_logits) / 2
    return logits_triple_4_7_5_768

def RL_multinomial_sample(sentence_hidden, logits, masked_entity_batch, \
                          batch_size, linearModel,\
                          total_qid_triple_embeddings_bs_7_5_768,
                          max_entity_num_threshold,\
                          max_triple_num_threshold,\
                          max_reward_every_100_steps,\
                          batch_qid_entity_importance_tensor, \
                          batch_qid_triples_importance_indices_tensor, \
                          batch_qid_triples_importance_values_tensor,\
                          confidence_reward):
    sentence_hidden = (sentence_hidden.sum(1) / sentence_hidden.size(1) ).unsqueeze(1)


    logits_expand = logits.unsqueeze(2).expand(batch_size, max_entity_num_threshold, max_triple_num_threshold, 768)

    cos_similarity = torch.nn.functional.cosine_similarity(logits_expand, total_qid_triple_embeddings_bs_7_5_768,dim=-1)
    cos_similarity = torch.softmax(cos_similarity,dim=-1).unsqueeze(-1)

    sub_logits = (cos_similarity * total_qid_triple_embeddings_bs_7_5_768 ).sum(2)
    logits_4_7_768 = (logits + sub_logits) / 2


    
    epsilon = torch.rand(1).cuda()
    alpha = torch.tensor(0.5).cuda()  
    avg_confidence_reward = confidence_reward.sum(-1) / batch_size
    if epsilon < avg_confidence_reward:
        new_logits = logits_4_7_768
    else:
        new_logits = alpha * logits_4_7_768 + (1 - alpha) * ((batch_qid_entity_importance_tensor.unsqueeze(-1) * logits_4_7_768) + logits_4_7_768) / 2
    
    logits_4_7_768 = new_logits
    

    sim_sen_and_7_entities = torch.cosine_similarity(sentence_hidden, logits_4_7_768, dim =-1)
    mul_dist = Categorical(torch.softmax(sim_sen_and_7_entities, dim=-1))
    picked_index = mul_dist.sample(sample_shape=[3]).transpose(1,0)
    picked_matrix = torch.zeros_like(sim_sen_and_7_entities)
    action = picked_matrix.scatter(-1, picked_index, 1)

    picked_action = action * masked_entity_batch
    picked_sim_values = sim_sen_and_7_entities.scatter(-1, picked_index, 1)
    log_prob = picked_sim_values 

    return action, picked_action, log_prob


def RL_max_sample(sentence_hidden, logits, masked_entity_batch, batch_size, prepare_for_zero_tensor, prepare_for_picked_action_max):
    sentence_hidden = (sentence_hidden.sum(1) / sentence_hidden.size(1) ).unsqueeze(1)
    logits_4_7_768 = logits
    sim_sen_and_7_entities = torch.cosine_similarity(sentence_hidden, logits_4_7_768, dim =-1)


    action_max = torch.argmax(sim_sen_and_7_entities, dim=1)  
    zero_tensor = prepare_for_zero_tensor
    picked_action_max = prepare_for_picked_action_max
    picked_action_max = zero_tensor.scatter_(1, action_max.unsqueeze(-1), picked_action_max)
    picked_action_max = picked_action_max * masked_entity_batch
    return action_max, picked_action_max

def RL_multinomial_triple_sample(picked_entity_hidden_batch_after_masked, triple_batch_hidden, masked_triple_batch, batch_size,
                                 linearModel,batch_qid_triple_element_pos_tensor,total_qid_triple_embeddings_bs_7_5_15_768,
                                 max_entity_num_threshold, max_triple_num_threshold,
                                 batch_qid_entity_importance_tensor, \
                                 batch_qid_triples_importance_indices_tensor, \
                                 batch_qid_triples_importance_values_tensor,
                                 confidence_reward):


    three_element_span = classify_element_start_end_pos(batch_qid_triple_element_pos_tensor, batch_size, max_entity_num_threshold, max_triple_num_threshold)
    span_embeddings_triple, span_mask_triple, final_span_indices_masked_triple = get_full_spans(total_qid_triple_embeddings_bs_7_5_15_768, three_element_span)
    logits_triple_4_7_5_768 = after_calculate_similarity_logits(total_qid_triple_embeddings_bs_7_5_15_768, span_embeddings_triple, span_mask_triple, batch_size, max_entity_num_threshold, max_triple_num_threshold)

    
    epsilon = torch.rand(1).cuda()
    alpha = torch.tensor(0.5).cuda()  
    avg_confidence_reward = confidence_reward.sum(-1) / batch_size
    if epsilon < avg_confidence_reward:
        new_logits = logits_triple_4_7_5_768
    else:
        new_logits = alpha * logits_triple_4_7_5_768 + (1 - alpha) * ((batch_qid_triples_importance_values_tensor.unsqueeze(-1)*logits_triple_4_7_5_768) + logits_triple_4_7_5_768) / 2
    
    logits_triple_4_7_5_768 = new_logits
    

    sim_entities_and_5_triples = torch.cosine_similarity(picked_entity_hidden_batch_after_masked.unsqueeze(2), logits_triple_4_7_5_768, dim =-1)

    mul_dist = Categorical(torch.softmax(sim_entities_and_5_triples, dim=-1))  
    picked_index = mul_dist.sample(sample_shape=[3]).permute(1,2,0)
    picked_matrix = torch.zeros_like(sim_entities_and_5_triples)
    action_triple = picked_matrix.scatter(-1, picked_index, 1)

    picked_triple = action_triple * masked_triple_batch 
    picked_sim_values = sim_entities_and_5_triples.scatter(-1, picked_index, 1)
    action_triple_log_prob = picked_sim_values  

    enhanced_entity_hidden = (picked_triple.unsqueeze(-1) * triple_batch_hidden)
    enhanced_entity_hidden = enhanced_entity_hidden.sum(2) / enhanced_entity_hidden.size(2) 


    return action_triple, action_triple_log_prob, enhanced_entity_hidden

def RL_max_triple_sample(picked_entity_hidden_batch_after_masked, triple_batch_hidden, masked_triple_batch, batch_size, prepare_for_zero_triple_tensor,
                         prepare_for_picked_action_triple_max):
    logits_triple_4_7_5_768 = triple_batch_hidden
    sim_entities_and_5_triples = torch.cosine_similarity(picked_entity_hidden_batch_after_masked.unsqueeze(2), logits_triple_4_7_5_768, dim =-1)

    action_triple_max = torch.argmax(sim_entities_and_5_triples, dim=-1)
    
    zero_triple_tensor = prepare_for_zero_triple_tensor

    picked_action_triple_max = prepare_for_picked_action_triple_max
    picked_action_triple_max = zero_triple_tensor.scatter_(2, action_triple_max.unsqueeze(-1), picked_action_triple_max)
    picked_action_triple_max = picked_action_triple_max * masked_triple_batch
    enhanced_picked_action_triple_max = picked_action_triple_max.unsqueeze(-1) * triple_batch_hidden  
    return action_triple_max, picked_action_triple_max, enhanced_picked_action_triple_max



def entity_selection(logits, entity_pos_sequence, masked_entity_batch, batch_size, linearModel, prepare_for_zero_tensor,
                     prepare_for_picked_action_max, total_qid_triple_embeddings_bs_7_5_768,
                     max_entity_num_threshold, max_triple_num_threshold, max_reward_every_100_steps,
                     batch_qid_entity_importance_tensor,
                     batch_qid_triples_importance_indices_tensor,
                     batch_qid_triples_importance_values_tensor,
                     confidence_reward):
    hidden = logits

    
    span_embeddings, span_mask, final_span_indices_masked = get_full_spans(hidden, entity_pos_sequence)
    entity_hidden_batch_after_masked = span_embeddings * span_mask.unsqueeze(-1)
    entity_hidden_batch_after_masked = entity_hidden_batch_after_masked.sum(2) / entity_hidden_batch_after_masked.size(2)
    entity_hidden_batch_after_masked = entity_hidden_batch_after_masked * masked_entity_batch.unsqueeze(-1)

    action,\
    picked_action,\
    log_prob = RL_multinomial_sample(hidden, entity_hidden_batch_after_masked,
                                    masked_entity_batch,
                                    batch_size, linearModel,
                                    total_qid_triple_embeddings_bs_7_5_768,
                                    max_entity_num_threshold,
                                    max_triple_num_threshold,
                                    max_reward_every_100_steps,
                                    batch_qid_entity_importance_tensor,
                                    batch_qid_triples_importance_indices_tensor,
                                    batch_qid_triples_importance_values_tensor,
                                    confidence_reward)
    
    action_max, picked_action_max = RL_max_sample(hidden, entity_hidden_batch_after_masked, masked_entity_batch, batch_size,
                                                  prepare_for_zero_tensor, prepare_for_picked_action_max)
    picked_entity_hidden_batch_after_masked = picked_action.unsqueeze(-1)*entity_hidden_batch_after_masked

    return action, picked_action, log_prob, action_max, picked_action_max, final_span_indices_masked, picked_entity_hidden_batch_after_masked



def dynamic_injection(picked_entity_hidden_batch_after_masked, logits, masked_triple_batch, picked_action, batch_size, linearModel,
                      prepare_for_zero_triple_tensor, prepare_for_picked_action_triple_max,
                      max_entity_num_threshold, max_triple_num_threshold,
                      batch_qid_triple_element_pos_tensor, total_qid_triple_embeddings_bs_7_5_15_768,
                      batch_qid_entity_importance_tensor,
                      batch_qid_triples_importance_indices_tensor,
                      batch_qid_triples_importance_values_tensor,
                      confidence_reward):
    logits = (picked_action.unsqueeze(-1) * logits.view(batch_size, max_entity_num_threshold, max_triple_num_threshold * 768)).view((batch_size, max_entity_num_threshold, max_triple_num_threshold, 768))
    logits = logits * masked_triple_batch.unsqueeze(-1)
    action_triple, action_triple_log_prob, enhanced_entity_hidden = RL_multinomial_triple_sample(picked_entity_hidden_batch_after_masked, logits,
                                                                                                 masked_triple_batch,
                                                                                                 batch_size,
                                                                                                 linearModel,
                                                                                                 batch_qid_triple_element_pos_tensor,
                                                                                                 total_qid_triple_embeddings_bs_7_5_15_768,
                                                                                                 max_entity_num_threshold,
                                                                                                 max_triple_num_threshold,
                                                                                                 batch_qid_entity_importance_tensor,
                                                                                                 batch_qid_triples_importance_indices_tensor,
                                                                                                 batch_qid_triples_importance_values_tensor,
                                                                                                 confidence_reward)
    action_triple_max, picked_action_triple_max, enhanced_picked_action_triple_max = RL_max_triple_sample(picked_entity_hidden_batch_after_masked, logits,
                                                                                                          masked_triple_batch,
                                                                                                          batch_size,
                                                                                                          prepare_for_zero_triple_tensor,
                                                                                                          prepare_for_picked_action_triple_max)
    return action_triple, action_triple_log_prob, enhanced_entity_hidden, action_triple_max, picked_action_triple_max, enhanced_picked_action_triple_max



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_train_path", type=str, default=data_path+'roberta_pretrain_data/',
                        help="pretrain train path to file")
    parser.add_argument("--pretrain_dev_path", type=str,
                        default=data_path+'roberta_pretrain_data/',
                        help="pretrain dev path to file")
    parser.add_argument("--max_seq_length", type=int, default=184, help="max seq length of input sequences")
    parser.add_argument("--knowledge_text_len", type=int, default=20, help="max seq length of input knowledge text")

    parser.add_argument("--do_train", type=bool, default=True, help="If do train")
    parser.add_argument("--do_lower_case", type=bool, default=True, help="If do case lower")
    parser.add_argument("--train_batch_size", type=int, default=4, help="train_batch_size")  
    parser.add_argument("--eval_batch_size", type=int, default=96, help="eval_batch_size")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="num_train_epochs")
    parser.add_argument("--learning_rate", type=float, default=4e-5, help="learning rate")  
    parser.add_argument("--warmup_proportion", type=float, default=.09,
                        help="warmup_proportion")  
    parser.add_argument("--no_cuda", type=bool, default=False, help="prevent use GPU")
    parser.add_argument("--local_rank", type=int, default=-1, help="If we ares using cluster for training")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="gradient_accumulation_steps")  
    parser.add_argument("--fp16", type=bool, default=False, help="If use apex to train")
    parser.add_argument("--loss_scale", type=int, default=0, help="loss_scale")
    parser.add_argument("--bert_config_json", type=str, default="roberta_base/bert_config.json",
                        help="bert_config_json")
    parser.add_argument("--vocab_file", type=str, default="roberta_base/vocab.txt",
                        help="Path to vocab file")
    parser.add_argument("--output_dir", type=str,
                        default="output",
                        help="output_dir")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15, help="masked_lm_prob")
    parser.add_argument("--max_predictions_per_seq", type=int, default=72, help="max_predictions_per_seq")
    parser.add_argument("--cache_dir", type=str, default='roberta_base', help="cache_dir")
    parser.add_argument("--model_name_or_path", type=str, default=data_path+"roberta_base",
                        help="model_name_or_path")
    parser.add_argument('--eval_pre_step', type=float, default=.196,
                        help="The percent of how many train with one eval run")
    parser.add_argument('--max_entity_num_threshold', type=int, default=7,
                        help="The max entity number in each sentence")
    parser.add_argument('--max_triple_num_threshold', type=int, default=5,
                        help="The max triple number for each entity")
    parser.add_argument('--update_baseline_reward_step_threshold', type=float, default=0.1,
                        help="The max triple number for each entity")
    parser.add_argument('--use_train_data_threshold', type=float, default=0.1,
                        help="The max triple number for each entity")
    args = parser.parse_args()



    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        n_gpu = torch.cuda.device_count()
        world_size = n_gpu
        torch.distributed.init_process_group(backend="nccl", init_method="env://", rank=local_rank,
                                             world_size=world_size)

        assert (torch.distributed.get_world_size() >= 2)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                        filename='execute.log')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError('Output dir is not empty!')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    set_seed(args.seed)
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path)
   
    if args.do_train:

        train_dataset = OurENRIEDataset(args=args,
                                        data_path=args.pretrain_train_path,
                                        max_seq_length=args.max_seq_length,
                                        masked_lm_prob=args.masked_lm_prob,
                                        max_predictions_per_seq=args.max_predictions_per_seq,
                                        tokenizer=tokenizer, data_type='train')


        num_train_optimization_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

        eval_dataset = OurENRIEDataset(args=args,
                                       data_path=args.pretrain_dev_path,
                                       max_seq_length=args.max_seq_length,
                                       masked_lm_prob=args.masked_lm_prob,
                                       max_predictions_per_seq=args.max_predictions_per_seq,
                                       tokenizer=tokenizer, data_type='dev')

        if (args.local_rank != -1):
            train_sampler = DistributedSampler(train_dataset, shuffle=False)
            eval_sampler = DistributedSampler(eval_dataset, shuffle=False)

        else:
            train_sampler = RandomSampler(train_dataset)
            eval_sampler = SequentialSampler(eval_dataset)
            print('single gpu sampler!!!')

        train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                      batch_size=args.train_batch_size, num_workers=2, shuffle=False, drop_last=True)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=2,
                                     shuffle=False)

    else:
        raise ValueError('Not Training Model! Please set the do_train=True!')

    
    missing_keys = set()
    model = RobertaForMaskedLM.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path)
    linearModel = torch.nn.Linear(768, 2, bias=True)

    linearModel.to(device)
    model.to(device)
   
 
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    new_add_param = [(n, p) for n, p in param_optimizer if n in missing_keys]
    pretrain_parm = [(n, p) for n, p in param_optimizer if n not in missing_keys]

    new_optimizer_grouped_parameters = [
        {'params': [p for n, p in new_add_param if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in new_add_param if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    old_optimizer_grouped_parameters = [
        {'params': [p for n, p in pretrain_parm if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in pretrain_parm if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
   
    optimizer = BertAdam(new_optimizer_grouped_parameters, lr=args.learning_rate)
    n_gpu = max(n_gpu, 1)

    for g in old_optimizer_grouped_parameters:
        optimizer.add_param_group(g)
    mylinear_para = list(linearModel.parameters())
    new_mylinear_para = []
    for g in mylinear_para:
        new_mylinear_para.append({'params': g})
    for g in new_mylinear_para:
        optimizer.add_param_group(g)

    if (args.fp16):
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        linearModel = DDP(linearModel, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        linearModel = torch.nn.DataParallel(linearModel)

    if (args.local_rank >= 0):
        torch.distributed.barrier()

    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    global_step = 0
    best_loss = 100000
    import time
    if args.do_train:

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))  
        logger.info("  Batch size = %d", args.train_batch_size)  
        logger.info("  Num steps = %d", num_train_optimization_steps)  
        model.train()
        import datetime
        fout = None
        if (args.local_rank <= 0):
            writer = SummaryWriter('TensorBoard_roberta_base_logs/tensorboard_loss_output')
            fout = open(os.path.join(args.output_dir, "model_start_time.{}".format(datetime.datetime.now())), 'w')

        total_train_step = len(train_dataloader) * args.num_train_epochs  
        total_eval_step = int(len(eval_dataset) / args.eval_batch_size)  
        if args.local_rank <= 0:
            logger.info('In DP/DDP, Total step:{}, Eval step: {}'.format(total_train_step, total_eval_step))

 
        if args.local_rank != -1:
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        args.warmup_proportion * num_train_optimization_steps * args.num_train_epochs,
                                                        num_train_optimization_steps * args.num_train_epochs)
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        args.warmup_proportion * total_train_step,
                                                        total_train_step)

        tr_loss = 0
        loss_step = 0
        batch_loss = 0
        total_time = []
        total_size = 0
        stack_total_reward = None
        for epoch in range(int(args.num_train_epochs)):
            if (args.local_rank != -1):
                train_dataloader.sampler.set_epoch(epoch)
            confidence_reward = torch.mean(torch.tensor([0.5]*args.train_batch_size)).cuda()
            TotalReward_cache = []
            max_reward_every_100_steps = 0.0
            tqdm_var = tqdm(train_dataloader)
            for step, batch in enumerate(tqdm_var):

                batch0 = tuple(t.to(device) for t in batch)

               
                prepare_for_zero_tensor = torch.zeros(args.train_batch_size, args.max_entity_num_threshold).cuda()
                prepare_for_zero_triple_tensor = torch.zeros(args.train_batch_size, args.max_entity_num_threshold, args.max_triple_num_threshold).cuda()

                prepare_for_picked_action_max = torch.ones(args.train_batch_size, args.max_entity_num_threshold).cuda()
                prepare_for_picked_action_triple_max = torch.ones(args.train_batch_size, args.max_entity_num_threshold, args.max_triple_num_threshold).cuda()

                prepare_for_tmp_for_mask_span = torch.zeros(args.train_batch_size, args.max_entity_num_threshold, args.max_seq_length).cuda()
                prepare_for_temp_ones = torch.ones(args.train_batch_size, args.max_seq_length).cuda()


                all_input_ids, all_input_mask, all_segment_ids, all_label_ids,\
                lm_label_array_entity, lm_label_array_token, \
                batch_qid_triple_inputId_tensor,\
                batch_qid_triple_attMask_tensor,\
                batch_qid_triple_element_pos_tensor, \
                batch_qid_entity_importance_tensor, \
                batch_qid_triples_importance_indices_tensor, \
                batch_qid_triples_importance_values_tensor, \
                entity_pos_sequence, math_num_cut_q_qid = batch0


                batch_qid_triple_inputId_tensor = batch_qid_triple_inputId_tensor.squeeze(1)
                batch_qid_triple_attMask_tensor = batch_qid_triple_attMask_tensor.squeeze(1)
                batch_qid_triple_element_pos_tensor = batch_qid_triple_element_pos_tensor.squeeze(1)
                batch_qid_entity_importance_tensor = batch_qid_entity_importance_tensor.squeeze(1)
                batch_qid_triples_importance_indices_tensor = batch_qid_triples_importance_indices_tensor.squeeze(1)
                batch_qid_triples_importance_values_tensor = batch_qid_triples_importance_values_tensor.squeeze(1)
               
                mask_batch_qid_triple_tensor = torch.where((batch_qid_triple_attMask_tensor.sum(-1)) > 0, 1, 0)

                total_size += all_input_ids.size(0)
                start_time = time.time()
                after_triple_enhanced_batch = None
                outputs = model(
                    input_ids=all_input_ids,
                    attention_mask=all_input_mask,
                    token_type_ids=torch.zeros(all_input_ids.size(0), 128).long().cuda(),
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    labels=all_label_ids,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=False,
                    after_triple_enhanced_batch=after_triple_enhanced_batch
                ) 


                
                total_qid_triple_embeddings_bs_7_5_768, \
                total_qid_triple_embeddings_bs_7_5_15_768 = get_batch_triple_hidden(batch_qid_triple_inputId_tensor,
                                                                                 batch_qid_triple_attMask_tensor,
                                                                                 model,
                                                                                 args.train_batch_size,
                                                                                 args.max_entity_num_threshold,
                                                                                 args.max_triple_num_threshold)

                masked_entity_batch = torch.where(entity_pos_sequence.sum(-1) > 0, 1, 0)

                action, picked_action, log_prob, action_max, picked_action_max, \
                final_span_indices_masked, \
                picked_entity_hidden_batch_after_masked = entity_selection(outputs[2], entity_pos_sequence, 
                                                             masked_entity_batch, args.train_batch_size,
                                                             linearModel, prepare_for_zero_tensor,
                                                             prepare_for_picked_action_max,
                                                             total_qid_triple_embeddings_bs_7_5_768,
                                                             args.max_entity_num_threshold,
                                                             args.max_triple_num_threshold,
                                                             max_reward_every_100_steps,
                                                             batch_qid_entity_importance_tensor,
                                                             batch_qid_triples_importance_indices_tensor,
                                                             batch_qid_triples_importance_values_tensor,
                                                             confidence_reward)

                

                action_triple, action_triple_log_prob, enhanced_entity_hidden, action_triple_max, picked_action_triple_max, \
                enhanced_picked_action_triple_max = dynamic_injection(picked_entity_hidden_batch_after_masked, total_qid_triple_embeddings_bs_7_5_768,
                                                                      mask_batch_qid_triple_tensor, picked_action,
                                                                      args.train_batch_size, linearModel,
                                                                      prepare_for_zero_triple_tensor,
                                                                      prepare_for_picked_action_triple_max,
                                                                      args.max_entity_num_threshold,
                                                                      args.max_triple_num_threshold,
                                                                      batch_qid_triple_element_pos_tensor,
                                                                      total_qid_triple_embeddings_bs_7_5_15_768,
                                                                      batch_qid_entity_importance_tensor,
                                                                      batch_qid_triples_importance_indices_tensor,
                                                                      batch_qid_triples_importance_values_tensor,
                                                                      confidence_reward)


                entity_pos_sequence_start_end_pair = entity_pos_sequence  
                tmp_for_mask_span = prepare_for_tmp_for_mask_span  

                mask_span_pos_with_one = tmp_for_mask_span.scatter_(2, final_span_indices_masked, 1) 
                mask_span_pos_with_one = mask_span_pos_with_one * masked_entity_batch.unsqueeze(-1)
                
                mask_span_pos_with_one_bs_128 = torch.where(mask_span_pos_with_one.sum(1)!=0, 0, 1) 
                entity_pos_hidden_zero = mask_span_pos_with_one_bs_128.unsqueeze(-1) * outputs[2]

                mask_span_pos_hidden = torch.bmm((mask_span_pos_with_one.permute(0, 2, 1)), enhanced_entity_hidden) 
                final_batch_embedding = entity_pos_hidden_zero + mask_span_pos_hidden  
               
                after_triple_enhanced_batch = final_batch_embedding
                outputs_tokeId = model(
                    input_ids=all_input_ids,
                    attention_mask=all_input_mask,
                    token_type_ids=torch.zeros(all_input_ids.size(0), 128).long().cuda(),
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    labels=all_label_ids,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=False,           
                    after_triple_enhanced_batch=after_triple_enhanced_batch
                )  
                temp_zeros = prepare_for_tmp_for_mask_span 
                entity_pos_mask_span_zero_max = final_span_indices_masked * picked_action_max.unsqueeze(-1)
                entity_pos_mask_zero_max = temp_zeros.scatter_(2, entity_pos_mask_span_zero_max.long(), 1)
                mask_span_pos_with_one_max = entity_pos_mask_zero_max * masked_entity_batch.unsqueeze(-1)
                mask_span_pos_with_one_bs_128_max = torch.where(mask_span_pos_with_one_max.sum(1)!=0, 0, 1) 
                entity_pos_hidden_zero_max =  mask_span_pos_with_one_bs_128_max.unsqueeze(-1) * outputs[2] 
                enhanced_picked_action_triple_max_bs_7_768 = enhanced_picked_action_triple_max.sum(2) / enhanced_picked_action_triple_max.size(2) 
                
                enhanced_picked_action_entity_max_bs_7_768 = picked_action_max.unsqueeze(-1) * enhanced_picked_action_triple_max_bs_7_768   
                mask_span_pos_with_one_max = mask_span_pos_with_one * picked_action_max.unsqueeze(-1) 
                mask_span_pos_hidden_max = torch.bmm((mask_span_pos_with_one_max.permute(0, 2, 1)),
                                                     enhanced_picked_action_entity_max_bs_7_768) 
                final_max_batch_embedding = mask_span_pos_hidden_max + entity_pos_hidden_zero_max 
                

                after_triple_enhanced_batch = final_max_batch_embedding
                outputs_tokeId_max = model(
                    input_ids=all_input_ids,
                    attention_mask=all_input_mask,
                    token_type_ids=torch.zeros(all_input_ids.size(0), 128).long().cuda(),
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    labels=all_label_ids,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=False
                    after_triple_enhanced_batch=after_triple_enhanced_batch
                )  
                tmp_mask_entity = torch.where(lm_label_array_entity != (-100), 1, 0)  

                reverse_predict_entity = tmp_mask_entity * outputs_tokeId[1][0].argmax(dim=2) 
                ground_truth_label_array_entity = lm_label_array_entity * tmp_mask_entity  

                tmp_mask_token = torch.where(lm_label_array_token != (-100), 1, 0) 
                reverse_predict_token = tmp_mask_token * outputs_tokeId[1][0].argmax(dim=2) 
                ground_truth_label_array_token = lm_label_array_token * tmp_mask_token 

                tmp_mask_all_max = torch.where(all_label_ids != (-100), 1, 0)  
                reverse_predict_all_max = tmp_mask_all_max * outputs_tokeId_max[1][0].argmax(dim=2)  
                ground_truth_label_array_all_max = all_label_ids * tmp_mask_all_max  
                
                ground_truth_label_array_entity = torch.where(ground_truth_label_array_entity==0, -100, ground_truth_label_array_entity)  
                ground_truth_label_array_token = torch.where(ground_truth_label_array_token==0, -100, ground_truth_label_array_token)  
                ground_truth_label_array_all_max = torch.where(ground_truth_label_array_all_max==0, -100, ground_truth_label_array_all_max)

                entity_ACC_reward = torch.eq(reverse_predict_entity, ground_truth_label_array_entity).sum(1)  
                token_ACC_reward = torch.eq(reverse_predict_token, ground_truth_label_array_token).sum(1)  
                baseline_reward = torch.eq(reverse_predict_all_max, ground_truth_label_array_all_max).sum(1) 

                Total_reward = entity_ACC_reward + token_ACC_reward

                if step == 0:
                    confidence_reward = torch.mean(torch.tensor([0.5]*args.train_batch_size)).cuda()  
                    TotalReward_cache.append(Total_reward)
                else:
                    if step % 100 == 0:
                        TotalReward_cache.append(Total_reward)
                        max_reward_every_100_steps = torch.max(torch.stack(TotalReward_cache),dim=0).values
                        confidence_reward = (torch.nan_to_num(Total_reward / max_reward_every_100_steps)).sum(-1) / args.train_batch_size
                        if confidence_reward > torch.mean(torch.tensor([1.0]*args.train_batch_size)).cuda() :
                            confidence_reward = Total_reward / (Total_reward + max_reward_every_100_steps)
                        TotalReward_cache = []
                    else:
                        TotalReward_cache.append(Total_reward)
                        if step < 100:
                            confidence_reward = torch.mean(torch.tensor([0.5]*args.train_batch_size)).cuda()
                        else:
                            confidence_reward = (torch.nan_to_num(Total_reward / max_reward_every_100_steps)).sum(-1) / args.train_batch_size
                            if confidence_reward > torch.mean(torch.tensor([1.0]*args.train_batch_size)).cuda():
                                confidence_reward = Total_reward / (Total_reward + max_reward_every_100_steps)

                update_baseline_reward_step = int(len(train_dataloader) * args.update_baseline_reward_step_threshold)
                if epoch == 0 and step == 0:
                    baseline_reward = torch.zeros(Total_reward.size()).cuda()
                    stack_total_reward = Total_reward.view(1, Total_reward.size(0))
                elif (step+1) % update_baseline_reward_step == 0:
                    baseline_reward = torch.mean(stack_total_reward.float(), dim=0)
                    stack_total_reward = Total_reward.view(1, Total_reward.size(0))
                else:
                    stack_total_reward = torch.cat((stack_total_reward, Total_reward.view(1, Total_reward.size(0))), dim=0)
                reward_diff = Total_reward - baseline_reward

                rl_loss_entity_selection = (-(Total_reward - baseline_reward) * (log_prob).sum(1)) / args.max_entity_num_threshold
                rl_loss_dynamic_injection = (-(Total_reward - baseline_reward) * ((action_triple_log_prob).sum(1).sum(1))) / args.max_entity_num_threshold / args.max_triple_num_threshold
                rl_loss = rl_loss_entity_selection + rl_loss_dynamic_injection
                rl_loss = torch.mean(rl_loss)

                loss = outputs_tokeId[0] + rl_loss

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps 

                batch_loss += loss.mean().item()
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                end_time = time.time()
                total_time.append(end_time - start_time)
                tr_loss += float(loss.item() * args.gradient_accumulation_steps)
                if args.local_rank <= 0:
                    logger.info("epoch: {}   Step: {} / {}   total loss: {} | mlm loss: {} | rl loss: {}  | mean reward diff : {} | Total reward: {}".format(epoch,
                                                                                                                  step,
                                                                                                                  len(train_dataloader),
                                                                                                                  loss.item(),
                                                                                                                  outputs_tokeId[0].mean().cpu().tolist(),
                                                                                                                  rl_loss.item(),
                                                                                                                 sum(reward_diff.cpu().tolist())/len(reward_diff.cpu().tolist()),
                                                                                                                Total_reward.sum(0).item()))
                if loss.item() > 50:
                    if args.local_rank <= 0:
                        logger.info('---------loss exploration!!!!----------')
                    sys.exit(0)
                loss_step += 1
                if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):                   
                    if args.local_rank <= 0:
                        writer.add_scalar('LOSS', batch_loss, global_step=global_step)
                    batch_loss = 0
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                    optimizer.zero_grad()

                    if (step + 1) % int(len(train_dataloader)*0.2) == 0:
                       
                        if args.fp16:
                            if args.local_rank <= 0:
                                logger.info(
                                    '**************************************************************************')
                                model_to_save = model.module if hasattr(model,
                                                                        'module') else model  # Only save the model it-self
                                output_model_file = os.path.join(args.output_dir,
                                                                 str(global_step) + "_pytorch_model.bin")
                                logger.info('Saving checkpoint into {}'.format(output_model_file))
                                torch.save(model_to_save.state_dict(), output_model_file)
                                logger.info('Saving checkpoint Done!')
                                logger.info(
                                    '**************************************************************************')
                        else:
                            
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Only save the model it-self
                            output_model_file = os.path.join(args.output_dir, str(global_step) + "_pytorch_model.bin")
                            if (args.local_rank <= 0):
                                logger.info(
                                    '**************************************************************************')
                                logger.info('Saving checkpoint into {}'.format(output_model_file))
                                torch.save(model_to_save.state_dict(), output_model_file)
                                logger.info('Saving checkpoint Done!')
                                logger.info(
                                    '**************************************************************************')
                        
                        tr_loss = 0
                        loss_step = 0
        print('total time: {}'.format(sum(total_time)))
        if (args.local_rank <= 0):
            logger.info('Training Done!')

if __name__ == "__main__":
    main()

