import os
import time
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import argparse
from gpt2_training.train_utils import load_model, boolean_string, set_lr, get_eval_list_same_length

from tensorboardX import SummaryWriter
from keras import backend as K                                 
from keras.preprocessing.sequence import pad_sequences     
from keras.models import load_model
from gensim.models import word2vec
import tensorflow as tf
from torch import optim
from lsp_model import Adam
from Emo_detector.detect_emotion import re_emo_score, prepare_model
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3) 
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
from main1 import chatbot
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1) 
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
from transformers import AdamW
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from data_loader import BucketingDataLoader, DynamicBatchingLoader, DistributedBucketingDataLoader
from chat_load import post_set
from gpt2_training.train_utils import set_lr
#tf.keras.backend.set_session(sess)
#model = word2vec.Word2Vec.load('word2vec.model')
#model2 = load_model("model.h5")
import string
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW
def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
    """
    # batch support!
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
       # print(values.shape)
        min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(logits < min_values, 
                             torch.ones_like(logits, dtype=logits.dtype) * -float('Inf'), 
                             logits)
    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        sorted_logits = sorted_logits.masked_fill_(sorted_indices_to_remove, filter_value)
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)
    
    return logits
#def train()
temperature = 0.7 #2.2
top_k = 50        #50
top_p = 0.95
def make_response(model, sentences, tokenizer, first_input, fil, turn):
    with torch.no_grad():
        sentences = [tokenizer.encode(x) for x in sentences]
        t = []
        for i in range(len(sentences)):
            t_0 = [0 for i in range(len(list(first_input[i])))]
            t_1 = [1 for i in range(len(list(sentences[i]))-1)]
            sentences[i] = list(first_input[i]) + list(sentences[i])
         #   print(tokenizer.decode(sentences[i]))
            t.append(t_0[:] + t_1[:])
            #print(tokenizer.decode(sentences[i]))
        m = []

        # for i in range(len(sentences)):
        #     if len(sentences[i]) == 1:
        #         sentences[i].append(3)
        sentences = [x[:-1] for x in sentences]
        p = []
        for i in range(len(sentences)):
            temp_m = [1 for x in range(len(sentences[i]))]
            m.append(temp_m[:])
            #p.append(list(range(len(sentences[i])))
        eos = [tokenizer.encoder["<|endoftext|>"]]

        prev_input = pad_sequence([torch.LongTensor(x) for x in sentences], batch_first=True, padding_value=0).to(device)
        m = pad_sequence([torch.LongTensor(x) for x in m], batch_first=True, padding_value=0).to(device)
        t = pad_sequence([torch.LongTensor(x) for x in t], batch_first=True, padding_value=0).to(device)
        if turn:
            _, past = model(prev_input, past=None, attention_mask=m, token_type_ids=t)
        else:
            _, past = model(prev_input, past=None, attention_mask=m)
        prev_input = torch.LongTensor([[eos] * len(sentences)]).to(device)
        temp_sen = [[] for i in range(len(sentences))]
        for i in range(128):
            prev_input, past = model(prev_input, past=past)
            #print(logits.squeeze(0).squeeze(1).shape)
            #logits = logits[:, :, :] / temperature
            #logits = top_filtering(logits, top_k=top_k, top_p=top_p)
            prev_input = prev_input.squeeze(0).squeeze(1)
            prev_input = prev_input / temperature
            if fil:
                prev_input = top_filtering(prev_input, top_k=top_k, top_p=top_p)
            prev_input = torch.softmax(prev_input, dim=-1)

            prev_input = torch.multinomial(prev_input, num_samples=1)
            #print(prev_input.shape)
            #prev_word = prev_input.item()
            if i == 0:
                for j in range(len(sentences)):
                    
                    temp_sen[j].append(prev_input[j].item())
                continue
            flag = 1
            
            for j in range(len(sentences)):
                if temp_sen[j][-1] != eos[0]: 
                    flag = 0
                    temp_sen[j].append(prev_input[j].item())
            if flag == 1: break
        a = [tokenizer.decode(x).replace('<|endoftext|>', '') for x in temp_sen]
    return a

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                    help='pretrained model name or path to local checkpoint')
    parser.add_argument('--filename', type=str,
                    help='pretrained model name or path to local checkpoint')
    parser.add_argument('--fil', type=boolean_string, default=False, help='turn on progress bar')
    parser.add_argument('--turn', type=boolean_string, default=False, help='turn on progress bar')
    parser.add_argument("--inter", type=str, default="gpt")
    args = parser.parse_args()
    model_train = GPT2LMHeadModel.from_pretrained(args.model)
    model_train.to(device)

   # if args.inter == 'gpt':
    model_bot = GPT2LMHeadModel.from_pretrained('models/medium/')
    model_bot.to(device)
    model_bot.eval()
    #elif args.inter == 'google':

    jack = chatbot.Chatbot()
    jack.main(['--test', 'daemon', '--rootDir', 'deepqa', '--maxLength', '20'])
   # elif args.inter == 'retrieve':
    with torch.no_grad():
        from retrieval_model.retrieval_chatbot import Retrievalchatbot
        ret_model = Retrievalchatbot()
    
    model_train.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    eos = [tokenizer.encoder["<|endoftext|>"]]
    s = post_set('data/test.tsv', tokenizer)
    train_dataloader = DataLoader(s, batch_size=128, shuffle=False, num_workers=2)
    with torch.no_grad():
        with open(args.filename, 'w') as f:
            for inputs_id, mask, ll in tqdm(train_dataloader):
                inputs_id = inputs_id.to(device)
                mask = mask.to(device)
                prev_input, past = model_train(inputs_id, past=None, attention_mask=mask)
                prev_input = torch.LongTensor([[eos] * inputs_id.shape[0]]).to(device)
                temp_sen = [[] for i in range(inputs_id.shape[0])]

                for i in range(128):
                    prev_input = prev_input.to(device)
                    logits, past = model_train(prev_input, past=past)
                    logits = logits.squeeze(0).squeeze(1)
                    logits = logits / temperature
                    if args.fil:
                        logits = top_filtering(logits, top_k=top_k, top_p=top_p)
                    logits = torch.softmax(logits, dim=-1)
                    prev_input = torch.multinomial(logits, num_samples=1)
                    if i == 0:
                        for j in range(inputs_id.shape[0]):
                            
                            temp_sen[j].append(prev_input[j].item())
                        continue
                    flag = 1
                    
                    for j in range(0, inputs_id.shape[0]):
                        if temp_sen[j][-1] != eos[0]: 
                            flag = 0
                            temp_sen[j].append(prev_input[j].item())
                    if flag == 1: break
                a = [tokenizer.decode(x) for x in temp_sen]
                first_input = list(inputs_id.cpu().detach().numpy())
                for j in range(inputs_id.shape[0]):
                    l = ll[j]
                    first_input[j] = first_input[j][:l+1]
                    first_input[j][-1] = eos[0]
                
                #if args.inter == 'gpt':
                k = make_response(model_bot, a, tokenizer, first_input, args.fil, args.turn)
                print(k)
            #elif args.inter == 'google':
                #k = []
                for j in range(inputs_id.shape[0]):
                    k.append(jack.daemonPredict(sentence=a[j].replace('<|endoftext|>', '')))
            # elif args.inter == 'retrieve':
                ii = []
                for j in range(inputs_id.shape[0]): 
                # ii = [tokenizer.decode(x[:-1]) for x in first_input]
                    ii.append([tokenizer.decode(first_input[j][:-1]), a[j].replace('<|endoftext|>', '')])
                    
                k.extend(ret_model.get_response(ii))
               # print(ret_model.get_response(ii))
                fir = [tokenizer.decode(x).replace('<|endoftext|>', '') for x in first_input]

                for i in range(inputs_id.shape[0]):
                    f.write(fir[i] + '\n')
                    f.write(a[i].replace('<|endoftext|>', '') + '\n')
                    f.write(k[i] + '\n')
                    f.write(k[i+inputs_id.shape[0]] +'\n')
                    f.write(k[i+inputs_id.shape[0]*2] + '\n')
                    f.write('-----------------------\n')
if __name__ == '__main__':
    main()
