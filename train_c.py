import os
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from os.path import join
import re
from argparse import ArgumentParser

from Emo_detector.detect_emotion import re_emo_score, prepare_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from chat_load import post_set
from lsp_model.optim import Adam

import string
from tqdm import tqdm
torch.manual_seed(100)
emo_dict = {
                "<afraid>": [-0.12, 0.79], 
                "<angry>": [-0.42, 0.79], 
                "<annoyed>": [-0.44, 0.66], 
                "<anticipating>": [0.32, 0.06], # expectant
                "<anxious>":[-0.72, -0.8] , 
                    "<apprehensive>": [-0.77, -0.6], 
                "<ashamed>": [-0.45, -0.5], 
                    "<caring>": [0.25, -0.5], # not worried 
                "<confident>": [0.51, -0.2], 
                "<content>": [0.82, -0.55], 
                    "<devastated>": [-0.8, -0.5], 
                "<disappointed>": [-0.8, -0.03], 
                "<disgusted>": [-0.67, 0.49], 
                "<embarrassed>": [-0.32, -0.6], # caring
                "<excited>": [0.7, 0.72], 
                    "<faithful>": [0.6, 0.2], 
                    "<furious>": [-0.7, 0.85], 
                    "<grateful>": [0.8, -0.08], 
                "<guilty>": [-0.4, -0.43], # feel guilt
                "<hopeful>": [0.62, -0.3], 
                "<impressed>": [0.38, -0.07], 
                "<jealous>": [-0.08, 0.56], 
                "<joyful>": [0.85, 0.15], 
                    "<lonely>": [-0.2, -0.8], 
                    "<nostalgic>": [-0.5, -0.2],  # sentimental
                    # "prepared": , # confident
                    "<proud>": [0.45, 0.07],
                "<sad>": [-0.82, -0.4], 
                    # "sentimental": ,# nostalgic
                "<surprised>": [0.42, 0.79],  # astonished
                    # "terrified": , # afraid
                    "<trusting>": [0.3, 0.2]
            }

detect_model, detect_processor, emotion_tokenizer = prepare_model()
bad_word = ["4r5e", "5h1t", "5hit", "a55", "anal", "anus", "ar5e", "arrse", "arse", "ass", "ass-fucker", "asses", "assfucker", "assfukka", "asshole", "assholes", "asswhole", "a_s_s", "b!tch", "b00bs", "b17ch", "b1tch", "ballbag", "balls", "ballsack", "bastard", "beastial", "beastiality", "bellend", "bestial", "bestiality", "bi+ch", "biatch", "bitch", "bitcher", "bitchers", "bitches", "bitchin", "bitching", "bloody", "blow job", "blowjob", "blowjobs", "boiolas", "bollock", "bollok", "boner", "boob", "boobs", "booobs", "boooobs", "booooobs", "booooooobs", "breasts", "buceta", "bugger", "bum", "bunny fucker", "butt", "butthole", "buttmuch", "buttplug", "c0ck", "c0cksucker", "carpet muncher", "cawk", "chink", "cipa", "cl1t", "clit", "clitoris", "clits", "cnut", "cock", "cock-sucker", "cockface", "cockhead", "cockmunch", "cockmuncher", "cocks", "cocksuck", "cocksucked", "cocksucker", "cocksucking", "cocksucks", "cocksuka", "cocksukka", "cok", "cokmuncher", "coksucka", "coon", "cox", "crap", "cum", "cummer", "cumming", "cums", "cumshot", "cunilingus", "cunillingus", "cunnilingus", "cunt", "cuntlick", "cuntlicker", "cuntlicking", "cunts", "cyalis", "cyberfuc", "cyberfuck", "cyberfucked", "cyberfucker", "cyberfuckers", "cyberfucking", "d1ck", "damn", "dick", "dickhead", "dildo", "dildos", "dink", "dinks", "dirsa", "dlck", "dog-fucker", "doggin", "dogging", "donkeyribber", "doosh", "duche", "dyke", "ejaculate", "ejaculated", "ejaculates", "ejaculating", "ejaculatings", "ejaculation", "ejakulate", "f u c k", "f u c k e r", "f4nny", "fag", "fagging", "faggitt", "faggot", "faggs", "fagot", "fagots", "fags", "fanny", "fannyflaps", "fannyfucker", "fanyy", "fatass", "fcuk", "fcuker", "fcuking", "feck", "fecker", "felching", "fellate", "fellatio", "fingerfuck", "fingerfucked", "fingerfucker", "fingerfuckers", "fingerfucking", "fingerfucks", "fistfuck", "fistfucked", "fistfucker", "fistfuckers", "fistfucking", "fistfuckings", "fistfucks", "flange", "fook", "fooker", "fuck", "fucka", "fucked", "fucker", "fuckers", "fuckhead", "fuckheads", "fuckin", "fucking", "fuckings", "fuckingshitmotherfucker", "fuckme", "fucks", "fuckwhit", "fuckwit", "fudge packer", "fudgepacker", "fuk", "fuker", "fukker", "fukkin", "fuks", "fukwhit", "fukwit", "fux", "fux0r", "f_u_c_k", "gangbang", "gangbanged", "gangbangs", "gaylord", "gaysex", "goatse", "God", "god-dam", "god-damned", "goddamn", "goddamned", "hardcoresex", "hell", "heshe", "hoar", "hoare", "hoer", "homo", "hore", "horniest", "horny", "hotsex", "jack-off", "jackoff", "jap", "jerk-off", "jism", "jiz", "jizm", "jizz", "kawk", "knob", "knobead", "knobed", "knobend", "knobhead", "knobjocky", "knobjokey", "kock", "kondum", "kondums", "kum", "kummer", "kumming", "kums", "kunilingus", "l3i+ch", "l3itch", "labia", "lmfao", "lust", "lusting", "m0f0", "m0fo", "m45terbate", "ma5terb8", "ma5terbate", "masochist", "master-bate", "masterb8", "masterbat*", "masterbat3", "masterbate", "masterbation", "masterbations", "masturbate", "mo-fo", "mof0", "mofo", "mothafuck", "mothafucka", "mothafuckas", "mothafuckaz", "mothafucked", "mothafucker", "mothafuckers", "mothafuckin", "mothafucking", "mothafuckings", "mothafucks", "mother fucker", "motherfuck", "motherfucked", "motherfucker", "motherfuckers", "motherfuckin", "motherfucking", "motherfuckings", "motherfuckka", "motherfucks", "muff", "mutha", "muthafecker", "muthafuckker", "muther", "mutherfucker", "n1gga", "n1gger", "nazi", "nigg3r", "nigg4h", "nigga", "niggah", "niggas", "niggaz", "nigger", "niggers", "nob", "nob jokey", "nobhead", "nobjocky", "nobjokey", "numbnuts", "nutsack", "orgasim", "orgasims", "orgasm", "orgasms", "p0rn", "pawn", "pecker", "penis", "penisfucker", "phonesex", "phuck", "phuk", "phuked", "phuking", "phukked", "phukking", "phuks", "phuq", "pigfucker", "pimpis", "piss", "pissed", "pisser", "pissers", "pisses", "pissflaps", "pissin", "pissing", "pissoff", "poop", "porn", "porno", "pornography", "pornos", "prick", "pricks", "pron", "pube", "pusse", "pussi", "pussies", "pussy", "pussys", "rectum", "retard", "rimjaw", "rimming", "s hit", "s.o.b.", "sadist", "schlong", "screwing", "scroat", "scrote", "scrotum", "semen", "sex", "sh!+", "sh!t", "sh1t", "shag", "shagger", "shaggin", "shagging", "shemale", "shi+", "shit", "shitdick", "shite", "shited", "shitey", "shitfuck", "shitfull", "shithead", "shiting", "shitings", "shits", "shitted", "shitter", "shitters", "shitting", "shittings", "shitty", "skank", "slut", "sluts", "smegma", "smut", "snatch", "son-of-a-bitch", "spac", "spunk", "s_h_i_t", "t1tt1e5", "t1tties", "teets", "teez", "testical", "testicle", "tit", "titfuck", "tits", "titt", "tittie5", "tittiefucker", "titties", "tittyfuck", "tittywank", "titwank", "tosser", "turd", "tw4t", "twat", "twathead", "twatty", "twunt", "twunter", "v14gra", "v1gra", "vagina", "viagra", "vulva", "w00se", "wang", "wank", "wanker", "wanky", "whoar", "whore", "willies", "willy", "xrated", "xxx"]
bad_dict = {}
for w in bad_word:
    bad_dict[w] = 1


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
temperature = 1 #2.2
top_k = 50        #50
top_p = 0.95
device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_response(model, sentences, tokenizer, first_input):
    with torch.no_grad():

        sentences = [tokenizer.encode(x) for x in sentences]
        t = []
        for i in range(len(sentences)):
            t_0 = [0 for i in range(len(list(first_input[i])))]
            t_1 = [1 for i in range(len(list(sentences[i]))-1)]
            sentences[i] = list(first_input[i]) + list(sentences[i])

            t.append(t_0[:] + t_1[:])
        mask= []


        sentences = [x[:-1] for x in sentences]
        for i in range(len(sentences)):
            mask.append([1 for x in range(len(sentences[i]))])
        eos = [tokenizer.encoder["<|endoftext|>"]]

        prev_input = pad_sequence([torch.LongTensor(x) for x in sentences], batch_first=True, padding_value=0).to(device_1)
        mask = pad_sequence([torch.LongTensor(x) for x in mask], batch_first=True, padding_value=0).to(device_1)
        _, past = model(prev_input, past=None, attention_mask=mask)
        prev_input = torch.LongTensor([[eos] * len(sentences)]).to(device_1)
        temp_sentence = [[] for i in range(len(sentences))]
        for i in range(128):
            prev_input, past = model(prev_input, past=past)

            prev_input = prev_input.squeeze(0).squeeze(1)
            prev_input = prev_input / 0.7
            prev_input = torch.softmax(prev_input, dim=-1)

            prev_input = torch.multinomial(prev_input, num_samples=1)

            if i == 0:
                for j in range(len(sentences)):
                    temp_sentence[j].append(prev_input[j].item())
                continue
            flag = 1
            
            for j in range(len(sentences)):
                if temp_sentence[j][-1] != eos[0]: 
                    flag = 0
                    temp_sentence[j].append(prev_input[j].item())
            if flag == 1: break
    return [[tokenizer.decode(x).replace('<|endoftext|>', '')] for x in temp_sentence]
    
def train(model_train, inputs_id, mask, model_2, model_bot, tokenizer, ll, args, batch_size):
    loss = 0
    inputs_id = inputs_id.to(device_0)
    emo_embed = emo_dict['<'+args.emotion+'>']
    eos = [tokenizer.encoder["<|endoftext|>"]]
    mask = mask.to(device_0)
    prev_input, past = model_train(inputs_id, past=None, attention_mask=mask)
    inputs_id = inputs_id.to(device_1)
    mask = mask.to(device_1)
    with torch.no_grad():
        prev_input, past_bot = model_2(inputs_id, past=None, attention_mask=mask)
    prev_input = torch.LongTensor([[eos] * inputs_id.shape[0]]).to(device_0)

    temp_sentence = [[] for i in range(inputs_id.shape[0])]
    emotion_loss = [0 for i in range(inputs_id.shape[0])]
    coherence_loss = [0 for i in range(inputs_id.shape[0])]

    for i in range(40):
        prev_input = prev_input.to(device_0)
        logits, past = model_train(prev_input, past=past)
        prev_input = prev_input.to(device_1)

        with torch.no_grad():
            logits_bot, past_bot = model_2(prev_input, past=past_bot)

        logits = logits.squeeze(0).squeeze(1)
        logits = logits / temperature

        logits = torch.softmax(logits, dim=-1)
        with torch.no_grad():
            logits_bot = torch.softmax(logits_bot.squeeze(0).squeeze(1) / temperature, dim=-1)
        prev_input = torch.multinomial(logits[:], num_samples=1)

        probs = []
        for j in range(inputs_id.shape[0]):
            if i != 0 and temp_sentence[j][-1] == eos[0]: 
                continue
            probs.append(logits_bot[j][prev_input[j][0].item()].item())
        if len(probs) == 0:
            avg_prob = 0
        else:
            avg_prob = sum(probs) / len(probs)

        for j in range(inputs_id.shape[0]):
            if i != 0 and temp_sentence[j][-1] == eos[0]: continue
            temp_loss = F.cross_entropy(logits[j].unsqueeze(0), prev_input.view(-1)[j].unsqueeze(0))
            coherence_loss[j] += (logits_bot[j][prev_input[j][0].item()].item() - avg_prob) * temp_loss
            emotion_loss[j] += temp_loss

        if i == 0:
            for j in range(inputs_id.shape[0]):
                temp_sentence[j].append(prev_input[j].item())
            continue
        flag = 1
        
        for j in range(0, inputs_id.shape[0]):
            if temp_sentence[j][-1] != eos[0]: 
                flag = 0
                temp_sentence[j].append(prev_input[j].item())
        if flag == 1: break
    decode_temp_sentence = [tokenizer.decode(x) for x in temp_sentence]
    eos = [tokenizer.encoder["<|endoftext|>"]]
    first_input = list(inputs_id.cpu().detach().numpy())
    for j in range(inputs_id.shape[0]):
        l = ll[j]
        first_input[j] = first_input[j][:l+1]
        first_input[j][-1] = eos[0]
    inter_response = []
    if 'gpt' in args.inter:
        inter_response.extend(make_response(model_bot, decode_temp_sentence, tokenizer, first_input))
    # if 'google' in args.inter:
    #     #k = []
    #     for j in range(inputs_id.shape[0]):
    #         k.append([jack.daemonPredict(sentence=a[j].replace('<|endoftext|>', ''))])
    # if 'retrieve' in args.inter:
    #     ii = []
    #     for j in range(inputs_id.shape[0]): 
    #         # ii = [tokenizer.decode(x[:-1]) for x in first_input]
    #         ii.append([tokenizer.decode(first_input[j][:-1]), a[j].replace('<|endoftext|>', '')])
    #     rps = ret_model.get_response(ii)
    #     k.extend([[x] for x in rps])

    #test_score += avg_prob
    sent_input = []

    for j in range(inputs_id.shape[0]*len(args.inter)):
        l = ll[j%inputs_id.shape[0]]
        sent_input.append([tokenizer.decode(inputs_id[j%inputs_id.shape[0]][:l]), decode_temp_sentence[j%inputs_id.shape[0]].replace('<|endoftext|>', ''), inter_response[j][0]])
    emo, embans = re_emo_score(detect_model, detect_processor, emotion_tokenizer, sent_input, len(inter_response))


    temp_score = []

#-----------------emotion-----------------------------

    for e in embans:
        temp_score.append(np.sum((e - emo_embed)**2))
       

    score = [0 for i in range(len(temp_score) // len(args.inter))]

    for j in range(len(temp_score) // len(args.inter)):
        for k in range(len(args.inter)):
            score[j] += temp_score[j + batch_size*k]
#----------------specific word-------------------------------------------
    score = np.array([0 for w in range(inputs_id.shape[0])])
    for j in range(inputs_id.shape[0]*len(args.inter)):
        for word in bad_dict.keys():
            if re.search(r"\b{}\b".format(word.lower()), k[j].lower().strip()):
                score[j%8] += 1

    score = np.array(score) / len(args.inter)
    score = score - np.mean(score)

    for j in range(inputs_id.shape[0]):
        loss -= (score[j]) * emotion_loss[j] #/ len(temp_sentence[j])
        loss += coherence_loss[j] * args.ra #/ len(temp_sentence[j])
    return loss, sum(temp_score), avg_prob
def main():
    parser = ArgumentParser()
    parser.add_argument("--emotion", type=str, default="angry")
    parser.add_argument("--writer", type=str, default="")
    parser.add_argument("--save", type=str, default="model/save/")
    parser.add_argument("--model", type=str, default='model/turn')
    parser.add_argument("--ra", type=float, default=3)
    parser.add_argument("--inter", type=str, default="gpt", nargs='+', required=True)
    args = parser.parse_args()

    os.makedirs('model/' + args.model, exist_ok=True)
    

    np.random.seed(100)
    torch.random.manual_seed(100)
    torch.cuda.manual_seed(100)
    model_train = GPT2LMHeadModel.from_pretrained(args.model)
    model_2 = GPT2LMHeadModel.from_pretrained(args.model)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


    if 'gpt' in args.inter:
        model_bot = GPT2LMHeadModel.from_pretrained('models/medium/')
        model_bot.to(device_1)
        model_bot.eval()
    #
    # if 'google' in args.inter:
    #     from main1 import chatbot
    #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1) 
    #     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #     jack = chatbot.Chatbot()
    #     jack.main(['--test', 'daemon', '--rootDir', 'deepqa', '--maxLength', '20'])
    # if 'retrieve' in args.inter:
    #     with torch.no_grad():
    #         from retrieval_model.retrieval_chatbot import Retrievalchatbot
    #         ret_model = Retrievalchatbot()
    writer = SummaryWriter('runs/'+args.writer+'/')
    param_optimizer = list(model_train.named_parameters())
    no_decay = ['bias', 'ln']   # no decay for bias and LayerNorm (ln)
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = Adam(optimizer_grouped_parameters, 5e-6,
                     max_grad_norm=1.0)

    model_train.to(device_0)
    model_2.to(device_1)
    model_2.eval()
    batch_size = 8

        


    
    post = post_set('data/train_raw.tsv', tokenizer)
    train_dataloader = DataLoader(post, batch_size=batch_size, shuffle=True, num_workers=2)
    batch = 0
    temp_score = 0
    loss = 0
   
    test_score = 0
    for global_step in range(1):
        model_train.train()
        for inputs_id, mask, ll in tqdm(train_dataloader):
            batch += 1
            batch_loss, score, avg_prob = train(model_train, inputs_id, mask, model_2, model_bot, tokenizer, ll, args, batch_size)
            loss += batch_loss

            test_score += avg_prob
            temp_score += score

            if batch % 4 == 0:
                loss.backward()
                optimizer.step()
                writer.add_scalar('loss', loss, batch)
                optimizer.zero_grad()  
                loss = 0
            if batch % 20 == 0:
                writer.add_scalar('reward', temp_score/batch_size/20, batch)
                writer.add_scalar('test_reward', test_score/20, batch)
                print("Reward:%.2f,    test:%.6f   "%(temp_score/batch_size/20/3, test_score/20))
                test_score = 0
                temp_score = 0
            if batch % 2500 == 0:
                torch.save(
                    {k: (v.cpu() if v is not None else None)  # save to cpu tensors
                        for k, v in model_train.state_dict().items()},
                    join(f'model/{args.save}/',
                            f'{args.save}-{batch}.pkl'))

if __name__ == "__main__":
    main()
