
# coding: utf-8

# In[1]:


import pandas as pd
from os import path
import re
import progressbar
import datetime
import numpy as np
import matplotlib.pyplot as plt
import random


# In[2]:


get_ipython().magic('matplotlib inline')
plt.figure()


# #### Initialisation

# In[3]:


dataset_dir = 'data'

tiny_sber_data_csv = path.join(dataset_dir,'Dialog1_ разметка_чатов_MK_v01.csv')
large_sber_data_xlsx = path.join(dataset_dir,'tech_suply.xlsx')


# In[4]:


input_data=pd.read_csv(tiny_sber_data_csv)


# In[4]:


input_data=pd.read_excel(large_sber_data_xlsx)


# #### Formating and tokinazing

# In[5]:


f_t_data = input_data.loc[:, "CHAT_ID":"TYPE"]


# In[6]:


def form_toc(df, ser_name):
    df[ser_name]=df[ser_name].apply(lambda x: re.sub(r'[\s+]', ' ', x))
    df[ser_name]=df[ser_name].apply(lambda x: re.sub(r'(\\n)', ' ', x))
    df[ser_name]=df[ser_name].apply(lambda x: re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", x))
    df[ser_name]=df[ser_name].apply(lambda x: re.sub(r"([\*\"\'\\\/\|\{\}\[\]\;\:\<\>\,\.\?\*\(\)])", r" \1 ", x))
    df[ser_name]=df[ser_name].apply(lambda x: re.sub(' +', ' ', x))


# In[7]:


form_toc(f_t_data, 'TEXT')


# #### Spliting on dialogs and adding service tokens

# Get list of chats

# In[8]:


def get_chat_id_list(df):
    bar = progressbar.ProgressBar()
    chat_id_serial = df['CHAT_ID'].value_counts()[df['CHAT_ID'].value_counts()>1]
    chat_list = list()
    for ind in bar(chat_id_serial.index):
        chat_list.append(df[df['CHAT_ID']==ind].copy().reset_index(drop=True))
    return chat_list


# In[9]:


def merg_chats(chats):
    bar = progressbar.ProgressBar()
    df=pd.DataFrame()
    for chat in bar(chats):
        df = pd.concat([df,chat])
    return df


# Insert pauses between dialogs with long delay

# In[10]:


def insert_pauses(chats, tag="<PAUSE>",minutes=2, seconds=0):
    bar = progressbar.ProgressBar()
    bar.init()
    res_chats= list()
    for chat in bar(chats):
        setect_speaker = 'MANAGER'
        one_df =  chat
        one_row = pd.DataFrame(data=[[None, None, "CORPORATE", tag, None]], columns=one_df.columns.values)
        time_pattern = "%d-%b-%y %H.%M.%S.%f %p"
        time_trashhold = datetime.timedelta(minutes=minutes, seconds=seconds)
        #print(one_row)
        time_fn = datetime.datetime.strptime
        for ind in one_df.index.values.tolist()[:-1]:
            cur_row =one_df.loc[ind,:]
            next_row =one_df.loc[ind+1,:]
            if (cur_row["SPEAKER"]==setect_speaker) and (next_row["SPEAKER"]==setect_speaker):
                delta_time = time_fn(next_row['TIMESTAMP'], time_pattern) - time_fn(cur_row['TIMESTAMP'], time_pattern)
                if delta_time > time_trashhold:
                    #print("Index = {}  and time {}".format(ind, delta_time))
                    one_df = pd.concat([one_df.loc[:ind,:], one_row, one_df.loc[ind+1:,:]])
        res_chats.append(one_df.reset_index(drop=True))
    return res_chats


# Insert pause in begin dialog, if manager have start dialog

# In[11]:


def insert_start_pauses(chats, tag=" <PAUSE> "):
    bar = progressbar.ProgressBar()
    bar.init()
    res_chats= list()
    for chat in bar(chats):
        setect_speaker = 'MANAGER'
        one_df =  chat
        one_row = pd.DataFrame(data=[[None, None, "CORPORATE", tag, None]], columns=one_df.columns.values)
        if one_df['SPEAKER'][0] == setect_speaker:
            one_df = pd.concat([one_row, one_df.loc[0:,:]])
        res_chats.append(one_df.reset_index(drop=True))
    return res_chats


# Delete repeates and if repeates count more then repeate_trash_hold drop that diologs

# In[12]:


def delete_dublicates(chats,repeate_trash_hold = 5):
    bar = progressbar.ProgressBar()
    bar.init()
    res_chats= list()
    bad_chats= list()
    repeate_trash_hold = 5
    for chat in bar(chats):
        setect_speaker = 'MANAGER'
        one_df =  chat
        repeate_count = 0
        for ind in one_df.index.values.tolist()[0:-1]:
            cur_row =one_df.loc[ind,:]
            next_row =one_df.loc[ind+1,:]
            if (cur_row["SPEAKER"]==next_row["SPEAKER"]) and (cur_row["TEXT"]==next_row["TEXT"]):
                repeate_count+=1
                if ind == 0:
                    one_df = pd.concat([one_df.loc[ind+1:,:]])
                else:
                    one_df = pd.concat([one_df.loc[:ind-1,:], one_df.loc[ind+1:,:]])
        if repeate_count < repeate_trash_hold:
            res_chats.append(one_df.reset_index(drop=True))
        else:
            bad_chats.append(one_df.reset_index(drop=True))
    return res_chats, bad_chats


# Concatenate near dialogs by one author

# In[13]:


def concatenate_in_sentence(chats):
    bar = progressbar.ProgressBar()
    bar.init()
    res_chats= list()
    for chat in bar(chats):
        prev_speak = ''
        remove_list = list()
        for index, row in chat.iterrows():
            if prev_speak == row['SPEAKER']:
                chat.loc[index,'TEXT'] = chat.loc[index-1,'TEXT'] + ' ' + chat.loc[index,'TEXT']
                remove_list.append(index-1)
            prev_speak = row['SPEAKER']
        chat=chat.drop(chat.index[remove_list]).reset_index(drop=True)
        res_chats.append(chat.reset_index(drop=True))
    return res_chats


# If end dialog is sended by corporate then delete it

# In[14]:


def delete_corp_endings(chats):
    bar = progressbar.ProgressBar()
    bar.init()
    res_chats= list()
    for chat in bar(chats):
        setect_speaker = 'CORPORATE'
        one_df =  chat
        if one_df['SPEAKER'].values[-1] == setect_speaker:
            one_df = one_df.iloc[:-1,:]
        res_chats.append(one_df)
    return res_chats


# In[15]:


def delete_one_recplic_chat(chats, dial_filter=0):
    bar = progressbar.ProgressBar()
    bar.init()
    res_chats= list()
    for chat in bar(chats):
        if len(chat)>dial_filter:
            res_chats.append(chat)
    return res_chats


# In[16]:


chats=get_chat_id_list(f_t_data)


# In[17]:


res_chats = chats # insert_pauses(chats)


# In[18]:


res_chats = insert_pauses(chats)


# In[18]:


res_chats1 = insert_start_pauses(res_chats)


# In[19]:


res_chats2, _ = delete_dublicates(res_chats1)


# In[20]:


res_chats3 = concatenate_in_sentence(res_chats2)


# In[21]:


res_chats4 = delete_corp_endings(res_chats3)


# In[22]:


res_chats5 = delete_one_recplic_chat(res_chats4)


# #### Statistic

# In[23]:


def df_select(df,speaker = None):
    if speaker:
        return df[df['SPEAKER']==speaker]
    return df


# In[24]:


def get_distib_tokens(chats,speaker = None):
    bar = progressbar.ProgressBar()
    bar.init()
    chats_lens= list()
    all_lens= list()
    for chat in bar(chats):
        chat_lens= list()
        chat = df_select(chat,speaker)
        for index, row in chat.iterrows():
            text_len = len([item for item in row['TEXT'].split() if item])
            chat_lens.append(text_len)
            all_lens.append(text_len)
        chat_lens.append(chat_lens)
    return all_lens


# In[130]:


def word_disrt(sentence_lens, word_max = 500):
    sentence_count = len(sentence_lens)
    word_distr = list()
    lens_ser = pd.Series(sentence_lens)
    for it_index in range(0,word_max, 1):
        word_distr.append(len(lens_ser[lens_ser>it_index])/sentence_count)
    
    return word_distr


# In[65]:


all_lens = get_distib_tokens(res_chats4)


# In[63]:


pd.Series(all_lens).plot()


# In[145]:


distr = word_disrt(get_distib_tokens(res_chats4), 200)


# In[146]:


pd.Series(distr).plot()


# In[139]:


man_distr = word_disrt(get_distib_tokens(res_chats4, 'MANAGER'), 200)


# In[140]:


cor_distr = word_disrt(get_distib_tokens(res_chats4, 'CORPORATE'), 200)


# In[135]:


pd.Series(d3).plot()


# In[147]:


[(x,distr[x])for x in range(0,200,10) ]


# In[143]:


[(x,man_distr[x])for x in range(0,200,10) ]


# In[148]:


[(x,cor_distr[x])for x in range(0,200,10) ]


# 130 длина последовательности при которой сохраняется более 0,9% диалогов менеджера

# In[23]:


def add_service_tag(chats):
    bar = progressbar.ProgressBar()
    bar.init()
    res_chats= list()
    man_start_tag = " <MAN_START> "
    cor_start_tag = " <COR_START> "
    cor_speaker = 'CORPORATE'
    
    cor_row = pd.DataFrame(data=[[None, None, "CORPORATE", cor_start_tag, None]], columns=chats[0].columns.values)
    man_row = pd.DataFrame(data=[[None, None, "MANAGER", man_start_tag, None]], columns=chats[0].columns.values)
    for chat in bar(chats):
        one_df =  chat
        for ind in one_df.index.values.tolist():
            if (one_df["SPEAKER"][ind]==cor_speaker):
                one_df = pd.concat([one_df.loc[:ind-1,:], cor_row,one_df.loc[ind:,:]])
            else:
                one_df = pd.concat([ one_df.loc[:ind-1,:],man_row, one_df.loc[ind:,:]])
        res_chats.append(one_df.reset_index(drop=True))
    return res_chats


# In[24]:


res_chats6 = add_service_tag(res_chats5)


# In[25]:


res_chats7 = concatenate_in_sentence(res_chats6)


# In[26]:


def add_padding_and_indexing(context_chats,answer_chats):
    bar = progressbar.ProgressBar()
    bar.init()
    res_answer_chats = list()
    for context_chat, answer_chat in bar(zip(context_chats,answer_chats)):
        context_dial, answer_dial = context_chat,answer_chat[answer_chat['SPEAKER']=='MANAGER']
        lens = context_dial['TEXT'].map(lambda x: len([item for item in x.split() if item])).tolist()
        context_ids = dict()
        pads = dict()
        context_len = 250

        for answer_ind, row in answer_dial.iterrows():
            current_context_len = 0
            context_ids[answer_ind]=list()
            pads[answer_ind]=0    
            for context_ind in range(answer_ind-1,-1,-1):
                prev_len = current_context_len
                current_context_len+=lens[context_ind]
                if (current_context_len <= context_len):
                    context_ids[answer_ind].append(context_ind)
                    pads[answer_ind] = context_len - current_context_len
                else:
                    pads[answer_ind] = context_len - prev_len
                    break
        answer_dial = answer_dial.join(pd.DataFrame([pads.get(i) for i in range(answer_dial.index[-1]+1)],columns =['PADS']))
        answer_dial = answer_dial.join(pd.DataFrame([context_ids.get(i) for i in range(answer_dial.index[-1]+1)],columns =['CONTEXT_IDS']))
        res_answer_chats.append(answer_dial)
    return res_answer_chats


# In[27]:


res_chats8 = add_padding_and_indexing(res_chats7,res_chats5)


# In[28]:


def zipper(context_chats, ans_chats):
    return [(con,ans) for con,ans in zip(context_chats,ans_chats)]


# In[29]:


full_chats = zipper(res_chats7, res_chats8)


# In[30]:


def dataset_make(zip_chats, train_file, label_file, pad_enable = True):
    bar = progressbar.ProgressBar()
    bar.init()
    pad_tonken = '<PAD> '
    with open(train_file, 'wt') as train:
        with open(label_file, 'wt') as label:
            for context_chat, answer_chat in bar(zip_chats):
                for _, row in answer_chat.iterrows():
                    label.write(re.sub(' +', ' ', row['TEXT'])+'\n')
                    train_line = ''
                    if pad_enable:
                        train_line+= pad_tonken*int(row['PADS'])
                    row['CONTEXT_IDS'].sort()
                    for index in row['CONTEXT_IDS']:
                        train_line+= context_chat['TEXT'][index]
                    train_line = re.sub(' +', ' ', train_line)
                    train.write(train_line + '\n') 
    


# In[31]:


def rand_split_list(source, first_portion):
    source_len=len(source)
    targ1_list=list()
    targ2_list=list()
    rand_indexs = random.sample(range(source_len), int(source_len*first_portion))
    for ind,item in enumerate(source):
        if ind in rand_indexs:
            targ1_list.append(item)
        else:
            targ2_list.append(item)
    return targ1_list, targ2_list


# In[32]:



dataset_dir = 'data/nmt'
cor_train = path.join(dataset_dir,'train.cor')
man_train = path.join(dataset_dir,'train.man')
cor_test = path.join(dataset_dir,'test.cor')
man_test= path.join(dataset_dir,'test.man')
cor_dev_test = path.join(dataset_dir,'dev_test.cor')
man_dev_test = path.join(dataset_dir,'dev_test.man')
cor_all = path.join(dataset_dir,'used_text.cor')
man_all = path.join(dataset_dir,'used_text.man')
man_voc = path.join(dataset_dir,'vocab.man')
cor_voc = path.join(dataset_dir,'vocab.cor')
part_of_all_data = 1
train_part_of_used_data = 0.98
dev_part_of_test = 0.5
pad_enable = False


# In[33]:


used_df, _ = rand_split_list(full_chats,part_of_all_data)
train_df, test_df = rand_split_list(used_df,train_part_of_used_data)
dev_test, test = rand_split_list(test_df,dev_part_of_test)
dataset_make(train_df,cor_train,man_train,pad_enable)
dataset_make(dev_test,cor_dev_test,man_dev_test,pad_enable)
dataset_make(test,cor_test,man_test,pad_enable)
dataset_make(used_df,cor_all,man_all,pad_enable)


# #### Создание словаря

# In[34]:


def mkvocab(inputfile):
    bar = progressbar.ProgressBar()
    bar.init()
    words = list()
    with open(inputfile) as f:
        for line in bar(f):
            words.extend(line.split())
    return set(words)


# In[35]:


def save_vocab(vocab, filename):
    with open(filename, 'wt') as f:
        bar = progressbar.ProgressBar()
        bar.init()
        f.write("<unk>\n")
        f.write("<s>\n")
        f.write("</s>\n")
        for item in bar(vocab):
            f.write("%s\n" % item)


# In[36]:


cor_vocab = mkvocab(cor_all)
man_vocab = mkvocab(man_all)
save_vocab(man_vocab, man_voc)
save_vocab(cor_vocab, cor_voc)

