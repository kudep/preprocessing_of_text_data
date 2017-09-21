import pandas as pd
from os import path
import re
import progressbar
import datetime
import numpy as np
import random

def regex_pipe(column, pattern_from,pattern_to):
    bar = progressbar.ProgressBar()
    bar.init()
    for ptrn_from, ptrn_to in bar(zip(pattern_from,pattern_to)):
        column=column.apply(lambda x: re.sub(ptrn_from, ptrn_to, x))
    return column

def tokinazing_column(df, column_name):
    pattern_from = [r'[\s+]',r'(\\n)',r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*",r"([\*\"\'\\\/\|\{\}\[\]\;\:\<\>\,\.\?\*\(\)])",' +']
    pattern_to = [r' ',r' ',r" \1 ",r" \1 ",r' ',]
    df[column_name] = regex_pipe(df[column_name], pattern_from,pattern_to)
    return df




# #### Spliting on dialogs and adding service tokens

# Get list of chats

def get_list_column_name(df, column_name):
    bar = progressbar.ProgressBar()
    chat_id_serial = df[column_name].value_counts()[df[column_name].value_counts()>1]
    chat_list = list()
    for ind in bar(chat_id_serial.index):
        chat_list.append(df[df[column_name]==ind].copy().reset_index(drop=True))
    return chat_list


# unsupport


def merg_chats(chats, info=True):
    if info:
        bar = progressbar.ProgressBar()
        iner_bar = bar(chats)
    df=pd.DataFrame()
    if info:
        for chat in bar(chats):
            df = pd.concat([df,chat])
    else:
        for chat in chats:
            df = pd.concat([df,chat])
    return df.reset_index(drop=True)



def merg_chats_with_window(chats,wind_len=1):
    bar = progressbar.ProgressBar()
    res_chats= list()
    chat_len = len(chats)
    for ind in bar(range(chat_len)):
        merg_df = merg_chats(chats[ind:wind_len+ind]+chats[:max(ind-chat_len+wind_len,0)], info=False)
        res_chats.append(merg_df)
    return res_chats


# Insert pauses between dialogs with long delay



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


def add_start_stop_tags(chats, start_tag_enable = None):
    bar = progressbar.ProgressBar()
    bar.init()
    res_chats= list()
    if start_tag_enable:
        chat_start_tag = " <CHAT_START> "
    else:
        chat_start_tag = " "
    chat_stop_tag = " <CHAT_STOP> "
    man_speaker = 'MANAGER'

    start_row = pd.DataFrame(data=[[None, None, "MANAGER", chat_start_tag, None]], columns=chats[0].columns.values)
    stop_row = pd.DataFrame(data=[[None, None, "MANAGER", chat_stop_tag, None]], columns=chats[0].columns.values)
    for chat in bar(chats):
        one_df =  chat
        for ind in one_df.index.values.tolist():
            if (one_df["SPEAKER"][ind]==man_speaker):
                one_df = pd.concat([ one_df.loc[:ind-1,:], start_row, one_df.loc[ind:,:]])
                break
        one_df = pd.concat([one_df,stop_row])
        res_chats.append(one_df.reset_index(drop=True))
    return res_chats

# In[15]:


def delete_one_recplic_chat(chats, dial_filter=0):
    return delete_small_chat(chats, dial_filter)

def delete_small_chat(chats, dial_filter=0):
    bar = progressbar.ProgressBar()
    bar.init()
    res_chats= list()
    for chat in bar(chats):
        if len(chat)>dial_filter:
            res_chats.append(chat)
    return res_chats


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

def add_indexing(context_chats,answer_chats,context_len = 10,man_replic_max = 10, drop_begin_man_replic = 0):
    bar = progressbar.ProgressBar()
    bar.init()
    res_answer_chats = list()
    for context_chat, answer_chat in bar(zip(context_chats,answer_chats)):
        context_dial, answer_dial = context_chat,answer_chat[answer_chat['SPEAKER']=='MANAGER']
        lens = context_dial['TEXT'].map(lambda x: len([item for item in x.split() if item])).tolist()
        context_ids = dict()
        context_len = 250

        for answer_ind, row in answer_dial.iterrows():
            current_context_len = 0
            context_ids[answer_ind]=list()
            drop_begin_man_replic_count = drop_begin_man_replic
            man_replic_count = man_replic_max
            for context_ind in range(answer_ind-1,-1,-1):
                if (context_dial['SPEAKER'][context_ind]=="MANAGER"):
                    if drop_begin_man_replic_count > 0:
                        drop_begin_man_replic_count-=1
                        continue
                    if man_replic_count > 0:
                        man_replic_count-=1
                    else:
                        continue
                prev_len = current_context_len
                current_context_len+=lens[context_ind]
                if (current_context_len <= context_len):
                    context_ids[answer_ind].append(context_ind)
                else:
                    break
        answer_dial = answer_dial.join(pd.DataFrame([0 for i in range(answer_dial.index[-1]+1)],columns =['PADS']))
        answer_dial = answer_dial.join(pd.DataFrame([context_ids.get(i) for i in range(answer_dial.index[-1]+1)],columns =['CONTEXT_IDS']))
        res_answer_chats.append(answer_dial)
    return res_answer_chats

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




def mkvocab(inputfile):
    bar = progressbar.ProgressBar()
    bar.init()
    words = list()
    with open(inputfile) as f:
        for line in bar(f):
            words.extend(line.split())
    return set(words)


def save_vocab(vocab, filename):
    with open(filename, 'wt') as f:
        bar = progressbar.ProgressBar()
        bar.init()
        f.write("<unk>\n")
        f.write("<s>\n")
        f.write("</s>\n")
        for item in bar(vocab):
            f.write("%s\n" % item)
