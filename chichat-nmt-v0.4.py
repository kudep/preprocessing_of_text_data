# coding: utf-8
import os
import pandas as pd
from os import path
from utils.pandas_utils_for_sber_chitchat import tokinazing_column
from utils.pandas_utils_for_sber_chitchat import *
from utils.pipeline_tools import *


## In this pipeline be added


### Pipe of functions
functions_pipe=list()

### Start function
def empty_function(data_streams):
    print_out('', False)
    print_out('Start pipe')
pass

functions_pipe.append(empty_function)

## Formating and tokinazing

###Cats columns: df => df  input columns range "CHAT_ID" to "TYPE"
def cats_colunms(data_streams):
     data = get_stream_data(data_streams)
     #Perfom function task
     print_out('Cats columns: df => df  input columns range "CHAT_ID" to "TYPE"')
     data = data.loc[:, "CHAT_ID":"TYPE"]
     #Push data into streams
     push_stream_data(data_streams,data,cats_colunms)

functions_pipe.append(cats_colunms)

###Tokinazing: df => df  input column name "TEXT"
def tokinazing(data_streams):
    data = get_stream_data(data_streams)
    #Perfom function task
    print_out('Tokinazing: df => df  input column name "TEXT"')
    data = tokinazing_column(data, "TEXT")
    #Push data into streams
    push_stream_data(data_streams,data,tokinazing)

functions_pipe.append(tokinazing)


###Split data dialogues to list: df => list  input column name "CHAT_ID"
def split_to_list(data_streams):
    data = get_stream_data(data_streams)
    #Perfom function task
    print_out('Split data dialogues to list: df => list  input column name "CHAT_ID"')
    data = get_list_column_name(data,'CHAT_ID')
    #Push data into streams
    push_stream_data(data_streams,data,split_to_list)

functions_pipe.append(split_to_list)

# res_chats = chats # insert_pauses(chats)

###Insert pause in chat begin: list => list
def pauses_insert_in_begin(data_streams):
    data = get_stream_data(data_streams)
    #Perfom function task
    print_out('Insert pause in chat begin: list => list')
    data = insert_start_pauses(data)
    #Push data into streams
    push_stream_data(data_streams,data,pauses_insert_in_begin)

functions_pipe.append(pauses_insert_in_begin)


###Delete duplicates: list => list
def dublicate_delete(data_streams):
    data = get_stream_data(data_streams)
    #Perfom function task
    print_out('Delete duplicates: list => list')
    data, _ = delete_dublicates(data)
    #Push data into streams
    push_stream_data(data_streams,data,dublicate_delete)

functions_pipe.append(dublicate_delete)


###Sentence concatinate: list => list
def sentence_concat(data_streams):
    data = get_stream_data(data_streams)
    #Perfom function task
    print_out('Sentence concatinate: list => list')
    data = concatenate_in_sentence(data)
    #Push data into streams
    push_stream_data(data_streams,data,sentence_concat)

functions_pipe.append(sentence_concat)


### corp_ending_delete: list => list
def corp_ending_delete(data_streams):
    data = get_stream_data(data_streams)
    #Perfom function task
    print_out('corp_ending_delete: list => list')
    data = delete_corp_endings(data)
    #Push data into streams
    push_stream_data(data_streams,data,corp_ending_delete)

functions_pipe.append(corp_ending_delete)


### small_chat_delete: list => list
def small_chat_delete(data_streams):
    data = get_stream_data(data_streams)
    #Perfom function task
    print_out('small_chat_delete: list => list')
    data = delete_small_chat(data,0)
    #Push data into streams
    push_stream_data(data_streams,data,small_chat_delete)

functions_pipe.append(small_chat_delete)


# 130 длина последовательности при которой сохраняется более 0,9% диалогов менеджера


# ### add_start_and_stop_tags: list => list
# def add_start_and_stop_tags(data_streams):
#     data = get_stream_data(data_streams)
#     #Perfom function task
#     print_out('add_start_and_stop_tags: list => list')
#     data = add_start_stop_tags(data)
#     #Push data into streams
#     push_stream_data(data_streams,data,add_start_and_stop_tags)
#
# functions_pipe.append(add_start_and_stop_tags)
#
#
# functions_pipe.append(sentence_concat)


# ### merg_chats: list => list
# def merg_chats(data_streams):
#     data = get_stream_data(data_streams)
#     #Perfom function task
#     print_out('merg_chats: list => list')
#     data = merg_chats_with_window(data,2)
#     #Push data into streams
#     push_stream_data(data_streams,data,merg_chats)
#
# functions_pipe.append(merg_chats)


### add_serv_tag: list => list
def add_serv_tag(data_streams):
    data = get_stream_data(data_streams)
    #Perfom function task
    print_out('add_serv_tag: list => list')
    data = add_service_tag(data)
    #Push data into streams
    push_stream_data(data_streams,data,add_serv_tag)

functions_pipe.append(add_serv_tag)


functions_pipe.append(sentence_concat)


### add_index: list,list => list  input man_replic_max
def add_index(data_streams):
    assert (data_streams[len(data_streams)-3]['FUNCTION'] is small_chat_delete)
    assert (data_streams[len(data_streams)-1]['FUNCTION'] is sentence_concat)
    man_data = data_streams[len(data_streams)-3]['DATA']
    context_data = data_streams[len(data_streams)-1]['DATA']
    #Perfom function task
    print_out('add_index: list,list => list')
    data = add_indexing(context_data,man_data,context_len = 15,man_replic_max = 2, drop_begin_man_replic = 0)
    #Push data into streams
    push_stream_data(data_streams,data,add_index)

functions_pipe.append(add_index)



print_pipe(functions_pipe)


### Initialisation

dataset_dir = '/home/kuznetsov/qa_dialog/preprocessing/data'

# tiny_sber_data_csv = path.join(dataset_dir,'dial_tiny.csv')
large_sber_data_xlsx = path.join(dataset_dir,'tech_suply.xlsx')
input_data=pd.read_excel(large_sber_data_xlsx)

####Data download
# input_data=pd.read_csv(tiny_sber_data_csv)

### Data streams
data_streams = list()

push_stream_data(data_streams,input_data,empty_function)

exetutor(data_streams,functions_pipe)


# data_streams[10]['DATA'][0]
# data_streams[11]['DATA'][0]
# data_streams[11]['DATA'][0]


def zipper(context_chats, ans_chats):
    return [(con,ans) for con,ans in zip(context_chats,ans_chats)]


# In[29]:

assert (data_streams[len(data_streams)-1]['FUNCTION'] is add_index)
assert (data_streams[len(data_streams)-2]['FUNCTION'] is sentence_concat)
full_chats = zipper(data_streams[len(data_streams)-2]['DATA'], data_streams[len(data_streams)-1]['DATA'])


save_dir = path.join(dataset_dir,'nmt')
os.mkdir(save_dir)
cor_train = path.join(save_dir,'train.cor')
man_train = path.join(save_dir,'train.man')
cor_test = path.join(save_dir,'test.cor')
man_test= path.join(save_dir,'test.man')
cor_dev_test = path.join(save_dir,'dev_test.cor')
man_dev_test = path.join(save_dir,'dev_test.man')
cor_all = path.join(save_dir,'used_text.cor')
man_all = path.join(save_dir,'used_text.man')
man_voc = path.join(save_dir,'vocab.man')
cor_voc = path.join(save_dir,'vocab.cor')
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

cor_vocab = mkvocab(cor_all)
man_vocab = mkvocab(man_all)
save_vocab(man_vocab, man_voc)
save_vocab(cor_vocab, cor_voc)
