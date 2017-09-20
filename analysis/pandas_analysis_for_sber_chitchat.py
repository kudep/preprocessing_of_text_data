# coding: utf-8
import pandas as pd
from os import path
import matplotlib.pyplot as plt
import random
from utils.pandas_utils_for_sber_chitchat import tokinazing_column
from utils.pandas_utils_for_sber_chitchat import *
from utils.pipeline_tools import *
get_ipython().magic('matplotlib inline')
plt.figure()

# #### Statistic


def df_select(df,speaker = None):
    if speaker:
        return df[df['SPEAKER']==speaker]
    return df


# In[24]:
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
