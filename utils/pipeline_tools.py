# coding: utf-8
import datetime



### Support functions
def print_out(text, time_flag = True):
    print_out.cur_time
    del_time =datetime.datetime.now() -  print_out.cur_time
    print_out.cur_time = datetime.datetime.now()
    if time_flag:
        print('{}: '.format(del_time.total_seconds()) +str(text))
    else:
        print(str(text))
print_out.cur_time  = datetime.datetime.now()

### Wrapper function
def data_wrapper(data_streams, **kwargs):
    # print_out('Data wrapper',False)
    frame = dict()
    for key in kwargs:
        frame[key] = kwargs[key]
    data_streams.append(frame)


def push_stream_data(data_streams,data,func):
    frame = dict()
    frame['DATA'] = data
    frame['FUNCTION'] = func
    data_wrapper(data_streams,**frame)


### Unwrapper function
def data_unwrapper(data_streams,frame_index = None):
    # print_out('Data unwrapper',False)
    if frame_index:
        return data_streams[frame_index]
    else:
        return data_streams[-1]


def get_stream_data(data_streams):
    frame = data_unwrapper(data_streams)
    return frame['DATA']


### Executor perfoms functions from pipe
def exetutor(data_streams,functions_pipe):
    for function in functions_pipe:
        # print_out(function)
        function(data_streams)


### Print pipe graph
def print_pipe(functions_pipe):
    for index, function in enumerate(functions_pipe):
        print('{} '.format(index) + str(function))
