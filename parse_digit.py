
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import click

@click.command()
@click.option('--mark', type=str, default=None, help='只显示名字包含mark的实验')
@click.option('--root', type=str, default='./saved-digit', help='要显示的实验路径')
@click.option('--select', type=str, default=None, help='指定第几个tgtdomain')
def show(mark, root, select):

    name_list = os.listdir(root)
    # 过滤
    if mark is not None:
        name_list = [name for name in name_list if mark in name]
    name_list = sorted(name_list)

    rst = None
    for i, name in enumerate(name_list):
        logpath = os.path.join(root, name, 'test.log')
        if select is not None:    
            logpath_ = os.path.join(root, name, select)
            if os.path.exists(logpath_):
                logpath = logpath_
        
        if os.path.exists(logpath):
            df = pd.read_csv(logpath, index_col=0)
            if rst is None:
                rst = pd.DataFrame(columns=df.columns)
            rst.loc[name] = df.values[0]
    rst['mean'] = rst.values[:,1:].mean(1) # 第1列是 mnist，不参与计算
    #print(rst)

    df2 = avg_run(rst)
    print(df2)

def avg_run(df):
    ''' 同一组实验参数重复跑了n次，该函数负责把这个n平均掉
        stn_H_0.3_0.1_run0  --->  stn_H_0.3_0.1
    '''
    src_index = df.index
    new_index = [i.split('_run')[0] for i in src_index]
    obj_index = sorted(list(set(new_index)))

    columns = df.columns
    init = np.zeros([len(obj_index), len(columns)])

    df2 = pd.DataFrame(init, index=obj_index, columns=columns)
    obj_index_count = {}
    for index in new_index:
        obj_index_count.update({index:0})

    for srcidx, objidx in zip(src_index, new_index):
        src_row = df.loc[srcidx]
        df2.loc[objidx] += src_row
        obj_index_count[objidx] += 1
    
    count = []
    for objidx in obj_index:
        df2.loc[objidx] /= obj_index_count[objidx]
        count.append(obj_index_count[objidx])
    df2['N'] = count
    return df2

if __name__=='__main__':
    show()

