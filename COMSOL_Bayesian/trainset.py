import glob
import pandas as pd
import os
import torch

def trainX(model):
    df = pd.read_table(model, encoding='cp1252', header=None)
    # para = df[0].str.contains('Parameterkombination')
    para = df[0][0].split('{')
    para = [i.replace("}", "") for i in para]
    para = [j.replace("'", "") for j in para]
    para = [j.replace(" ", "") for j in para]
    para_ = para[1].split(',')
    parameter = {
        k:float(v)
        for k,v in(
            item.split(':',1) for item in para_
        )
    }
    df_para = pd.DataFrame([parameter])

    # print('head : ',df_para.columns.values.tolist())
    m1_w = df_para['M1_W'].item()
    m1_h = df_para['M1_H'].item()
    m2_h = df_para['M2_H'].item()
    m2_w = df_para['M2_W'].item()
    offset = df_para['offset'].item()
    return [m1_h, m1_w, m2_h, m2_w, offset]

def trainY(model):
    df = pd.read_table(model, encoding="cp1252", sep=r"\s+",
    comment="#",
    header=None,    
    names=["Zeit", "Fx"])
    return df['Fx'].mean()

def train_set(DataFolder):
    exp_files = glob.glob(os.path.join(DataFolder, "*.txt"))
    train_x_list = []
    train_y_list = []
    train_con_list = []

    for files in exp_files:
        # print(files)
        x = trainX(files)
        y = trainY(files)
    
        train_x_list.append(x)
        train_y_list.append(y)
   
    
    train_x = torch.tensor(train_x_list, dtype=torch.float64)
    train_y = torch.tensor(train_y_list, dtype=torch.float64)
    train_con = torch.tensor(train_con_list,dtype = torch.float64)
    # print(train_x)
    return train_x, train_y
