import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset



class Dataset_UKDA(Dataset):
    def __init__(self, file_path, start, end):
        # dataset = get_dataset(file_path).reshape(-1,48)[3000:3100]
        dataset = get_dataset(file_path).reshape(-1,48)[start*365:end*365]
        self.dataset = torch.from_numpy(dataset).type(torch.float32).unsqueeze(-1)
        
    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.dataset.shape[0]



def get_dataset(file_path):
    data = pd.read_csv(file_path)
    aggregated_data = pd.DataFrame()
    aggregated_data["time"] = pd.to_datetime(data["time"])
    for i in range((data.shape[1]-2)//10):
        temp_data = data.iloc[:, (1+i*10):(1+(i+1)*10)].sum(axis=1)
        temp_name = "user_" + str(i+1)
        aggregated_data[temp_name] = temp_data

    aggregated_data["month"] = [aggregated_data["time"][i].month for i in range(aggregated_data.shape[0])]
    aggregated_data["day"] = [aggregated_data["time"][i].day for i in range(aggregated_data.shape[0])]
    aggregated_data["hour"] = [aggregated_data["time"][i].hour for i in range(aggregated_data.shape[0])]
    aggregated_data["minute"] = [aggregated_data["time"][i].minute for i in range(aggregated_data.shape[0])]

    dataset = aggregated_data.iloc[:, 1:-4].values.T.reshape(aggregated_data.shape[1]-5, -1, 48)
    return dataset