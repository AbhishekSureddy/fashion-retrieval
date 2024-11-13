import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import requests

class UniqloDataset(Dataset):
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.df = self.df[self.df['is_good']]
        self.data = self.df.to_dict('records')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        temp = self.data[idx]
        temp["idx"] = idx
        temp['image'] = Image.open(requests.get(self.data[idx]['img_url'], stream=True).raw)
        return temp