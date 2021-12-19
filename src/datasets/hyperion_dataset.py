import torch
import pandas as pd

class HyperionDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.df = pd.read_csv(filename, na_filter=False)

    def __getitem__(self, idx):
        return self.df[idx]

    def __len__(self):
        return len(self.df.index)

    def train_val_split(self, val_perc):
        return 0    #return train_set, val_set


    # Dataset loading and preprocessing
    def fill_null_features(self):
        for c in ['Domanda','Testo']:
            for i in range(0,len(self.df.index)):  
                if not self.df[c][i]:
                    j=i
                    while j>0: 
                        j-=1
                        if self.df[c][j]:
                            self.df[c][i] = self.df[c][j]
                            break

    #Delete examples with empty label
    def filter_empty_labels(self):
        filter = self.df["Repertorio"] != ""
        self.df = self.df[filter]

    #Convert to lower case
    def to_lower_case(self):
        self.df = self.df.applymap(str.lower)

    #Lables uniformation uncased
    def uniform_labels(self):
        self.df['Repertorio'].replace('implicazioni','implicazione', inplace=True)
        self.df['Repertorio'].replace('previsioni','previsione', inplace=True)




    