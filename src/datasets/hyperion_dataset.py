import torch
import pandas as pd
from transformers import AutoTokenizer
from sklearn import preprocessing

LABELS = [
            'anticipazione',
            'causa',
            'commento',
            'conferma',
            'considerazione',
            'contrapposizione',
            'deresponsabilizzazione',
            'descrizione',
            'dichiarazione di intenti',
            'generalizzazione',
            'giudizio',
            'giustificazione',
            'implicazione',
            'non risposta',
            'opinione',
            'possibilità',
            'prescrizione',
            'previsione',
            'proposta',
            'ridimensionamento',
            'sancire',
            'specificazione',
            'valutazione'
    ]

class HyperionDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer_name):
        #fill_null_features(df)
        df = filter_empty_labels(df)
        df = to_lower_case(df)
        uniform_labels(df)          
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) 
        self.encodings = tokenize(df, tokenizer).items()
        self.labels = encode_labels(df).tolist()    

    def __getitem__(self, idx):
        return self.encodings[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)   


# Dataset loading and preprocessing
def fill_null_features(df):
        for c in ['Domanda','Testo']:
            for i in range(0,len(df.index)):  
                if not df[c][i]:
                    j=i
                    while j>0: 
                        j-=1
                        if df[c][j]:
                            df[c][i] = df[c][j]
                            break

#Delete examples with empty label
def filter_empty_labels(df):
    filter = df["Repertorio"] != ""
    return df[filter]

#Convert to lower case
def to_lower_case(df):
    return df.applymap(str.lower)


#Lables uniformation uncased
def uniform_labels(df):
    df['Repertorio'].replace('implicazioni','implicazione', inplace=True)
    df['Repertorio'].replace('previsioni','previsione', inplace=True)

def tokenize(df, tokenizer):
    return tokenizer(
        df['Stralcio'].tolist(),
        #df['Domanda'].tolist(),
        max_length=512,
        add_special_tokens=True,
        return_attention_mask=True,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

def encode_labels(df):
    le = preprocessing.LabelEncoder()
    le.fit(LABELS)
    return le.transform(df['Repertorio'])

def decode_labels(encoded_labels):
    le = preprocessing.LabelEncoder()
    le.fit(LABELS)
    return le.inverse_transform(encoded_labels)

def train_val_split(df, val_perc=0.1):
    gb = df.groupby('Repertorio')
    train_list = []
    val_list = []

    for x in gb.groups:
        class_df = gb.get_group(x)

        # Validation set creation
        val = train.sample(frac=val_perc)
        train = pd.concat([train,val]).drop_duplicates(keep=False)

        #train_list.append(train.head(500))
        train_list.append(train)
        val_list.append(val)

    train_df = pd.concat(train_list)
    val_df = pd.concat(val_list)
    return HyperionDataset(train_df), HyperionDataset(val_df)