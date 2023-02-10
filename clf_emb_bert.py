import pandas as pd
import pickle
from transformers import AutoModel,AutoTokenizer
import numpy as np
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

### this script to get two dfs
## one is for  embeddings from last_hidden_state and 
## the other is get df for UMAP


df1 = pd.read_csv('./clf_emb_bert.csv')

model_ckpt = 'bert-base-chinese'
model = AutoModel.from_pretrained(model_ckpt)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

df = pd.DataFrame(columns=['clf', 'clf_vector', 'phrase'])

for phr,clf in dict(zip(df1['phrase'],df1['clf_form'])).items():
    tokenized_p = tokenizer.tokenize(phr)
    try:
        index_pos = tokenized_p.index(clf) + 1
    except ValueError:
        print(clf, phr)
        continue
    index_pos = tokenized_p.index(clf)+1
    inputs = tokenizer(phr, return_tensors='pt')
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    
    arr = last_hidden_states[0][index_pos].detach().numpy()
    
    df = df.append({'clf': clf, 
                    'clf_vector': arr, 
                    'phrase': phr}, 
                   ignore_index=True)
    
X = np.array(list(df['clf_vector']))
y = list(df['clf'])

# scale features to [0,1] range
X_scaled = MinMaxScaler().fit_transform(X)

# initialize and fit UMAP
mapper = UMAP(n_components=2, metric='cosine',random_state=2023).fit(X_scaled)

# create a dataframe of 2D embeddings
df_emb1 = pd.DataFrame(mapper.embedding_, columns=['X', 'Y'])
df_emb1['label'] = y

df_emb1.to_pickle('./clf_emb_bert_umap1.pkl')
df.to_pickle('./clf_emb_bert.pkl')

