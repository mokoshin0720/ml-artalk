import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    filename = 'data/artemis_dataset.csv'
    df = pd.read_csv(filename)
    
    abr = tqdm(total=len(df))
    for idx, row in df.iterrows():
        img = row['painting']
        comment_list = df[df['painting'] == img]['utterance']