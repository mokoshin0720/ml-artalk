import pandas as pd

def list_utterances_of_img(
    origin_df,
    painting,
):
    return origin_df[origin_df['painting'] == painting]['utterance'].values.tolist()

if __name__ == '__main__':
    filename = 'data/artemis_dataset.csv'
    df = pd.read_csv(filename)
    
    for idx, row in df.iterrows():
        filename = row['painting']
        print(list_utterances_of_img(df, filename))