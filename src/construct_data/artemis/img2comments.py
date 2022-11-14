import pandas as pd

def get_comments_of_img(
    origin_df,
    filename,
):
    return origin_df[origin_df['painting'] == filename]['utterance'].values.tolist()

if __name__ == '__main__':
    filename = 'data/artemis_dataset.csv'
    df = pd.read_csv(filename)
    
    for idx, row in df.iterrows():
        filename = row['painting']
        print(get_comments_of_img(df, filename))