# originalのcsvを読み込む
# pathが存在していればnew_csvに追加
import pandas as pd
import os

if __name__ == "__main__":
    original_df = pd.read_csv("data/artemis_train_dataset.csv")
    mask_path = 'data/image_info/train/mask/'
    
    result = []
    for idx, row in original_df.iterrows():
        if os.path.isfile(mask_path + str(idx) + '.txt'):
            pass
        else:
            continue
        
        d = {}
        d['id'] = idx
        d['art_style'] = row['art_style']
        d['painting'] = row['painting']
        d['emotion'] = row['emotion']
        d['utterance'] = row['utterance']
        d['repetition'] = row['repetition']
        result.append(d)
        
    result_df = pd.DataFrame(
        data=result,
    )
    
    result_df.to_csv('train.csv', index=False)