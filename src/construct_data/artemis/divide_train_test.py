from sklearn.model_selection import train_test_split
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('data/artemis_dataset.csv')

    train_df, test_df = train_test_split(data, train_size=0.9, test_size=0.1)
    train_df.to_csv('data/artemis_train_dataset.csv', index=False)
    test_df.to_csv('data/artemis_test_dataset.csv', index=False)
