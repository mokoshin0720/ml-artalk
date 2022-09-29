import json
import pandas as pd
import os
import requests

def translate_text(text):
    API_KEY = os.getenv("API_KEY")

    url="https://translation.googleapis.com/language/translate/v2"
    url += "?key=" + API_KEY
    url += "&q=" + text
    url += "&source=en&target=ja"

    rr = requests.get(url)
    unit_aa = json.loads(rr.text)
    return unit_aa["data"]["translations"][0]["translatedText"]

if __name__ == '__main__':
    filename = 'artemis_mini.csv'
    df = pd.read_csv(filename)

    ja_sentences = []
    for index, row in df.iterrows():
        print(index, "文目: ", index / len(df), "%")
        sentence = translate_text(row['utterance'])
        ja_sentences.append(sentence)
    
    df['ja_utterance'] = ja_sentences

    df.to_csv('artemis_mini_translated.csv')
