import spacy
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import nltk
import pickle
import os
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
nltk.download('averaged_perceptron_tagger')

def prepare_dataset(test_size):
    filename = 'data/brysbaer.xlsx'
    df = pd.read_excel(filename)

    concrete_word_list = extract_noun(df[df['Conc.M'] >= 3]['Word'].tolist())
    abstract_word_list = extract_noun(df[df['Conc.M'] < 3]['Word'].tolist())
    
    train_set = [concrete_word_list, abstract_word_list]

    x = np.stack([list(nlp(w))[0].vector for part in train_set for w in part])
    y = [label for label, part in enumerate(train_set) for _ in part]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    
    return x_train, x_test, y_train, y_test

def train(x_train, x_test, y_train, y_test):
    names = [
        "Logistic Regression", 
        "Nearest Neighbors", 
        "Linear SVM",
        "RBF SVM",
        "Polynomial SVM",
        "Sigmoid SVM", 
        "Decision Tree",
        "Random Forest",
        "Naive Bayes",
    ]
    
    classifiers = [
        LogisticRegression(),
        KNeighborsClassifier(),
        SVC(kernel="linear", probability=True),
        SVC(kernel="rbf", probability=True),
        SVC(kernel="poly", probability=True),
        SVC(kernel="sigmoid", probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        GaussianNB(),
    ]
    
    result = []
    
    for name, clf in zip(names, classifiers):
        print('train {} ...'.format(name))
        clf.fit(x_train, y_train)
        train_score = clf.score(x_train, y_train)
        test_score = clf.score(x_test, y_test)
        
        if not os.path.exists('models/abstract_classifier'):
            os.makedirs('models/abstract_classifier')
        filename = 'models/abstract_classifier/{}.sav'.format(name)
        pickle.dump(clf, open(filename, 'wb'))
        
        result.append([train_score, test_score])
        
    df_result = pd.DataFrame(result, columns=['train', 'test'], index=names).sort_values('test', ascending=False)
    
    print(df_result)
    
    plt.rcParams['figure.subplot.bottom'] = 0.2
    plt.figure()
    df_result.plot(kind='bar', alpha=0.5, grid=True, figsize=(10, 10))
    plt.savefig('classifiers.png')
    plt.close('all')
    
    best_name = str(df_result.head(1).index[0])
    best_filename = 'models/abstract_classifier/{}.sav'.format(best_name)
    best_classifier = pickle.load(open(best_filename, 'rb'))
    
    return best_classifier
    
def extract_noun(word_list):
    result = []
    
    for word in word_list:
        word_pos = nltk.pos_tag([str(word)])
        if word_pos[0][1] in ['NN', 'NNS', 'NNP', 'NNPS', 'PP', 'PRP$']: result.append(word_pos[0][0])
    
    return result

def classify(classifier, token):
    classes = ['concrete', 'abstract']

    return classes[classifier.predict([token.vector])[0]]

def plot_graphes(classifier, x_test, y_test):
    plt.rcParams['font.family'] = 'Noto Sans JP'
    
    pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test, pred)
    cm = pd.DataFrame(
        data=cm,
        index=['具体名詞', '抽象名詞'],
        columns=['具体名詞', '抽象名詞']
    )
    
    sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='Blues')
    plt.yticks(rotation=0)
    plt.xlabel("Actual", fontweight='bold')
    plt.ylabel("Predict", fontweight='bold')
    plt.savefig('confusion_matrix.png')
    
    prob = classifier.predict_proba(x_test)[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, prob)
    print("閾値の数: {}".format(len(threshold)))
    
    auc = metrics.auc(fpr, tpr)
    print('auc:', auc)
    plt.figure(figsize = (5, 5))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('FPR: False Positive Rate', fontsize = 13)
    plt.ylabel('TPR: True Positive Rate', fontsize = 13)
    plt.grid()
    plt.show()
    plt.savefig('auc.png')
    
if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")
    
    x_train, x_test, y_train, y_test = prepare_dataset(test_size=0.2)    
    
    print('Dataset prepared...')
    classifier =  train(x_train, x_test, y_train, y_test)
    plot_graphes(classifier, x_test, y_test)
    
    
    # for token in nlp("Have a seat in that chair with comfort and drink some juice to soothe your thirst."):
    #     if token.pos_ == 'NOUN':
    #         result = classify(classifier, token)
    #         print(token, result)
    
    
    
    # result
    #                         train      test
    # RBF SVM              0.885744  0.836527
    # Linear SVM           0.835384  0.822113
    # Logistic Regression  0.833099  0.819652
    # Nearest Neighbors    0.851072  0.802953
    # Random Forest        0.951090  0.802953
    # Polynomial SVM       0.846019  0.771840
    # Sigmoid SVM          0.733960  0.733169
    # Decision Tree        0.951090  0.728072
    # Naive Bayes          0.669626  0.677448
    # 閾値の数: 1215
    # auc: 0.9062162082955416