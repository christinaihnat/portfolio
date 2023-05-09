import os, warnings, re
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns

# IMPORT DATA
val = pd.read_csv("data/emotion-labels-val.csv")
train = pd.read_csv("data/emotion-labels-train.csv")
test = pd.read_csv("data/emotion-labels-test.csv")

# DATASET EXPLORATION
### Let's look at each dataset to see if they can be combined
print("Columns:")
print("val: ", val.columns)
print("train: ", train.columns)
print("test: ", test.columns)

print("\nData Types")
print("val: \n", val.dtypes)
print("\ntrain: \n", train.dtypes)
print("\ntest: \n", test.dtypes)

### Let's combined the datasets. They will be splits later for the model.
df = pd.concat((val,train,test), sort=False)

# DATASET DESCRIPTION
### Checking datatypes
df.dtypes

### Checking shape
print("Rows:", df.shape[0])
print("\nFeatures:")
print(df.columns.tolist())

### Checking for missing values
print("\nMissing values:", df.isnull().sum().values.sum())

### Checking for duplicates
print("\nDuplicates:", len(df)-len(df.drop_duplicates()))

### Checking the number of unique values
print("\nUnique values:")
print(df.nunique())

# DATA EXPLORATION
### Let's look at the target column.
sns.countplot(train.label)

### Let's create a wordcloud for each emotion to help see the difference between them.
from wordcloud import WordCloud
import matplotlib.pyplot as plt

labels = list(df.label.unique())

for emo in labels:
    plt.figure(figsize=(12,8))
    wc = WordCloud(width=600,height=300).generate(' '.join(df[df.label==emo].text))
    plt.imshow(wc)
    plt.title(emo.title(),pad=20)
    plt.show()
    print('\n\n')

### Let's check out the len of the tweet.
df_mod = df
df_mod["text_len"] = df_mod.text.apply(len)

fig = sns.FacetGrid(df_mod, col = "label")
fig.map(sns.distplot, "text_len")
fig.add_legend()

### Let's explore the word count of the tweet.
df_mod["word_count"] = df_mod["text"].str.split().str.len()
df_mod

fig = sns.FacetGrid(df_mod, col = "label")
fig.map(sns.distplot, "word_count")
fig.add_legend()

### Looking at the most frequency word/symbol (could be hashtag or emoji)
import nltk
from nltk.probability import FreqDist

all_words = ' '.join([word for word in df_mod['text'].values])
tokenized_words = nltk.tokenize.word_tokenize(all_words.lower())

fdist = FreqDist(tokenized_words)
top_10 = fdist.most_common(10)
fdist = pd.Series(dict(top_10))
sns.barplot(y=fdist.index, x=fdist.values, color='blue');

### Let's look at the number of capital letters. 
### Hypothesis: capital letters help express emotions through text. 
### The number of capital letters is significant to determine the emotion.

def count_cap(text):
    return sum(1 for c in text if c.isupper())

df_mod['capital_count'] = df.text.apply(count_cap)

fig = sns.FacetGrid(df_mod, col = "label")
fig.map(sns.distplot, "capital_count")
fig.add_legend()

### descriptive statistics
df_mod.describe().T

# PREPROCESSING
### Let's stem and lemmatize the words.
from nltk.stem import SnowballStemmer, WordNetLemmatizer

stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def stem_words(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

def lemmatize_words(text):
    return ' '.join([lemmatizer.lemmatize(token) for token in text.split()])

### Let's change the emojis to text versions.
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted

class Emojifier(TransformerMixin, BaseEstimator):
    """
    Converts characters like :) :( :/ to a unique value
    """
    def __init__(self, emoji_pattern=r'[:;Xx][)(\/D]|[)(\/D][:;]'):
        self.emoji_pattern = emoji_pattern

    def fit(self, X, y=None):
        emoji_list = set()
        pattern = re.compile(self.emoji_pattern)

        for line in X:
            emoji_list.update(pattern.findall(line))


        self.found_emojis_ = {}
        for i, emoji in enumerate(emoji_list):
            self.found_emojis_[emoji] = '<EMOJI_%d>' % i

        return self

    def transform(self, X):
        # Validate
        check_is_fitted(self, ['found_emojis_'])

        # Transform
        data = pd.Series(X)
        for emoji, name in self.found_emojis_.items():
            data = data.str.replace(emoji, name, regex=False)

        return data.values

emojifier = Emojifier()
emojifier.fit(df_mod['text'])
print(emojifier.found_emojis_)

df_mod['text'] = emojifier.transform(df_mod['text'])
df_mod['text'][:20]

### processing the tweets. 
#### Steps:
#- Replace numbers with abbreviations (i.e. 1,000 to 1k)
#- Standard numbers (i.e. 1000 and 1,000 as 1k)
#- Change contractions to keep it consist
#- Change all lettets to lower case
#- Remove stop words
#- Stem words
#- Lemmatize words
#- Remove double white space
#- Remove trailing white space
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

def text_preprocess(text):
    text = str(text).lower()
    text = text.replace(',000,000,000 ', 'b ')
    text = text.replace(',000,000 ', 'm ')
    text = text.replace(',000 ', 'k ')
    text = re.sub(r'([0-9]+)000000000', r'\1b', text)
    text = re.sub(r'([0-9]+)000000', r'\1m', text)
    text = re.sub(r'([0-9]+)000', r'\1k', text)
    
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    text_decontracted = []
    
    for word in text.split():
        if word in contractions:
            word = contractions[word]
        text_decontracted.append(word)
    
    text_decontracted = ' '.join(text_decontracted)
    text = ' '.join([word for word in text_decontracted.split() if word.lower() not in stopwords.words('english') and not word.isdigit()])
    
    text = text.replace("'ve", " have")
    text = text.replace("n't", " not")
    text = text.replace("'re", " are")
    text = text.replace("'ll", " will")

    # Stemming 
    text = stem_words(text)
    
    # Lemmatization
    text = lemmatize_words(text)

    # Removing HTML tags
    text = BeautifulSoup(text)
    text = text.get_text()
    
    # Removes extra white space
    text = re.sub(r'\s+', ' ', text)
    # Remove trailing white space
    text = re.sub(r'$\s', '', text)
    
    return text

### Apply text process
df_mod['text'] = df_mod.text.apply(text_preprocess)
df_mod

# BASELINE MODEL
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report

target = "label"
cv = CountVectorizer()
x = cv.fit_transform(df_mod.text)
y = df_mod.label

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state=123)

algo = ['BernoulliNB', BernoulliNB(),
       'KNeighborsClassifier', KNeighborsClassifier(),
       'SVC', SVC(),
       'DecisionTreeClassifier', DecisionTreeClassifier(),
       'RandomForestClassifier', RandomForestClassifier()]

name = []
accuracy_scored=[]

for i in range(0, len(algo), 2):
    model = algo[i+1]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    name.append(algo[i])
    accuracy_scored.append(accuracy_score(y_test, y_pred))
    
result=pd.DataFrame(columns=['name', 'accuracy_score'])
result['name'] = name
result['accuracy_score']=accuracy_scored

result.sort_values('accuracy_score',ascending=False)

labels = list(df.label.unique())

R=RandomForestClassifier(random_state=123)
R.fit(x_train, y_train)
y_pred = R.predict(x_test)
print(f"accuracy : {accuracy_score(y_test, y_pred)*100}")
print(classification_report(y_test,y_pred,target_names=labels))

cm_array = confusion_matrix(y_test, y_pred)
cm_array_df = pd.DataFrame(cm_array/np.sum(cm_array), index=labels, columns=labels)
sns.heatmap(cm_array_df, annot=True, annot_kws={"size": 12}, fmt='.2%', cmap='Blues') 

# MODEL IMPROVEMENT BY ADDING FEATURES
from scipy.sparse import hstack, csr_matrix

target = "label"
cv = CountVectorizer()
x = cv.fit_transform(df_mod.text)
y = df_mod.label

features= ['text_len', 'word_count', 'capital_count']
x = hstack((x, csr_matrix(df_mod[features].values)))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state=123)

R=RandomForestClassifier(random_state=123)
R.fit(x_train, y_train)
y_pred = R.predict(x_test)
print(f"accuracy : {accuracy_score(y_test, y_pred)*100}")
print(classification_report(y_test,y_pred,target_names=labels))

cm_array = confusion_matrix(y_test, y_pred)
cm_array_df = pd.DataFrame(cm_array/np.sum(cm_array), index=labels, columns=labels)
sns.heatmap(cm_array_df, annot=True, annot_kws={"size": 12}, fmt='.2%', cmap='Blues') 

# BOOSTING WITH SENTIMENT ANALYSIS
from nltk.sentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
df_sent = df_mod
df_sent['polarity'] = df_sent['text'].apply(lambda x: analyzer.polarity_scores(x))
sentiment = df_sent['polarity'].apply(pd.Series)
df_sent = pd.concat([df_sent, sentiment], axis=1)
df_sent = df_sent.drop(columns = 'polarity', axis=1)
df_sent

df_sent.describe().T

target = "label"
cv = CountVectorizer()
x = cv.fit_transform(df_mod.text)
y = df_mod.label

features= ['neg', 'neu', 'pos', 'compound']
x = hstack((x, csr_matrix(df_sent[features].values)))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state=123)

R=RandomForestClassifier(random_state=123)
R.fit(x_train, y_train)
y_pred = R.predict(x_test)
print(f"accuracy : {accuracy_score(y_test, y_pred)*100}")
print(classification_report(y_test,y_pred,target_names=labels))

cm_array = confusion_matrix(y_test, y_pred)
cm_array_df = pd.DataFrame(cm_array/np.sum(cm_array), index=labels, columns=labels)
sns.heatmap(cm_array_df, annot=True, annot_kws={"size": 12}, fmt='.2%', cmap='Blues') 

# PARAMETER TUNING
labels = list(df.label.unique())

target = "label"
cv = CountVectorizer()
x = cv.fit_transform(df_mod.text)
y = df_mod.label

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state=123)

R=RandomForestClassifier(random_state=123)
R.fit(x_train, y_train)
y_pred = R.predict(x_test)
print(f"accuracy : {accuracy_score(y_test, y_pred)*100}")
print(classification_report(y_test,y_pred,target_names=labels))

cm_array = confusion_matrix(y_test, y_pred)
cm_array_df = pd.DataFrame(cm_array/np.sum(cm_array), index=labels, columns=labels)
sns.heatmap(cm_array_df, annot=True, annot_kws={"size": 12}, fmt='.2%', cmap='Blues') 

print(R.set_params())

from sklearn.model_selection import GridSearchCV

params = [{
    'random_state': [123],
    'n_estimators': [10, 20, 30, 60, 100],
    'criterion': ["gini", "entropy"],
    'min_samples_split': [0.1, 0.3, 0.5, 0.99], 
    'min_samples_leaf': [1, 2, 3, 5],
    'max_features': ["sqrt", "log2", None],
    'oob_score': [True, False],
    'warm_start': [True, False],
    'class_weight': ["balanced", "balanced_subsample", None]
}]

gs = GridSearchCV(RandomForestClassifier(), params, cv=3, scoring="accuracy")
gs.fit(x_train, y_train)

print('Best score:', gs.best_score_)
print('Best params:', gs.best_params_)

labels = list(df.label.unique())

target = "label"
cv = CountVectorizer()
x = cv.fit_transform(df_mod.text)
y = df_mod.label

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state=123)

R=RandomForestClassifier(class_weight = 'balanced', criterion = 'gini', max_features = 'log2',
                         min_samples_leaf = 1, min_samples_split = 0.1, n_estimators = 100,
                         oob_score = True, warm_start =True, random_state=123)
R.fit(x_train, y_train)
y_pred = R.predict(x_test)
print(f"accuracy : {accuracy_score(y_test, y_pred)*100}")
print(classification_report(y_test,y_pred,target_names=labels))

cm_array = confusion_matrix(y_test, y_pred)
cm_array_df = pd.DataFrame(cm_array/np.sum(cm_array), index=labels, columns=labels)
sns.heatmap(cm_array_df, annot=True, annot_kws={"size": 12}, fmt='.2%', cmap='Blues') 

#### Our final model is 87.80% accurate with the average F1-score at 88%. This is an improvement from the baseline model. 