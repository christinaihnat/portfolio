# IMPORT LIBRARIES
import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

### Exploration
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go

### Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

### Models
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

### Performance Model
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report

### Model Tuning
from sklearn.model_selection import GridSearchCV

# CHANGE WORKING DIRECTORY TO CURRENT
os.chdir(os.getcwd())

# IMPORT DATA
### Data is from Kaggle's "Telecom Churn Dataset"
### [Telecom Churn Dataset](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets/code?datasetId=255093&sortBy=voteCount)
def get_data():
    test = pd.read_csv("data/churn-bigml-20.csv")
    train = pd.read_csv("data/churn-bigml-80.csv")
    return train, test

# DATA DESCRIPTION
### Let's take a glance at the dataset. 
def dataset_description(df):
    print("Rows:", df.shape[0]) # number of rows
    print("\nNumber of features:", df.shape[1]) # numbers of features/columns
    print("\nFeatures:")
    print(df.columns.tolist()) # name of the fatures/columns
    print("\nMissing values:", df.isnull().sum().values.sum()) # number of missing values
    print("\nUnique values:") # number of unique values for each feature/column
    print(df.nunique())

### descriptive statistics
def get_descriptive_stats(df):
    print(df.describe().T)

### data types for each feature/column
def get_data_types(df):
    print(df.dtypes)

# EXPLORATION
### Let's explore each feature/column distribution and paired graphs
def graph_pairplot(df):
    out_cols = list(set(df.nunique()[df.nunique()<6].keys().tolist() + df.select_dtypes(include='object').columns.tolist()))
    viz_cols = [x for x in df.columns if x not in out_cols] + ['Churn']
    sns.pairplot(train[viz_cols], diag_kind="kde")
    plt.show()

### Explore correlation. Strong correlation can cause multi-colinearity. 
### Green = good (low correlation), Red = bad (high correlation)
def corr_graph(df):
    sns.set(style="white")
    mask = np.zeros_like(df.corr(), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(60,50))
    cmap = sns.diverging_palette(133, 10, as_cmap=True)  
    g = sns.heatmap(data=df.corr(), annot=True, cmap=cmap, ax=ax, mask=mask, annot_kws={"size":20},  cbar_kws={"shrink": 0.8} );
    bottom, top = ax.get_ylim() # prevent cut-off issues
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.tick_params(labelsize=25) 
    ax.set_yticklabels(g.get_yticklabels(), rotation=0);
    ax.set_xticklabels(g.get_xticklabels(), rotation=80);

### Explore the target (Churn) data. Checking for balance data. 
def explore_target(df, target):
    print("Percentage of not Churning: ", df[target].value_counts()[False]/len(train)*100)
    print("Percentage of Churning: ", df[target].value_counts()[True]/len(train)*100)
    num_false = df[target].value_counts()[False]
    num_true = df[target].value_counts()[True]
    pie_count = np.array([num_false, num_true])
    pie_label = [False, True]
    plt.pie(pie_count, labels=df.target.unique())
    plt.show() 
### Churn is unbalance. May need to balance for the model or look at different evaluation metrics

### Hypothesis: Total day calls, Account length has high importance if someone churns
def hypothesis_exploration(df):
    sns.boxplot(x=train["Churn"],y=train["Total day calls"])
    sns.boxplot(x=train["Churn"],y=train["Account length"])

# PREPROCESSING
### Remove non-relevant and correlated features/columns
def df_cleanup(df1, df2):
    col_to_drop = ['Area code', 'Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge']
    df1_v2 = df1.drop(columns = col_to_drop, axis = 1)
    df2_v2 = df2.drop(columns = col_to_drop, axis = 1)
    ### Combine datasets
    return pd.concat((df1_v2, df2_v2), sort=False)

### Change boolean (True/False) columns to numeric values
def clean_boolean(df):
    boo_cols = df.nunique()[df.nunique() == 2].keys().tolist()
    le = LabelEncoder()
    for i in boo_cols:
        df[i] = le.fit_transform(df[i])
    return df

# BASELINE MODEL
def baseline_model(data, target):
    df = pd.get_dummies(data, drop_first=True)
    x,y = df.drop(target,axis=1),df[[target]]  
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state=123)

    g=GaussianNB()
    b=BernoulliNB()
    KN=KNeighborsClassifier()
    SVC=SVC() 
    D=DecisionTreeClassifier()
    R=RandomForestClassifier()
    Log=LogisticRegression()
    XGB=XGBClassifier()

    algos=[g,b,KN,SVC,D,R,Log,XGB]
    algo_names=[
        'GaussianNB', 
        'BernoulliNB', 
        'KNeighborsClassifier', 
        'SVC', 
        'DecisionTreeClassifier', 
        'RandomForestClassifier', 
        'LogisticRegression',
        'XGBClassifier'
        ]

    accuracy_scored=[]
    precision_scored=[]
    recall_scored=[]
    f1_scored=[]

    for item in algos:
        item.fit(x_train, y_train)
        y_pred = item.predict(x_test)
        accuracy_scored.append(accuracy_score(y_test, y_pred))
        precision_scored.append(precision_score(y_test, y_pred))
        recall_scored.append(recall_score(y_test, y_pred))
        f1_scored.append(f1_score(y_test, y_pred))
    
    result=pd.DataFrame(columns=['f1_score','recall_score','precision_score','accuracy_score'],index=algo_names)
    result['f1_score']=f1_scored
    result['recall_score']=recall_scored
    result['precision_score']=precision_scored
    result['accuracy_score']=accuracy_scored

    print(result.sort_values('accuracy_score',ascending=False))
### Looks like XGBClassifer is the best option.

# MODEL IMPROVEMENTS
### Let's try with balance data
def balance_model(data, target):
    no_churn = data.loc[df[target]==0]
    churn = data.loc[df[target]==1]

    train_churn, test_churn = train_test_split(churn, test_size = 0.10, random_state=200)
    train_no_churn = no_churn.sample(n=len(train_churn), random_state=200)
    test_no_churn = no_churn.drop(train_no_churn.index)

    train_final = pd.concat((train_churn, train_no_churn), sort=True)
    test_final = pd.concat((test_churn, test_no_churn), sort=True)

    y_train = train_final[target]
    y_test = test_final[target]
    x_train = train_final.drop(columns = target, axis = 1)
    x_test = test_final.drop(columns=target, axis=1)

    g=GaussianNB()
    b=BernoulliNB()
    KN=KNeighborsClassifier()
    SVC=SVC() 
    D=DecisionTreeClassifier()
    R=RandomForestClassifier()
    Log=LogisticRegression()
    XGB=XGBClassifier()

    algos=[g,b,KN,SVC,D,R,Log,XGB]
    algo_names=[
        'GaussianNB', 
        'BernoulliNB', 
        'KNeighborsClassifier', 
        'SVC', 
        'DecisionTreeClassifier', 
        'RandomForestClassifier', 
        'LogisticRegression',
        'XGBClassifier'
        ]

    accuracy_scored=[]
    precision_scored=[]
    recall_scored=[]
    f1_scored=[]

    for item in algos:
        item.fit(x_train,y_train)
        item.predict(x_test)
        accuracy_scored.append(accuracy_score(y_test,item.predict(x_test)))
        precision_scored.append(precision_score(y_test,item.predict(x_test)))
        recall_scored.append(recall_score(y_test,item.predict(x_test)))
        f1_scored.append(f1_score(y_test,item.predict(x_test)))
    
    result=pd.DataFrame(columns=['f1_score','recall_score','precision_score','accuracy_score'],index=algo_names)
    result['f1_score']=f1_scored
    result['recall_score']=recall_scored
    result['precision_score']=precision_scored
    result['accuracy_score']=accuracy_scored

    print(result.sort_values('accuracy_score',ascending=False))
### The results are worst. Let's not use balance data. Instead we will need to evaluate the model with precision and recall in addition to accuracy.

### Model Tuning - XGBClassifier
def model_tunning(data, target):
    XGB = XGBClassifier()
    print(XGB.set_params())

    df = pd.get_dummies(data, drop_first=True)
    x, y = df.drop(target,axis=1),df[[target]]  
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state=123)

    params = [{
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.1, 0.3, 0.5, 0.9],
        'booster': ['gbtree', 'gblinear'],
        'gamma': [0, 0.5, 1],
        'reg_alpha': [0, 0.5, 1],
        'reg_lambda': [0.5, 1, 5],
        'base_score': [0.2, 0.5, 0.9]
    }]

    gs = GridSearchCV(XGBClassifier(), params, cv=3, scoring="accuracy")
    gs.fit(x_train, y_train)

    print('Best score:', gs.best_score_)
    print('Best params:', gs.best_params_)
### Default model values has better results. Let's use that instead.

### Confusion Matrix to examine the model in more details
def confusion_matrix(data, target):
    df = pd.get_dummies(data, drop_first=True)
    x,y = df.drop(target,axis=1),df[[target]]  
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state=123)
    #XGB = XGBClassifier(base_score=0.9, booster="gbtree", gamma=1, learning_rate=0.1, 
    #                    n_estimators=100, reg_alpha=0, reg_lambda=5)
    XGB = XGBClassifier()
    XGB.fit(x_train, y_train)
    y_pred = XGB.predict(x_test)
    print(f"accuracy : {accuracy_score(y_test, y_pred)*100}")
    print(f"precision: {precision_score(y_test,item.predict(x_test))}")
    print(f"recall: {recall_score(y_test,item.predict(x_test))}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix/np.sum(conf_matrix), annot=True, fmt='.2%', cmap='Blues')

### Let's look at which features/columns are the most important to the model.
def important_features(data, target):
    df = pd.get_dummies(data, drop_first=True)
    x,y = df.drop(target,axis=1),df[[target]]  
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state=123)
    #XGB = XGBClassifier(base_score=0.9, booster="gbtree", gamma=1, learning_rate=0.1, 
    #                    n_estimators=100, reg_alpha=0, reg_lambda=5)
    XGB = XGBClassifier()
    XGB.fit(x_train, y_train)

    coefficients  = pd.DataFrame(XGB.feature_importances_)
    column_df = pd.DataFrame(x_train.columns)
    coef_sumry = (pd.merge(coefficients,column_df,left_index= True, right_index= True, how = "left"))
    coef_sumry.columns = ["coefficients","features"]
    coef_sumry = coef_sumry.sort_values(by = "coefficients",ascending = False)

    #ax = coef_sumry.head(10).plot.bar(x='features', y='coefficients')
    fig = plt.figure(figsize = (15, 8))
 
    # creating the bar plot
    plt.bar(coef_sumry["features"][:10], coef_sumry["coefficients"][:10], color ='skyblue',
            width = 0.4)
 
    plt.title("Important Features")
    plt.show()

if __name__ == "__main__":
    train, test = get_data()
    print("churn-bigml-80.csv dataset: \n", train(head(10)))
    print("\n churn-bigml-20.csv dataset: \n", test(head(10)))

    # Dataset description
    print("\nDataset Description")
    print("Dataset exploration for train: \n", dataset_description(train))
    print("\nDataset explroation for test: \n", dataset_description(test))
    print("\n Datatypes : \n", get_data_types(train))

    # Exploration
    print("\n Dataset Exploration")
    print("\n Pair Plot of Features: \n", graph_pairplot(train))
    print("\n Correlation Graph: \n", corr_graph(train))
    print("\n Target Feature Exploration: \n", explore_target(train, "Target"))
    print("\n Explorating relevant features: \n", hypothesis_exploration(train))

    # Preprocessing
    telcom_data = df_cleanup(train,test)
    telcom_data = clean_boolean(telcom_data)
    print("\n Check cleaned dataset")
    print("\nDataset explroation for cleaned dataset: \n", dataset_description(telcom_data))
    print("\n Datatypes: \n", get_data_types(telcom_data))

    # Baseline model
    baseline_model(telcom_data, "Churn")

    # Improvements
    balance_model(telcom_data, "Churn")
    model_tunning(telcom_dat, "Churn")

    # Model evualation
    confusion_matrix(telcom_data, "Churn")
    important_features(telcom_data, "Churn")