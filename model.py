import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_excel("Dataset.xlsx")
from sklearn.impute import SimpleImputer
si = SimpleImputer(strategy='mean')
Num_column = ["Test1","Test2","Test3","Attendance","Distance_From_Institute"]
df[Num_column] = si.fit_transform(df[Num_column])
df.ffill(inplace=True)
z_scores = (df['Test1'] - df['Test1'].mean()) / df['Test1'].std()#formula of Z score
threshold = 2
outlier = (z_scores.abs() > threshold)
df = df[~outlier]
df['Distance_From_Institute'] = df['Distance_From_Institute'].abs()
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df.sample(2)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df['Attrition'] = label_encoder.fit_transform(df['Attrition'])
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High'], ['Poor', 'Average', 'Good']])

df[['Interest', 'Health Status']] = ordinal_encoder.fit_transform(df[['Interest', 'Health Status']])
correlation_matrix = df[['Test1', 'Test2','Test3', 'Interest', 'Attrition',"Attendance"]].corr()
print(correlation_matrix)
import nltk
nltk.download('vader_lexicon')
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
def get_sentiment_score(feedback):
    return sia.polarity_scores(feedback)['compound']
df['sentiment_score'] = df['Teacher_feedback'].apply(get_sentiment_score)

# Display the DataFrame with sentiment scores
print(df[['Teacher_feedback', 'sentiment_score']])
df.drop(['sentiment_score'],axis=1,inplace=True)
df.sample(3)
df.drop(['Gender'],axis=1,inplace=True)
df.sample(3)
df.drop(['Teacher_feedback'],axis=1,inplace=True)
df.sample(3)
df.drop(['Study Hours per Week', 'Distance_From_Institute', 'Health Status'], axis=1,inplace = True)
df.sample(3)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

LR = LogisticRegression(random_state=42, class_weight='balanced')

# Train the model
LR.fit(X_train, y_train)

y_pred = LR.predict(X_test)

print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred))
from sklearn.naive_bayes import GaussianNB

X = df.drop('Attrition', axis=1)
y = df['Attrition']


NB = GaussianNB()

# Train the model
NB.fit(X_train, y_train)


y_pred = NB.predict(X_test)

print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier

X = df.drop('Attrition', axis=1)
y = df['Attrition']


DT = DecisionTreeClassifier(random_state=42, class_weight='balanced')

# Train the model
DT.fit(X_train, y_train)

y_pred = DT.predict(X_test)

print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred))
import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(DT, file)