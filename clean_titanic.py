import numpy as np
import pandas as pd

df = pd.read_csv('C:\\Users\\steph\\Tech Talent South\\Group Final Project\\TTS_Titanic\\train.csv')
train_x = df.drop(columns=['PassengerId', 'Survived', 'Ticket','Name','Cabin'])
train_y = df['Survived']
age_avg = np.mean(df['Age'])
replace = {"Age":age_avg, 'Embarked':'S'}
train_x.fillna(replace, inplace = True)
train_y.to_csv('survived_column.csv')
#train_x.to_csv("clean_titanic.csv")

