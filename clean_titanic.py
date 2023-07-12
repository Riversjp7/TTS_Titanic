import numpy as np
import pandas as pd

#Takes data from a csv file and assigns it to a dataframe
df = pd.read_csv('C:\\Users\\steph\\Tech Talent South\\Group Final Project\\TTS_Titanic\\train.csv')

#Drops irrelevant data columns and assigns this to a training dataset
train_x = df.drop(columns=['PassengerId', 'Ticket','Name','Cabin'])

#Calculates the average age of the age column
age_avg = np.mean(df['Age'])

#Creates a dictionary using the average age and Southampton('S') for the Embarked column
replace = {"Age":age_avg, 'Embarked':'S'}

#Fills null values in the age column with the average age value and null values in the Embarked column with 'S'
train_x.fillna(replace, inplace = True)

#Creates a csv file
train_x.to_csv('clean_titanic.csv')