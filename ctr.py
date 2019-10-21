# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# To avoid Depricated warnings
import warnings
import os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Function for parsing data from date column as date
def parser(x):
    return pd.datetime.strptime(x, '%Y%m%d')


# Importing dataset as Pandas DataFrame
dataset = pd.read_csv("dataset.csv", parse_dates=[0], date_parser=parser, header=None)

# Droping NA values from Date, Market, Keyword
dataset = dataset[pd.isna(dataset[1]) == False]
dataset = dataset[pd.isna(dataset[0]) == False]
dataset = dataset[pd.isna(dataset[3]) == False]


# Making Date column as pandas.DateTime
dataset.iloc[:, 0] = pd.to_datetime(dataset.iloc[:, 0])

# Adding Day of Week number column into dataset
dataset[9] = dataset[0].apply(lambda x: x.dayofweek)

# Adding Is_Holiday column into dataset for Holiday Date's
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
cal = calendar()
holidays = cal.holidays(start=dataset[0].min(), end=dataset[0].max())
dataset[10] = dataset[0].isin(holidays)
# dataset.iloc[:, 10] = dataset.iloc[:, 10].astype("int")
dataset[10] = dataset[10].astype("int")

# NLP Operations
# Converting all keywords into lower case
dataset.iloc[:, 2] = dataset.iloc[:, 2].astype("str")
dataset[2] = dataset[2].apply(lambda x: x.lower())

# Importing NLTK Libraries
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

# Spliting keywords into words
dataset.iloc[:, 2] = dataset.iloc[:, 2].apply(lambda x: x.split())

# Function to stemming and lemmatization on words
def nlpo(x):
    ww = []
    for word in x:
        if not word in set(stopwords.words('english')):
          ww.append(lemmatizer.lemmatize(ps.stem(word)))
    return ww

dataset.iloc[:, 2] = dataset.iloc[:, 2].apply(lambda x: nlpo(x))

# joining the words again as sentence
dataset.iloc[:, 2] = dataset.iloc[:, 2].apply(lambda x: " ".join(x))

# Model Training and Testing
label_dict = {}
lcount = 0
def encode_labels(x):
    global lcount
    try:
        return label_dict[x]
    except:
        label_dict[x] = lcount
        lcount += 1
        return label_dict[x]

dataset[11] = dataset[2].apply(lambda x: encode_labels(x))

# Spliting the dataset
X = dataset.iloc[:, [1, 2, 4, 9, 10]].values
y = dataset.iloc[:, 5].values




# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_market = LabelEncoder()
X[:, 0] = labelencoder_market.fit_transform(X[:, 0])

# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_market = LabelEncoder()
X[:, 0] = labelencoder_market.fit_transform(X[:, 0])

labelencoder_keyword = LabelEncoder()
X[:, 1] = labelencoder_keyword.fit_transform(X[:, 1])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting the RandomForest model to training dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0, n_jobs=2)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


# Fitting the LightGBM Model to Training dataset
import lightgbm as ltb
model = ltb.LGBMRegressor()
model.fit(X, y)
predicted_y = model.predict(X_test)
