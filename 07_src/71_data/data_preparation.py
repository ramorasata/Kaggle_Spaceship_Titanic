"""
    Data preparation for data analysis
    ----------------------------------
    Encoding data and imputing missing values
    Author : Johary RAMORASATA
"""

# Dependencies
import numpy as np
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# Import Data
train_data = pd.read_csv("./01_data/11_raw/train.csv", header=0)
test_data = pd.read_csv("./01_data/11_raw/test.csv", header=0)

train_data, y_train = train_data[train_data.columns.difference(["Transported"])], train_data["Transported"]
test_data = test_data[train_data.columns]


# Feature preprocessing
simple_imputer = SimpleImputer(strategy="most_frequent")
train_df, test_df = train_data.copy(), test_data.copy()

def cleaner(df) :
    df = df.copy()
    df.index = df["PassengerId"]

    df["Destination"] = df.Destination.str[:4]
    df[["CryoSleep","VIP"]] = df[["CryoSleep","VIP"]].astype(bool)
    df[["PassengerGroup", "PassengerNum"]] = df.PassengerId.str.split("_", expand=True).astype(np.int32)
    df[["Deck","Num","Side"]] = df.Cabin.str.split("/", expand=True)
    df.drop(["Name","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck",
             "PassengerId","PassengerNum","Cabin","Num"], axis=1, inplace=True)
    df[["Age","Destination","HomePlanet","Deck","Side"]] = simple_imputer.fit_transform(df[["Age","Destination","HomePlanet","Deck","Side"]])
    return df

train_df, test_df = cleaner(train_df), cleaner(test_df)


# Feature Encoding
encoder = OneHotEncoder(drop="first")
encoder.fit(pd.concat([train_df[["Destination","HomePlanet","Deck","Side"]],
                       test_df[["Destination","HomePlanet","Deck","Side"]]]))

def encode(df) :
    index=df.index
    encoded_cols = pd.DataFrame(encoder.transform(df[["Destination","HomePlanet","Deck","Side"]]).toarray(),
                                                                    index=index, columns=encoder.get_feature_names_out())
    
    df.drop(["Destination","HomePlanet","Deck","Side"], axis=1, inplace=True)
    df = pd.concat([df, encoded_cols], axis=1)
    return df

train_df, test_df = encode(train_df), encode(test_df)


# Save files to pkl format
with open('./01_data/12_interim/CLEAN_TRAIN_DF.pkl', 'wb') as f:
    pickle.dump(train_df, f)

with open('./01_data/12_interim/Y_TRAIN.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('./01_data/12_interim/CLEAN_TEST_DF.pkl', 'wb') as f:
    pickle.dump(test_df, f)
