import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from ML_Algorithms import get_classifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("data.csv")

df.dropna(inplace=True)

df.loc[(df['rebuffer_count'] == 0), 'rebuffer_count'] = 0

df.loc[~(df['rebuffer_count'] == 0), 'rebuffer_count'] = 1

df  = df.rename(columns={'rebuffer_count':'label'})

df.drop(columns=["rebuffer_duration"],inplace=True)

df["content_type"] = pd.Categorical(df.content_type).codes

df["content_genre"] = pd.Categorical(df.content_genre).codes

df["playback_start_source"] = pd.Categorical(df.playback_start_source).codes

df["device_platform"] = pd.Categorical(df.device_platform).codes

df["player_device_platform"] = pd.Categorical(df.player_device_platform).codes

df["device_model"] = pd.Categorical(df.device_model).codes

df["network_mode"] = pd.Categorical(df.network_mode).codes

df["user_geoip_region"] = pd.Categorical(df.user_geoip_region).codes

df["user_geoip_isp"] = pd.Categorical(df.user_geoip_isp).codes

df["player_version"] = pd.Categorical(df.player_version).codes

X = df.drop(columns=["label"])

Y = df["label"]


X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.30, random_state=42)
model_list = ["NB", "GradientBoostingClassifier", "DecisionTree",
              "KNeighbors", "XGB"]
for model in model_list:
    print("#"*50)
    print(model)
    print("#"*50)

    clf = get_classifier(model)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy",round(accuracy_score(y_test,y_pred),4))
    print("Precision",round(precision_score(y_test,y_pred),4))
    print("Recall",round(recall_score(y_test,y_pred),4))
    print("F1",round(f1_score(y_test,y_pred),4))


