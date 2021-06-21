from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier



import numpy as np
def get_classifier(estimator):
    try:
        classifier = None

        if estimator == "DecisionTree":
            classifier =  DecisionTreeClassifier(max_depth=2)
        elif estimator=="NB":
            classifier = GaussianNB()
        elif estimator=="KNeighbors":
            classifier = KNeighborsClassifier(n_neighbors=3)
        elif estimator=="GradientBoostingClassifier":
            classifier =GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
        elif estimator=="XGB":
            classifier =XGBClassifier()

    except Exception as e:
        print(e)
        return

    try:
        object.__setattr__(classifier, "random_state", 0)
    except:
        pass
    return classifier

