from sklearn.base import ClassifierMixin
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

MODELS: list[list[ClassifierMixin]] = [
    # base
    [
        KNeighborsClassifier(),
        LogisticRegression(),
        DecisionTreeClassifier(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        GaussianNB(),
    ],
    # # more complex
    [RandomForestClassifier(), XGBClassifier(), MLPClassifier()],
    # base + base HP
    [
        KNeighborsClassifier(n_neighbors=1),
        KNeighborsClassifier(n_neighbors=3),
        KNeighborsClassifier(n_neighbors=5),
        ElasticNet(l1_ratio=0.1),
        ElasticNet(l1_ratio=0.5),
        ElasticNet(l1_ratio=0.9),
        DecisionTreeClassifier(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        GaussianNB(),
    ],
    # # base + complx + HP
    [
        KNeighborsClassifier(n_neighbors=1),
        KNeighborsClassifier(n_neighbors=3),
        KNeighborsClassifier(n_neighbors=5),
        ElasticNet(l1_ratio=0.1),
        ElasticNet(l1_ratio=0.5),
        ElasticNet(l1_ratio=0.9),
        DecisionTreeClassifier(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        GaussianNB(),
        RandomForestClassifier(n_estimators=50),
        RandomForestClassifier(n_estimators=100),
        RandomForestClassifier(n_estimators=150),
        XGBClassifier(),
        MLPClassifier(hidden_layer_sizes=[100]),
        MLPClassifier(hidden_layer_sizes=[50, 100, 50]),
        MLPClassifier(hidden_layer_sizes=[100, 100, 100]),
    ],
]


ROUTERS: list[ClassifierMixin] = [
    KNeighborsClassifier(n_neighbors=1),
    KNeighborsClassifier(n_neighbors=3),
    KNeighborsClassifier(n_neighbors=5),
    MLPClassifier(),
    RandomForestClassifier(),
]
