import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.metrics import classification_report, mean_squared_error

def load_training_data():
    data_treino = "../datasets/data_4000v/env_vital_signals.txt"
    data_validacao = "../datasets/data_800v/env_vital_signals.txt"

    df_treino = pd.read_csv(data_treino, header=None)
    X_treino = df_treino.drop(df_treino.columns[[0, 1, 2, 6, 7]], axis=1)
    value_treino = df_treino.iloc[:, 6]
    class_treino = df_treino.iloc[:, 7]

    df_validacao = pd.read_csv(data_validacao, header=None)
    value_validacao = df_validacao.iloc[:, 6]
    class_validacao = df_validacao.iloc[:, 7]
    X_validacao = df_validacao.drop(df_validacao.columns[[0, 1, 2, 6, 7]], axis=1)

    return X_treino, value_treino, class_treino, X_validacao, value_validacao, class_validacao

def train_DTClassifier(X_treino, class_treino, X_validacao, class_validacao, max_depth=100, min_samples_leaf=2):
    dtc = DecisionTreeClassifier(random_state=42, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    dtc.fit(X_treino, class_treino)
    class_pred = dtc.predict(X_validacao)

    print(f"DTClassifier max_depth={max_depth} min_samples_leaf={min_samples_leaf}") 
    print(classification_report(class_validacao, class_pred))
    print("")
    return dtc

def train_MLPClassifier(X_treino, class_treino, X_validacao, class_validacao, hidden_layer_sizes=(50,), max_iter=500):
    mlpc = MLPClassifier(random_state=42, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter) 
    mlpc.fit(X_treino, class_treino)

    class_pred = mlpc.predict(X_validacao)

    print(f"MLPClassifier hidden_layer_sizes={hidden_layer_sizes} max_iter={max_iter}") 
    print(classification_report(class_validacao, class_pred))
    print("")
    return mlpc

def train_DTRegressor(X_treino, value_treino, X_validacao, value_validacao, max_depth=100, min_samples_leaf=2):
    dtc = DecisionTreeRegressor(random_state=42, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    dtc.fit(X_treino, value_treino)
    value_pred = dtc.predict(X_validacao)

    print(f"DTRegressor max_depth={max_depth} min_samples_leaf={min_samples_leaf}") 
    print(f"    rmse {mean_squared_error(value_validacao, value_pred)**0.5}")
    print("")
    return dtc

def train_MLPRegressor(X_treino, value_treino, X_validacao, value_validacao, hidden_layer_sizes=(50,), max_iter=500):
    mlpc = MLPRegressor(random_state=42, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter) 
    mlpc.fit(X_treino, value_treino)

    value_pred = mlpc.predict(X_validacao)

    print(f"MLPRegressor hidden_layer_sizes={hidden_layer_sizes} max_iter={max_iter}") 
    print(f"    rmse {mean_squared_error(value_validacao, value_pred)**0.5}")
    print("")
    return mlpc


if __name__ == '__main__':
    X_treino, value_treino, class_treino, X_validacao, value_validacao, class_validacao = load_training_data()
    train_DTClassifier(X_treino, class_treino, X_validacao, class_validacao)
    train_MLPClassifier(X_treino, class_treino, X_validacao, class_validacao)

    train_DTRegressor(X_treino, value_treino, X_validacao, value_validacao)
    train_MLPRegressor(X_treino, value_treino, X_validacao, value_validacao)
