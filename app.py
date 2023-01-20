from flask import Flask, request, send_file
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
from numpy.random import randint
from sklearn import linear_model
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.metrics import mean_squared_error
import pickle

name = "Success"


app = Flask(__name__)
CORS(app)


def train_reg_model(data):
    encoded_classes = {}
    for i in range(len(data.dtypes.values)):
        if (data.dtypes.values[i] == "O"):
            col = data.columns.values[i]
            label_encoder = LabelEncoder()
            data[col] = label_encoder.fit_transform(data[col])
            encoded_classes[col] = list(label_encoder.classes_)

    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=101)
    ro_scaler = RobustScaler()
    x_train = ro_scaler.fit_transform(x_train)
    x_test = ro_scaler.transform(x_test)

    models = {
        'Linear_Regression': linear_model.LinearRegression(),


        'Ridge_Regression': linear_model.RidgeCV(alphas=(0.1, 0.3, 0.5, 0.7, 0.9, 1, 10, 20, 50)),
        'Lasso_Regression': linear_model.LassoCV(alphas=(0.1, 0.3, 0.5, 0.7, 0.9, 1, 10, 20, 50)),


        'Decision_Tree': DecisionTreeRegressor(random_state=42),
        'Random_Forest': RandomForestRegressor(random_state=42)
    }
    optimize_model = {
        'Decision_Tree': {'max_features': [5, 10], 'max_depth': [3, 5, 7, 9]},
        'Random_Forest': {'n_estimators': [10, 20, 30], 'max_features': [5, 10], 'max_depth': [3, 5, 7, 9]}
    }
    rmse = dict()
    for i, j in models.items():
        if i == 'Ridge_Regression':
            rid = j.fit(x_train, y_train)
            rd = linear_model.Ridge(rid.alpha_)
            rd.fit(x_train, y_train)
            y_pred = rd.predict(x_test)
            mse_rid = mean_squared_error(y_test, y_pred)
            rmse[i] = mse_rid**0.5

        elif i == 'Lasso_Regression':
            ls = j.fit(x_train, y_train)
            las = linear_model.Lasso(ls.alpha_)
            las.fit(x_train, y_train)
            y_pred = las.predict(x_test)
            mse_rid = mean_squared_error(y_test, y_pred)
            rmse[i] = mse_rid**0.5

        elif i == 'Linear_Regression':
            lr = j.fit(x_train, y_train)
            y_pred = lr.predict(x_test)
            mse_rid = mean_squared_error(y_test, y_pred)
            rmse[i] = mse_rid**0.5

        elif i == 'Decision_Tree' or i == 'Random_Forest':
            dt_grid = GridSearchCV(j, optimize_model[i])
            dt_grid.fit(x_train, y_train)
            y_pred = dt_grid.predict(x_test)
            mse_rid = mean_squared_error(y_test, y_pred)
            rmse[i] = mse_rid**0.5

    max_model_name = rmse['Linear_Regression']
    best_model_name = ''
    for i in rmse.keys():
        print(i, " ", rmse[i])
        if max_model_name > rmse[i]:
            max_model_name = rmse[i]
            best_model_name = i

    model = models[best_model_name]
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    global name
    name = best_model_name


@app.route('/', methods=["GET"])
def hello():
    return 'Hello, World'


def train_model(input_df):
    x = input_df.iloc[:, :-1]
    y = input_df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=0)
    models = {
        'Logistic_Regression': LogisticRegression(max_iter=1000, penalty='l2', C=0.04, random_state=0),
        'navie_bayes': GaussianNB(),
        'svc': SVC(random_state=0),
        'Random_Forest': RandomForestClassifier(random_state=0),
        'ada_boost': AdaBoostClassifier(learning_rate=0.01, random_state=0),
        'gradient_boost': GradientBoostingClassifier(random_state=0),
        'sgd': SGDClassifier(random_state=0),
        'Bagging_Classifer': BaggingClassifier(random_state=0),
        'knn_classifier': KNeighborsClassifier()
    }
    accuracy = []
    model_name_1 = []

    def train_model(model, model_name, x=x_train, y=y_train, x_test=x_test):
        model = model.fit(x, y)
        y_pred = model.predict(x_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        model_name_1.append(model_name)

    for model_name, model in models.items():
        train_model(model, model_name)

    my_dict = dict(zip(model_name_1, accuracy))
    my_dict

    max_model_name = my_dict['Logistic_Regression']
    best_model_name = ''
    for i in my_dict.keys():
        if max_model_name < my_dict[i]:
            max_model_name = my_dict[i]
            best_model_name = i

    model = models[best_model_name]
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    global name
    name = best_model_name

# @app.route('/inputs', methods=['POST'])
# def input_values():
#     data = request.files.get('file').read()
#     with open('model.csv', 'w') as f:
#         f.write(str(data, 'utf-8'))

#     input_df = pd.read_csv('model.csv', encoding='utf-8')
#     global name
#     if input_df.index[-1] > 5000:
#         input_df = input_df.sample(n=5000)
#         train_model(input_df)
#     else:
#         train_model(input_df)

#     return name


@app.route('/inputs', methods=['POST'])
def inputs():
    data = request.files.get('file').read()
    with open('model.csv', 'w') as f:
        f.write(str(data, 'utf-8'))

    input_df = pd.read_csv("model.csv", delimiter=',', encoding="utf-8")
    global name
    if input_df.index[-1] > 10000:
        input_df = input_df.sample(n=10000)
        train_reg_model(input_df)
    else:
        train_reg_model(input_df)
    return name


@app.route('/download', methods=['GET'])
def download_file():
    path = './model.pkl'
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
