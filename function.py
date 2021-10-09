from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def single_reg(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_hat_train)
    test_mse = mean_squared_error(y_test, y_hat_test)
    print('Train Root Mean Square Error:', train_mse**0.5)
    print('Test Root Mean Square Error:', test_mse**0.5)
    
    return model

def log_transform(x):
    return np.log(x)
transformer = FunctionTransformer(log_transform)


def full_reg(model, X_train, X_test, y_train, y_test):
    pipeline = Pipeline([('ss', StandardScaler()), ('transform', transformer), ('regressor', model)])
    
    X_train = X_train.values.reshape(-1,1)
    X_test = X_test.values.reshape(-1,1)

    pipeline.fit(X_train, y_train)
    pipeline.score(X_test, y_test)
    
    return model