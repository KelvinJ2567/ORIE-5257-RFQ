import joblib
from sklearn.linear_model import LogisticRegression

def train_logistic_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def save_model(model, filename='logistic_model.pkl'):
    joblib.dump(model, filename)

def load_model(filename='logistic_model.pkl'):
    return joblib.load(filename)

def predict(model, X):
    return model.predict(X)