import joblib
from sklearn.linear_model import LogisticRegression
import hashlib

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

def hash_tuple(t: tuple) -> str:
    """Get the hash value of a tuple

    Args:
        t: The tuple to hash.

    Returns:
        str. The hash value of the tuple.
    """
    md5 = hashlib.md5()
    md5.update(str(t).encode())
    return md5.hexdigest()