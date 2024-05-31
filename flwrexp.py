import flwr as fl
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

split_idx = len(X_train) // 2
X_train_client1, X_train_client2 = X_train[:split_idx], X_train[split_idx:]
y_train_client1, y_train_client2 = y_train[:split_idx], y_train[split_idx:]

class BreastCancerClient(fl.client.NumPyClient):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.model = LogisticRegression(max_iter=100)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model.fit(self.X_train, self.y_train) 
    def get_parameters(self, config):
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = 1 - accuracy_score(self.y_test, self.model.predict(self.X_test))
        return float(loss), len(self.X_test), {}

def client_fn(cid):
    if cid == "0":
        return BreastCancerClient(X_train_client1, y_train_client1, X_test, y_test)
    else:
        return BreastCancerClient(X_train_client2, y_train_client2, X_test, y_test)

strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2,
    min_available_clients=2,
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
