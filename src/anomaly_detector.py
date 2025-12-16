from sklearn.ensemble import IsolationForest
import numpy as np

class AnomalyDetector:
    """
    Unsupervised anomaly detector based on Isolation Forest.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state
        )

    def fit(self, X) -> None:
        """
        Fit the anomaly detection model on feature matrix X.
        """
        self.model.fit(X)

    def detect(self, X_new):
        """
        Detect anomalies on new data.
        Returns:
            mean_score: float
            labels: array-like (1 = normal, -1 = anomaly)
        """
        scores = self.model.decision_function(X_new)
        labels = self.model.predict(X_new)
        return float(np.mean(scores)), labels
