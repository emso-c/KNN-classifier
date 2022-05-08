from typing import Generator
from .distance_calculator import DistanceCalculator

class KNearestNeighborsClassifier:
    """Implementation the K-Nearest Neighbors Classifier."""

    def __init__(self, k:int, distance_calculation_method:str=DistanceCalculator.methods.EUCLIDEAN):
        """
        Args:
            k (int): The number of neighbors to use in the prediction.
            distance_calculation_method (str, optional): Name of the method to use for
                distance calculation. Defaults to DistanceCalculator.methods.EUCLIDEAN.
        """
        self.k = k
        self.distance_calculation_method = distance_calculation_method
        self.X = []
        self.y = []

    def fit(self, X:list, y:list) -> None:
        """Fit the model to the given data.

        Args:
            X (list): List of features.
            y (list): List of labels.
        """
        self.X = X
        self.y = y

    def _calculate_all_distances(self, target_features:list) -> list:
        """Calculate all distances for the given features.

        Args:
            target_features (list): List of features.
        Returns:

            list: List of distances.
        """

        instance_distances = []

        for features, label in zip(self.X, self.y):
            instance_distances.append({
                "features": features,
                "label": label,
                "distance": DistanceCalculator.calculate_distance(
                    target_features, features, method=self.distance_calculation_method
                )
            })
        return instance_distances

    def _count_labels(self, instance_distances:list) -> dict:
        """Count the labels for the given instance distances.

        Args:
            instance_distances (list): List of instance distances.

        Returns:
            dict: Dictionary with the labels as keys and the count as values.
        """
        label_count = {}
        for instance in instance_distances[:self.k]:
            label_count[instance["label"]] = label_count.get(instance["label"], 0) + 1
        return label_count

    def get_most_common_label(self, label_counts:dict) -> str:
        """Get the most common label from the given label counts.

        Args:
            label_counts (dict): Dictionary with the labels as keys and the count as values.
        
        Returns:
            str: The most common label.
        """

        #TODO fix same amount label issue, get closest instead.
        return max(label_counts, key=label_counts.get)

    def predict_one(self, target_features:list) -> float:
        """Predict the label for the given features.

        Args:
            features (list): List of features.

        Returns:
            float: The predicted label.
        """

        instance_distances = self._calculate_all_distances(target_features)

        instance_distances.sort(key=lambda instance: instance["distance"])

        label_counts = self._count_labels(instance_distances)

        return self.get_most_common_label(label_counts)

    def predict_many(self, target_features:list) -> Generator[float, None, None]:
        """Predict the labels for the given features.

        Args:
            features (list): List of features.

        Returns:
            list: List of predicted labels.
        """

        for feature in target_features:
            yield self.predict_one(feature)
