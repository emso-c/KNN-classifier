from dataclasses import dataclass
from .knn_classifier import KNearestNeighborsClassifier
from .distance_calculator import DistanceCalculator
from .preprocessor import split_dataset

from matplotlib import pyplot as plt

@dataclass
class KAccuracy:
    k: int
    accuracy: float

class KnnManager():
    """A class to manage and benchmark the KNN algorithm"""

    def __init__(self, k=3):
        self.k = k
        self.classifier = KNearestNeighborsClassifier(k)
        
        self.benchmark = []
        self.X = []
        self.y = []

    def fit(self, X, y) -> None:
        """Fit the model to the given data."""

        self.X = X
        self.y = y
        self.classifier.fit(X, y)

    def calculate_accuracy_score(self, y_predictions, y_test) -> float:
        """Calculate the accuracy score of the predictions.
        
        Args:
            y_predictions (list): The predictions made by the classifier.
            y_test (list): The actual labels of the test data.
        """

        return sum([1 for prediction, label in zip(y_predictions, y_test) if prediction == label]) / len(y_test)

    def get_single_observation(
            self, k_range:range, test_ratio:float=.3, seed=None, split_method="random",
            distance_calculation_method=DistanceCalculator.methods.EUCLIDEAN
        ) -> list[KAccuracy]:
        """Get the accuracy scores for a single observation.
        
        Args:
            k_range (range): The range of K values to test.
            test_ratio (float): The ratio of the dataset to use for testing.
            seed (int): The seed to use for the random split.
            split_method (str): The method to use for splitting the dataset.
            distance_calculation_method (str): The method to use for calculating the distance.
        
        Returns:
            list[KAccuracy]: The accuracy scores for each K value.
        """
        self.classifier.distance_calculation_method = distance_calculation_method
        k_accuracies = []
        for k_val in k_range:
            x_train, y_train, x_test, y_test = split_dataset(
                self.X, self.y, test_ratio=test_ratio,
                seed=seed, method=split_method
            )
            self.classifier.k = k_val
            self.classifier.fit(x_train, y_train)
            y_predictions = self.classifier.predict_many(x_test)
            k_accuracies.append(KAccuracy(
                k_val,
                self.calculate_accuracy_score(y_predictions, y_test)
            ))
        return k_accuracies

    def calculate_benchmark_metrics(
            self, k_range:range, benchmark_size:int, test_ratio:float=.3,
            distance_calculation_method=DistanceCalculator.methods.EUCLIDEAN, split_method="random"
        ) -> list[list[KAccuracy]]:
        """Get the accuracy scores for a benchmark.

        Args:
            k_range (range): The range of K values to test.
            benchmark_size (int): The number of observations to use for the benchmark.
            test_ratio (float): The ratio of the dataset to use for testing.
            distance_calculation_method (str): The method to use for calculating the distance.
            split_method (str): The method to use for splitting the dataset.
            
        Returns:
            list[list[KAccuracy]]: The list of accuracy scores for each observation."""
        self.benchmark = []

        seeds = [i+1 for i in range(benchmark_size)]
        for seed in seeds:
            self.benchmark.append(self.get_single_observation(
                k_range, test_ratio, seed, split_method, distance_calculation_method
            ))
        return self.benchmark

    def plot_accuracy_score(self, k_accuracies:list[KAccuracy]) -> None:
        """Plot the accuracy scores for a single observation.
        
        Args:
            k_accuracies (list[KAccuracy]): The accuracy scores for a single observation.
        """

        x, y = [], []
        for k_accuracy in k_accuracies:
            k, accuracy = k_accuracy
            x.append(k)
            y.append(accuracy)

        plt.plot(x, y)
        plt.xlabel("K values")
        plt.ylabel("% Accuracy")
        plt.title("Algorithm Performance")
        plt.show()

    def plot_accuracy_scores(self) -> None:
        """Plot the accuracy scores for the benchmark."""

        legend = [f"Observation #{i}" for i in range(1, len(self.benchmark)+1)]
        all_plot_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']*3
        colors = all_plot_colors[:len(self.benchmark)]

        for k_accuracies, color in zip(self.benchmark, colors):
            x, y = [], []
            for k_accuracy in k_accuracies:
                x.append(k_accuracy.k)
                y.append(k_accuracy.accuracy*100)
            plt.plot(x, y, color)

        plt.xlabel("K values")
        plt.ylabel("% Accuracy")
        plt.title("Algorithm Performance")
        plt.legend(legend)
        plt.show()

    @property
    def average_accuracy(self) -> float:
        """Get the average accuracy of the benchmark."""

        accuracy_scores = []
        for k_accuracies in self.benchmark:
            for k_accuracy in k_accuracies:
                accuracy_scores.append(k_accuracy.accuracy)

        return sum(accuracy_scores) / len(accuracy_scores)

    @property
    def min_accuracy(self) -> float:
        """Get the minimum accuracy of the benchmark."""

        accuracy_scores = []
        for k_accuracies in self.benchmark:
            for k_accuracy in k_accuracies:
                accuracy_scores.append(k_accuracy.accuracy)

        return min(accuracy_scores)

    @property
    def max_accuracy(self) -> float:
        """Get the maximum accuracy of the benchmark."""

        accuracy_scores = []
        for k_accuracies in self.benchmark:
            for k_accuracy in k_accuracies:
                accuracy_scores.append(k_accuracy.accuracy)

        return max(accuracy_scores)

    def find_k_bests(self) -> tuple[list, list]:
        """Find the K values with the highest accuracy."""

        k_bests, k_best_accuracies = [], []
        for k_accuracies in self.benchmark[1:]:
            k_best, k_best_accuracy = 0, 0
            for k_accuracy in k_accuracies:
                if k_accuracy.accuracy > k_best_accuracy:
                    k_best, k_best_accuracy = k_accuracy.k, k_accuracy.accuracy
            k_bests.append(k_best)
            k_best_accuracies.append(k_best_accuracy)
        return k_bests, k_best_accuracies

    def find_k_best(self) -> tuple[int, float]:
        """Find the K value with the highest accuracy."""

        k_bests, k_best_accuracies = self.find_k_bests()
        k_best_overall, k_best_accuracy_overall = k_bests[0], k_best_accuracies[0]
        for k_best, k_best_accuracy in zip(k_bests, k_best_accuracies):
            if k_best_accuracy > k_best_accuracy_overall:
                k_best_overall, k_best_accuracy_overall = k_best, k_best_accuracy
        return k_best_overall, k_best_accuracy_overall
