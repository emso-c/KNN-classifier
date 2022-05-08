import math

class DistanceCalculationMethods:
    EUCLIDEAN = 'euclidean'
    MANHATTAN = 'manhattan'
    CHEBYSHEV = 'chebyshev'
    MINKOWSKI = 'minkowski'

class DistanceCalculator:
    """Contains methods for calculating distances between two vectors."""

    methods = DistanceCalculationMethods

    @staticmethod
    def calculate_distance(feature_1:list, feature_2:list, method=DistanceCalculationMethods.EUCLIDEAN) -> float:
        if method == DistanceCalculationMethods.EUCLIDEAN:
            return DistanceCalculator.euclidean_distance(feature_1, feature_2)
        elif method == DistanceCalculationMethods.MANHATTAN:
            return DistanceCalculator.manhattan_distance(feature_1, feature_2)
        elif method == DistanceCalculationMethods.CHEBYSHEV:
            raise NotImplementedError
        elif method == DistanceCalculationMethods.MINKOWSKI:
            raise NotImplementedError

    @staticmethod
    def euclidean_distance(feature_1:list, feature_2:list) -> float:
        return math.sqrt(sum([(x - y)**2 for x, y in zip(feature_1, feature_2)]))

    @staticmethod
    def manhattan_distance(feature_1:list, feature_2:list) -> float:
        return sum([abs(x - y) for x, y in zip(feature_1, feature_2)])