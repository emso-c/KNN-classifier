from emsoKNN.collector import read_csv
from emsoKNN.preprocessor import parse_csv_data
from emsoKNN.knn_manager import KnnManager

if __name__ == '__main__':
    csv_data = read_csv('data/iris/iris.csv')
    X, y = parse_csv_data(csv_data)

    knn_manager = KnnManager()
    knn_manager.fit(X, y)

    knn_manager.calculate_benchmark_metrics(
        range(3, 20, 2), benchmark_size=10, test_ratio=.2, split_method="random",
    )

    print("worst accuracy:", knn_manager.min_accuracy)
    print("average accuracy:", knn_manager.average_accuracy)
    print("best accuracy:", knn_manager.max_accuracy)
    
    k_best, acc = knn_manager.find_k_best()
    print(f"{k_best} is the best k value with an accuracy of {acc*100}%")
    
    knn_manager.plot_accuracy_scores()

