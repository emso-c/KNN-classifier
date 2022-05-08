"""Functions to preprocess data."""

import random

# TODO add feature importance

def parse_csv_data(data:list, header=True, label_amount:int=1) -> tuple[list, list]:
    """Parse csv data into X features and y labels.
    
    Args:
        data (list): list of lists, each list is a row of the csv file.
        label_amount (int): amount of labels in the csv file. Defaults to 1.

    Returns:
        tuple(list, list): X features and y labels.
    """

    if label_amount > 1:
        raise NotImplementedError("Multiple labels are not supported yet.")

    X, y = [], []
    for row in data[int(header):]:
        X.append([float(x) for x in row[:-label_amount]])
        y.append(row[-label_amount])
    return X, y

def drop_column(csv_data:list, column_index:int) -> None:
    """Drop a column from csv data.

    Args:
        csv_data (list): list of lists, each list is a row of the csv file.
        column_index (int): integer of the column to be dropped.

    Returns:
        list: List of predicted labels.
    """
    for row in csv_data:
        del row[column_index]

def scale_data(dataset:list, percentage:float) -> list:
    """Scale parsed csv data by percentage. Percentage should be between 0 and 1.
    
    Args:
        dataset (list): list of lists, each list is a row of the csv file.
        percentage (float): percentage of data to be scaled.
    
    Returns:
        list: list of lists, each list is a row of the csv file.
    """

    if percentage < 0 or percentage > 1:
        raise ValueError("Percentage should be between 0 and 1.")

    return dataset[:int(len(dataset) * percentage)]

def encode_categorical_data(csv_data:list, method:str="integer") -> list:
    """Encode categorical data into integer format.

    Args:
        csv_data (list): list of lists, each list is a row of the csv file.
        method (str): method to encode the attributes. One of 'ord_sum', 'Integer'.
            'ord_sum': each unique string attribute is a summation of it's unicode characters.
            'integer': unique strings are indexed sequentially. Starting from 1.
            Defaults to 'integer'.

    Returns:
        list of lists, each list is a row of the csv file.
    """
    # TODO refactor logic

    encoded_csv_data = []
    sequence_map = {}
    for row in csv_data[1:]:
        encoded_row = []
        for value in row:
            if type(value) == str:
                try:
                    encoded_row.append(float(value))
                except ValueError:
                    if method == "ord_sum":
                        ch_sum = sum([ord(ch) for ch in value])
                        encoded_row.append(ch_sum)
                    elif method == "integer":
                        sequence_map[value] = sequence_map.get(value, len(sequence_map)+1)
                        encoded_row.append(sequence_map[value])
                    else:
                        raise ValueError("Invalid method")
        encoded_csv_data.append(encoded_row)
    return encoded_csv_data

def split_dataset_random(X:list, y:list, test_ratio:float=0.4, seed=None) -> tuple[list, list, list, list]:
    """Split dataset into train and test sets.

    Args:
        X (list): list of lists, each list is a row of the parsed csv file.
        y (list): list of labels.
        test_ratio (float): percentage of data to be used for testing.
        seed (int): seed for random.seed(). Defaults to None.
    
    Returns:
        tuple(list, list, list, list): train_X, train_y, test_X, test_y.
    """
    random.seed(seed)
    train_size = int(len(X) * (1-test_ratio))

    temp_dataset = list(zip(X, y))
    random.shuffle(temp_dataset)
    X, y = zip(*temp_dataset)

    return X[:train_size], y[:train_size], X[train_size:], y[train_size:]

def split_dataset_homogenous(X, y, test_ratio:float=0.4, seed=None):
    raise NotImplementedError

    train_size = int(len(X) * test_ratio)
    label_set = set(y)
    train_features, train_labels = [], []
    test_features, test_labels = [], []
    
    # shuffle data
    random.seed(seed)
    shuffled = list(zip(X, y))
    random.shuffle(shuffled)
    X, y = zip(*shuffled)

    # split data for each label
    label_dict = {label: [] for label in label_set}
    for feature_data, label in zip(X, y):
        label_dict[label].append(feature_data)

    label_pool = cycle(label_set)
    for label in label_pool:
        if label_dict[label]:
            train_features.append(label_dict[label])
        # TODO

    return train_features, train_labels, test_features, test_labels
    for label in set(y):
        label_features, label_labels = [], []
        for feature, label in zip(X, y):
            if label == label:
                label_features.append(feature)
                label_labels.append(label)
        train_features.append(label_features[:train_size])
        train_labels.append(label_labels[:train_size])
        test_features.append(label_features[train_size:])
        test_labels.append(label_labels[train_size:])
    return train_features, train_labels, test_features, test_labels

def split_dataset(X, y, test_ratio:float=0.4, seed=None, method='homogenous') -> tuple[list, list, list, list]:
    """Split dataset into train and test sets.
    
    Args:
        X (list): list of lists, each list is a row of the parsed csv file.
        y (list): list of labels.
        test_ratio (float): percentage of data to be used for testing.
        seed (int): seed for random.seed(). Defaults to None.
        method (str): method to split the dataset. One of 'homogenous', 'random'.
            'homogenous': split dataset into train and test sets based on homogeneity.
            'random': split dataset into train and test sets randomly.
            Defaults to 'homogenous'.
        
    Returns:
        tuple(list, list, list, list): train_X, train_y, test_X, test_y.
    """

    if method == 'homogenous':
        return split_dataset_homogenous(
            X=X,
            y=y,
            test_ratio=test_ratio,
            seed=seed
        )
    elif method == 'random':
        return split_dataset_random(
            X=X,
            y=y,
            test_ratio=test_ratio,
            seed=seed
        )
    else:
        raise ValueError("Invalid method")
