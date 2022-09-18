import numpy as np
import math
import sys


class Node:
    """
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    """

    def __init__(self):
        self.left = None
        self.right = None
        self.left_len = None
        self.right_len = None
        self.attr = None
        self.vote = None
        self.type = None


# TODO Define Some Important Metric Functions
def entropy(train_data, feature_index=None):
    # TODO Define Entropy Function
    if train_data.shape[0] == 0:
        return 0.0
    ent = 0
    if feature_index is not None:
        # Define P(x)
        positive_p = train_data[train_data[:, -1] == 1].shape[0] / train_data.shape[0]
        negative_p = train_data[train_data[:, -1] == 0].shape[0] / train_data.shape[0]

        # Define P(x, y)
        positive_negative_p = train_data[np.logical_and(train_data[:, -1] == 0, train_data[:, feature_index] == 1)].shape[0] / \
                              train_data.shape[0]  # p(x,y)
        positive_positive_p = train_data[np.logical_and(train_data[:, -1] == 1, train_data[:, feature_index] == 1)].shape[0] / \
                              train_data.shape[0]
        negative_positive_p = train_data[np.logical_and(train_data[:, -1] == 1, train_data[:, feature_index] == 0)].shape[0] / \
                              train_data.shape[0]
        negative_negative_p = train_data[np.logical_and(train_data[:, -1] == 0, train_data[:, feature_index] == 0)].shape[0] / \
                              train_data.shape[0]

        # Calculate H(Y|X)
        if positive_negative_p != 0 and positive_p != 0:
            ent -= positive_negative_p * math.log2(positive_negative_p / positive_p)
        if positive_positive_p != 0 and positive_p != 0:
            ent -= positive_positive_p * math.log2(positive_positive_p / positive_p)
        if negative_positive_p != 0 and negative_p != 0:
            ent -= negative_positive_p * math.log2(negative_positive_p / negative_p)
        if negative_negative_p != 0 and negative_p != 0:
            ent -= negative_negative_p * math.log2(negative_negative_p / negative_p)
    else:
        # Define P(x)
        positive_p = train_data[train_data[:, -1] == 1].shape[0] / train_data.shape[0]
        negative_p = train_data[train_data[:, -1] == 0].shape[0] / train_data.shape[0]

        # Calculate H(X)
        if positive_p != 0.0:
            ent -= positive_p * math.log2(positive_p)
        if negative_p != 0.0:
            ent -= negative_p * math.log2(negative_p)

    return ent


def mutual_information(train_data, feature_index):
    # TODO Define Mutual_Information Function
    hy = entropy(train_data)  # (H(Y)
    hyx = entropy(train_data, feature_index)  # H(Y|X)
    mi = hy - hyx  # H(Y) - H(Y|X)

    return mi


# TODO Implement Some Tool Functions
def divide_data(train_data, feature_index):
    data_left = train_data[train_data[:, feature_index] == 0]
    data_right = train_data[train_data[:, feature_index] == 1]
    return data_left, data_right


def load_data(data_path):
    data = []
    with open(data_path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip().split('\t')
            if idx == 0:
                idx2label = line
            else:
                data.append([int(value) for value in line])

    data = np.array(data)
    return data, idx2label


def error(label, prediction):
    n_sample = len(label)
    n_error = 0.0

    for i in range(n_sample):
        if label[i] != prediction[i]:
            n_error += 1

    error_rate = n_error / n_sample

    return error_rate


# TODO Implement Recursive Training Process
def train(train_data, max_depth):
    root = tree_recurse(train_data, max_depth)
    return root


def tree_recurse(train_data, max_depth, depth=0, visited=None):
    if visited is None:
        visited = []

    node = Node()

    # Check if features are identical
    identical = False
    if train_data.shape[0] != 0:
        features = train_data[0, :-1]
        for i in range(1, train_data.shape[0]):
            if train_data[i, :-1].tolist() != features.tolist():
                break
            if i == train_data.shape[0] - 1:
                identical = True

    # Other stopping criterion
    if depth >= max_depth or train_data.shape[0] == 0 or max(train_data[:, -1]) == min(
            train_data[:, -1]) or identical or len(visited) == train_data.shape[1] - 1:
        left_data, right_data = divide_data(train_data, -1)
        node.left_len = left_data.shape[0]
        node.right_len = right_data.shape[0]
        node.type = 'leaf'
        if node.left_len <= node.right_len:
            node.vote = 1
        else:
            node.vote = 0
    else:
        best_mi = None
        best_feature = None
        for feature_index in range(train_data.shape[1] - 1):
            if feature_index in visited:
                continue
            mi = mutual_information(train_data, feature_index)

            if best_mi is None or best_mi < mi:
                best_mi = mi
                best_feature = feature_index

        if best_mi >= 0:
            node.attr = best_feature
            left_data, right_data = divide_data(train_data, best_feature)
            node.left_len = left_data.shape[0]
            node.right_len = right_data.shape[0]
            node.left = tree_recurse(left_data, max_depth, depth + 1, visited + [best_feature])
            node.right = tree_recurse(right_data, max_depth, depth + 1, visited + [best_feature])
            node.type = 'inter'
        else:
            left_data, right_data = divide_data(train_data, -1)
            node.left_len = left_data.shape[0]
            node.right_len = right_data.shape[0]
            node.type = 'leaf'
            if node.left_len <= node.right_len:
                node.vote = 1
            else:
                node.vote = 0

    return node


# TODO Implement Predict Function
def predict(root, test_data):
    preds = []
    for example in test_data:
        pred = predict_sample(root, example)
        preds.append(pred)

    return preds


def predict_sample(root, example):
    curr_node = root
    while curr_node.type != 'leaf':
        if example[curr_node.attr] == 0:
            if curr_node.left is None:
                break
            curr_node = curr_node.left

        else:
            if curr_node.right is None:
                break

            curr_node = curr_node.right

    return curr_node.vote


# TODO Implement Function To Print Tree
def print_tree(node, idx2label, intend='| '):
    if node is None:
        return
    print('[%d 0/%d 1]' % (node.left_len, node.right_len))
    if node.attr is not None:
        print(intend + '%s = 0: ' % idx2label[node.attr], end='')
        print_tree(node.left, idx2label, intend=intend + '| ')
        print(intend + '%s = 1: ' % idx2label[node.attr], end='')
        print_tree(node.right, idx2label, intend=intend + '| ')


if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]
    train_data, idx2label = load_data(train_input)
    test_data, _ = load_data(test_input)
    root = train(train_data, max_depth)
    print_tree(root, idx2label)

    test_preds = predict(root, test_data)
    test_truth = test_data[:, -1]
    test_err = error(test_truth, test_preds)

    train_preds = predict(root, train_data)
    train_truth = train_data[:, -1]
    train_err = error(train_truth, train_preds)

    with open(train_out, 'w') as f:
        text = ''
        for pred in train_preds:
            text += str(pred) + '\n'
        f.write(text)

    with open(test_out, 'w') as f:
        text = ''
        for pred in test_preds:
            text += str(pred) + '\n'
        f.write(text)

    with open(metrics_out, 'w') as f:
        text = 'error(train): %f\nerror(test): %f' % (train_err, test_err)
        f.write(text)
