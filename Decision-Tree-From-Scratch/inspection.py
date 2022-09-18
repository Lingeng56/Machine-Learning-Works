import sys
import csv
from statistics import mode
import math


def train(label):
    label.sort(reverse=True)

    majority_vote = mode(label)
    # majority_vote = max(set(label), key=label.count)

    return majority_vote


def predict(train_label, majority_vote):
    n_sample = len(train_label)
    train_predict = [None] * n_sample

    for i in range(n_sample):
        train_predict[i] = majority_vote

    return train_predict


def error(label, prediction):
    n_sample = len(label)
    n_error = 0.0

    for i in range(n_sample):
        if label[i] != prediction[i]:
            n_error += 1

    error_rate = n_error / n_sample

    return error_rate


def entropy(num_class0, num_class1, n_sample):
    entropy = 0.0
    fraction0 = num_class0 / n_sample
    # print(fraction0)
    fraction1 = num_class1 / n_sample
    # print(fraction1)
    entropy = - (fraction0) * math.log(fraction0, 2) - (fraction1) * math.log(fraction1, 2)

    return entropy


if __name__ == '__main__':
    input = sys.argv[1]
    output = sys.argv[2]
    #       print("The input file is: %s" % (input))

    feature = []
    label = []

    num_class0 = 0
    num_class1 = 0
    n_sample = 0

    with open(input, 'r', newline='') as input_file:
        next(input_file)
        reader_train = csv.reader(input_file, delimiter='\t')
        for r in reader_train:
            feature.append(r[1:(len(r) - 1)])
            label.append(r[-1])
            if label[-1] == '0':
                num_class0 += 1
            else:
                num_class1 += 1

    n_sample = len(label)
    # print(num_class0, num_class1, n_sample)
    # for i in train_data:
    #         print(i)
    # print(feature)
    # print(len(label))

    majority_vote = train(label)
    # print(majority_vote)

    predict = predict(label, majority_vote)
    # print(len(predict))

    # ---------------------------------------------------------------------------------------------

    entropy = entropy(num_class0, num_class1, n_sample)
    # print(entropy)

    error = error(label, predict)
    # print(error)

    with open(output, "w") as output_file:
        output_file.write(
            "entropy: " + "{:.6f}".format(round(entropy, 6)) + "\n" + "error: " + "{:.6f}".format(round(error, 6)))
