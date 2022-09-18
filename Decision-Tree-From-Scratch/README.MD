# Decision Tree Implementation From Scratch
**作业要求见ass.pdf**
比较基础的决策树实现问题

## Requirements
- python >= 3
- numpy

You can easily download or clone this repository run `decision_tree.py` with commands below in terminal.

```shell
python decision_tree.py <train input> <test input> <max depth> <train out> <test out> <metrics out>

# For exam:
python decision_tree.py heart_train.tsv heart_test.tsv 2 heart_2_train.txt heart_2_test.txt heart_2_metrics.txt
 
```
For args:
1. `<train input>` path to the training input .tsv file 
2. `<test input>` path to the test input .tsv file 
3. `<max depth>` maximum depth to which the tree should be built
4. `<train out>` path of output .txt file to which the predictions on the training data should be written 
5. `<test out>` path of output .txt file to which the predictions on the test data should be written 
6. `<metrics out>` path of the output .txt file to which metrics such as train and test error should be written

## Some details
 The decision tree I implemented uses mutual information as a metric to 
 select feature segmentation points, and uses a recursive method to build 
 and print the entire decision tree.