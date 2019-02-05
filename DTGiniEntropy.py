
#   The program creates 2 Binary Decision Trees, the first using Gini Index and
#   the second using Information gain. The same test data and train data is used.
#   The decision trees created using both methods for the same training dataset is displayed.
#   Then incorrectly classified rows thru both methods is displayed as well as
#   accuracy of each method is displayed.

import csv
import math

#  Class Definitions

#  This class is used to store the condition for the decision nodes and also to split the dataset.
class Condition:

    # The condition comprises of the column and the value to be compared to.
    def __init__(self, column, value):
        self.column = column
        self.value = value

    # returns true or false based on if the row's particular column value satisfies the condition or not.
    def check(self, row):
        return row[self.column] == self.value

    def __repr__(self):
        return "Check if %s == %s ?" % (header[self.column], str(self.value))



#  A Decision Node contains the condition by which to direct the data to the next node.
class DecisionNode:

    def __init__(self, condition, true_branch, false_branch):
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch



#  The Leaf Node predicts the class of the dataset/row that reaches it.
class LeafNode:

    def __init__(self, dataset):
        self.predicted_class = dataset[0][-1]
        self.count = len(dataset)


#  Function definitions

#  Read the file and return the dataset as a list of lists.
def read_file(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        dataset = list(reader)
  #      print(train_data[5:])
        return dataset



# Returns a dictionary of label classes and their count (how many rows belong to each label class)
def label_value_counts(dataset):

    class_values = list(set(rec[-1] for rec in dataset))
    if len(class_values) == 1:
        return dict([(class_values[0], len(dataset))])
    class_values.sort()                           # to maintain the order of the elements.
    class_counts = dict.fromkeys(class_values, 0)
    for each_row in dataset:
        class_value = each_row[-1]
        class_counts[class_value] += 1
  #  print("counts: ", class_counts)
    return class_counts



# Split the dataset into 2, the rows that match go into the true_dataset and the rest go to false_dataset.
def split(dataset, condition):

    true_dataset, false_dataset = list(), list()

    for each_row in dataset:
        if condition.check(each_row):
            true_dataset.append(each_row)
        else:
            false_dataset.append(each_row)
    return true_dataset, false_dataset


#  Gini Index Functions

#  Calculate the gini_index at a node
def gini_index(dataset):
    gini = 1
    len_dataset = len(dataset)
    class_counts = label_value_counts(dataset)
    for each_label in class_counts:
        probabilty = class_counts[each_label] / float(len_dataset)
        gini -= probabilty**2
    return gini



#  Return the gini_split at a node by looking at its child nodes
def gini_split(true_node, false_node):

    prob = float(len(true_node)) / (len(true_node) + len(false_node))

    return prob * gini_index(true_node) + (1 - prob) * gini_index(false_node)



#  Determine the best decision condition at the current node using gini_split calculated for each condition
def best_decision_condition_gini(dataset):

    current_node_gini = gini_index(dataset)

    min_gini_cond = None
    min_gini = 9

    # if all rows belong to one class, no point of processing further.
    if current_node_gini == 0:
     #   print("Get out!")
        return min_gini_cond, min_gini

    no_of_features = len(dataset[0]) - 1

    # find the min gini index by trying every feature value in each feature of the dataset.
    for feature in range(no_of_features):

        # get the set of feature values for the feature.
        feature_values = set([row[feature] for row in dataset])

        for each_value in feature_values:  # for each value

            condition = Condition(feature, each_value)

            true_dataset, false_dataset = split(dataset, condition)

            # Skip this condition if it doesn't divide the dataset
            if len(true_dataset) == 0 or len(false_dataset) == 0:
              #  print("dataset empty")
                continue

            # Calculate the gini_split from this condition
            gini = gini_split(true_dataset, false_dataset)
        #    print(gini)

            if gini < min_gini:
                min_gini, min_gini_cond = gini, condition

    return min_gini_cond, min_gini



#  Build the tree recursively by finding the best condition to split the dataset and making that condition the node
def build_decision_tree_gini(dataset):

    condition, min_gini = best_decision_condition_gini(dataset)
  #  print("gain :", gain)

    # If gain is 0, all rows belong to the same label and no need to check further, they form a leaf node.
    if min_gini == 9:
        return LeafNode(dataset)

    # Split the dataset into 2 such that one satisfies the condition and the other not.
    true_dataset, false_dataset = split(dataset, condition)


    true_branch = build_decision_tree_gini(true_dataset)

    false_branch = build_decision_tree_gini(false_dataset)

    # create and return a decision node with the best condition and its branches
    return DecisionNode(condition, true_branch, false_branch)


# Information gain functions


#  Calculate the entropy at a node
def cal_entropy(dataset):
    entropy = 1
    len_dataset = len(dataset)
    class_counts = label_value_counts(dataset)
    for each_label in class_counts:
        ratio = class_counts[each_label] / float(len_dataset)
        entropy = - ratio * math.log2(ratio)
    return entropy



#  Return the info_gain at a node by looking at its child nodes
def info_gain(true_node, false_node):

    prob = float(len(true_node)) / (len(true_node) + len(false_node))

    return prob * cal_entropy(true_node) + (1 - prob) * cal_entropy(false_node)




#  Determine the best decision condition at the current node using info_gain calculated for each condition
def best_decision_condition_info_gain(dataset):

    current_node_entropy = cal_entropy(dataset)

    max_gain_cond = None
    max_gain = 0

    # if all rows belong to one class, no point of processing further.
    if current_node_entropy == 0:
     #   print("Get out!")
        return max_gain_cond, max_gain

    no_of_features = len(dataset[0]) - 1

    # find the max information gain by trying every feature value in each feature of the dataset.
    for feature in range(no_of_features):

        # get the set of feature values for the feature.
        feature_values = set([row[feature] for row in dataset])

        for each_value in feature_values:  # for each value

            condition = Condition(feature, each_value)

            true_dataset, false_dataset = split(dataset, condition)

            # Skip this condition if it doesn't divide the dataset
            if len(true_dataset) == 0 or len(false_dataset) == 0:
              #  print("dataset empty")
                continue

            # Calculate the info_gain from this condition
            gain = current_node_entropy - info_gain(true_dataset, false_dataset)
        #    print(gini)

            if gain  > max_gain:
                max_gain_cond, max_gain = condition, gain

    return max_gain_cond, max_gain



#  Build the tree recursively by finding the best condition to split the dataset and making that condition the node
def build_decision_tree_info_gain(dataset):

    condition, max_gain = best_decision_condition_info_gain(dataset)
  #  print("gain :", gain)

    # If gain is 0, all rows belong to the same label and no need to check further, they form a leaf node.
    if max_gain == 0:
        return LeafNode(dataset)

    # Split the dataset into 2 such that one satisfies the condition and the other not.
    true_dataset, false_dataset = split(dataset, condition)


    true_branch = build_decision_tree_info_gain(true_dataset)

    false_branch = build_decision_tree_info_gain(false_dataset)

    # create and return a decision node with the best condition and its branches
    return DecisionNode(condition, true_branch, false_branch)


#  Print the Decision Tree recursively
def print_tree(node, spacing=" "):

    if type(node) == LeafNode:
        print(spacing + "( Class: '%s' )" % node.predicted_class)
        return

    print(spacing, "[", str(node.condition), "]")

    print(spacing + ' True:')
    print_tree(node.true_branch, spacing + "    ")

    print(spacing + ' False:')
    print_tree(node.false_branch, spacing + "    ")



#  Proceed through the tree recursively till you reach a Leaf node.
def classify(node, row):

    # check if we reached a leaf node
    if type(node) == LeafNode:
        return node.predicted_class

    if node.condition.check(row):
        return classify(node.true_branch, row)
    else:
        return classify(node.false_branch, row)



if __name__ == '__main__':

    train_file = 'car.training.csv'

    train_dataset = read_file(train_file)

    header = ["buying", "maintenance", "persons", "doors", "boot", "safety", "label"]

    decision_tree_gini = build_decision_tree_gini(train_dataset)

    print(" Training file : ", train_file)

    print(" ***************   Decision Tree Symbols   *****************")
    print(" [....]  - Decision Node")
    print(" (....)  - Leaf Node")
    print(" *******************  ******************  *******************\n\n")
    print(" ******************     Decision Tree using Gini Index   ********************", "\n")

    print("Root Node")
    print_tree(decision_tree_gini)

    print(" \n\n*******************  ******************  *******************\n\n")

    decision_tree_info_gain = build_decision_tree_info_gain(train_dataset)
    print(" ******************     Decision Tree using Information gain   ********************", "\n")

    print("Root Node")
    print_tree(decision_tree_info_gain)
    print(" \n\n*******************  ******************  *******************\n\n")

    test_file = 'car.test.csv'


    test_dataset = read_file(test_file)

    accuracy_counter_gini = 0
    print()
    print(" Test file : ", test_file)
    print(" Test data Classification using Gini Index ")

    for row in test_dataset:
        pred_label = classify(decision_tree_gini, row)
        print("\n", row, pred_label, end='')
        if row[-1] == pred_label:
            accuracy_counter_gini += 1
        else:
            print("  *** Misclassified *** ", end='')

    print("\n\n Test data Classification using Information Gain ")

    accuracy_counter = 0
    for row in test_dataset:
        pred_label = classify(decision_tree_info_gain, row)
        print("\n", row, pred_label, end='')
        if row[-1] == pred_label:
            accuracy_counter += 1
        else:
            print("  *** Misclassified *** ", end='')

    dataset_count = len(test_dataset)


    print("Total data count     =", dataset_count)

    print("\n\nCorrectly classified using gini index =", accuracy_counter_gini)
    print("Gini Index Decision Tree accuracy = ", float(accuracy_counter_gini) * 100/dataset_count, "%")

    print("\n\nCorrectly classified using information gain =", accuracy_counter)
    print("Information Gain Decision Tree accuracy = ", float(accuracy_counter) * 100/dataset_count, "%")


