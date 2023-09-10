# imports
import numpy as np
import pandas as pd
import seaborn as sea
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing

df = pd.read_csv("/home/adrian/codingProjects/privateGit/machineLearningPractice/decisionTree/drug200.csv")

# did I miss spell decision?
# decisionTree code

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    # Determine if max depth of tree has been reached.
    def _is_finished(self, depth):
        if (depth >= self.max_depth or
            self.n_class_lables == 1 or
            self.n_samples < self.min_samples_split):
            return True
        return False
    
    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        # entropy equation
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy

    def _create_split(self, X, thresh):
        #debugging
        X = X.to_frame()
        
        #print('X: ', X) 
        #print('X type: ', type(X))
        #print('thresh: ', thresh)
        #print('thresh type: ', type(thresh))
        #print('X.shape', X.shape)
        
        left_idx = np.argwhere(X <= thresh).flatten() # Problem here
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def _information_gain(self, X, y, thresh):
        parent_loss = self._entropy(y)
        left_idx, right_idx = self._create_split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        # Information gain equation. 
        child_loss = (n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
        return parent_loss - child_loss

    def _best_split(self, X, y, features):
        split = {'score': -1, 'feat': None, 'thresh': None}
        
        # debugging
        #print('_best_split')
        #print('X: ', X)
        #print('X type: ', type(X))
        
        for feat in features:
            #X_feat = X[:, feat] # was orignally this
            X_feat = X.iloc[:, feat] # problem maybe here? Most likly
            
            #print('after X.iloc X_feat: ', X_feat)
            #print('type: ', type(X_feat))
            
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)
                
                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh
        
        return split['feat'], split['thresh']
            
    
    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_lables = len(np.unique(y))
        
        # stopping criteria
        if self._is_finished(depth):
            # debugging
            print('depth: ', depth)
            print('y :', y)

            if len(y) == 0:
                most_common_lable = 'Error Unknown'
                print('most_common_lable: ', most_common_lable)
                return Node(value = most_common_lable)

            most_common_lable = np.argmax(np.bincount(y))
            print('most_common_lable: ', most_common_lable)
            return Node(value = most_common_lable)
        
        # get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)
        
        # grow children recursively
        left_idx, right_idx = self._create_split(X.iloc[:, best_feat], best_thresh)
        left_child = self._build_tree(X.iloc[left_idx, :], y[left_idx], depth + 1) 
        right_child = self._build_tree(X.iloc[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)
    
    def _traverse_tree(self, X, node):
        if node.is_leaf():
            return node.value

        # debugging
        #print('traverse tree, X: ', X)
        #print('X type: ', type(X))
        #print('node.feature: ', node.feature)

        if X[node.feature] <= node.threshold:
            return self._traverse_tree(X, node.left)
        return self._traverse_tree(X, node.right)
     
    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict(self, X):
        #debugging
        #print('predict X: ', X)
        #print('X type: ', type(X)) 

        predictions = [self._traverse_tree(x, self.root) for i, x in X.iterrows()]
        return np.array(predictions)    

# Split the data
x = df.copy()
y = df['Drug']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33)
x_train = x_train.drop(['Drug'], axis = 1)
x_test = x_test.drop(['Drug'], axis = 1)

# Encode the catagroical data
# For this small amount of data we can use one-hot encoding to encode the catagroical data
def oneHotEncodingMuiltiple(dataFrame, columnsList):
    transformer = make_column_transformer((OneHotEncoder(), columnsList), remainder='passthrough', verbose_feature_names_out=False)
    transformed = transformer.fit_transform(dataFrame)
    dataFrame = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())
    return dataFrame

x_train = oneHotEncodingMuiltiple(x_train, ['Sex'])
x_test = oneHotEncodingMuiltiple(x_test, ['Sex'])

def ordinalEncodingMuiltiple(dataFrame, columnList):
    transformer = make_column_transformer((OrdinalEncoder(), columnList), remainder='passthrough', verbose_feature_names_out=False)
    transformed = transformer.fit_transform(dataFrame)
    dataFrame = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())
    return dataFrame

x_train = ordinalEncodingMuiltiple(x_train, ['BP', 'Cholesterol'])
x_test = ordinalEncodingMuiltiple(x_test, ['BP', 'Cholesterol'])

# Encode target variable
le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)

y_test = le.transform(y_test)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true) # accuracy formula
    return accuracy

dt = DecisionTree(max_depth=7)
dt.fit(x_train, y_train)

# See results of model
y_pred = dt.predict(x_test)
acc = accuracy(y_test, y_pred)
print('acc: ', acc)

# code now sorta work, but not sure if its really right... 







