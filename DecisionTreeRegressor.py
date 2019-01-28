import numpy as np
import os
import json
import operator

class MyDecisionTreeRegressor():
    def __init__(self, max_depth=5, min_samples_split=1):
        '''
        Initialization
        :param max_depth: type: integer
        maximum depth of the regression tree. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split: type: integer
        minimum number of samples required to split an internal node:

        root: type: dictionary, the root node of the regression tree.
        '''

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)

        You should update the self.root in this function.
        '''
        #append y to the last column of X
        input_X = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        #call split tree
        temp_root = self.split_tree(input_X, 1)
        self.root = temp_root
#         print("")
#         print("final root:",self.root)
        
        return self.root

    def predict(self, X):
        '''
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        '''
        #self.root
        y_predict = np.zeros(X.shape[0])
        
        #predict for each row
        for i in range(len(X)):
            y_predict[i] = self.traverse_tree(X[i], self.root)
        
        return y_predict

    def get_model_string(self):
        model_dict = self.root
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = self.root
        with open(file_name, 'w') as fp:
            json.dump(model_dict, fp)

    def traverse_tree(self, x, node):
        y = None
        index = node['splitting_variable']
        if x[int(index)] <= node['splitting_threshold']:
            child = node["left"]
        else:
            child = node["right"]
        
        if type(child) is dict:
            y = self.traverse_tree(x, child)
        else:
            y = child

        return y
        
    
    def split_tree(self, X, depth):
#         print("Depth:",depth)
#         print("X_shape:",X.shape)
        num_records,features = X.shape[0], X.shape[1]-1
        temp_root = {}

        j = 0
        s = 0.0
        
        c1, c2 = 0,0
        
        N = num_records
        temp_error = 0
        temp_error_1 = 0
        final_split_point = -1
        temp_split_point= -1
        for j in range(features):
#             print("-------------j:",j)
            #extract the j-th column of X and append with y
            X_j = np.concatenate((X[0:,j].reshape(-1, 1), X[0:,features].reshape(-1, 1)), axis=1)
            
            # sort based on selected attribute column only
            X_j = X_j[X_j[:,0].argsort()]
            
            #select the s with the smallest error
            for i in range(N-1):
                
                split_point = i+1
                R = np.split(X_j, [split_point], axis = 0)
                R1 = R[0]
                R2 = R[1]

                c1 = np.average(R1[:,1])
                c2 = np.average(R2[:,1])
                
                # sum of y_i - c
                error = round(np.sum(pow((R1[:,1] - c1),2)) + np.sum(pow((R2[:,1] - c2),2)),15)
                
                #Always keep the smallest error of each s in temp_error
                if i == 0 or temp_error > error:
                    temp_error = error
                    temp_split_point = split_point
                    s = X_j[i,0]
        
            #Always keep the smallest error of each j in temp_error_1
            #select j with the smallest error
            if j == 0 or temp_error_1 > temp_error:
                j_selected = j
                s_selected = s
                temp_error_1 = temp_error
                final_split_point = temp_split_point

        #sort X based on the selected feature column only
        X_sorted = X[X[:,j_selected].argsort()]
        
        #split X_sorted to two regions with the selected s
        regions = np.split(X_sorted, [final_split_point], axis = 0)        
        region_1 = regions[0]
        region_2 = regions[1]
        
        #decide if split or take value of the left and right child
        if len(region_1) >= self.min_samples_split and depth < self.max_depth:
            left = self.split_tree(region_1, depth + 1)
        else:
            left = np.average(region_1[:,features])          
            
        if len(region_2) >= self.min_samples_split and depth < self.max_depth:
            right = self.split_tree(region_2, depth + 1)     
        else:
            right = np.average(region_2[:,features])
                      
        temp_root['splitting_variable'] = j_selected
        temp_root['splitting_threshold'] = s_selected
        temp_root['left']= left
        temp_root['right']= right
#         print(temp_root)
        
        return temp_root


# For test
if __name__=='__main__':
    for i in range(3):
        x_train = np.genfromtxt("Test_data" + os.sep + "x_" + str(i) +".csv", delimiter=",")
        y_train = np.genfromtxt("Test_data" + os.sep + "y_" + str(i) +".csv", delimiter=",")

        for j in range(2):
            tree = MyDecisionTreeRegressor(max_depth=5, min_samples_split=j + 2)
            tree.fit(x_train, y_train)

            model_string = tree.get_model_string()

            with open("Test_data" + os.sep + "decision_tree_" + str(i) + "_" + str(j) + ".json", 'r') as fp:
                test_model_string = json.load(fp)

            print(operator.eq(model_string, test_model_string))

            y_pred = tree.predict(x_train)

            y_test_pred = np.genfromtxt("Test_data" + os.sep + "y_pred_decision_tree_"  + str(i) + "_" + str(j) + ".csv", delimiter=",")
            print(np.square(y_pred - y_test_pred).mean() <= 10**-10)

