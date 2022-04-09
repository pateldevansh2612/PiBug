import torch
import sklearn, sklearn.datasets, sklearn.model_selection
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from matplotlib import pyplot as plt


# ======================================================================================================
# Learner
# ======================================================================================================

class LinearPotentials(torch.nn.Module):
    def __init__(self, input_dim, output_dim, random_weights_init = True):
        super(LinearPotentials, self).__init__()
        self.num_features = input_dim
        self.num_classes = output_dim
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.history = {'train_loss':[], 'test_loss': [], 'accuracy':[]}
        if random_weights_init == True:
            print("Should Implement random weights Init")
            # YOUR CODE HERE
            # ????????????
            # initialize the weights of the self.linear layer to be uniform [0,1]. Can be 1 line of code.
            # Bonus if you can set the bias to 0!
            # ???????????
            torch.nn.init.uniform(self.linear.weight, a=0,b=1)
            self.linear.bias.data.fill_(0)
            
    def forward(self, x):
        ''' DESCRIPTION: This class function is called everytime model(data) is called. Essentially
                         model.forward(data) = model(data). All layersthat operate in this function
                         are updated by the backpropagation step.
        '''
        outputs = self.linear(x)
        return outputs
    
    def plot_learning_curves(self, title=''):
        title = 'Learning Curves Plot' if title == '' else title
        fig = plt.figure(figsize=(14, 4))
        iters = np.arange(0, len(self.history['train_loss']))
        plt.plot(iters, self.history['train_loss'], linestyle='dashed',  label = 'Train Loss')
        plt.plot(iters, self.history['test_loss'],  linestyle='-',  label = 'Test Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.title(title)
        #plot(iters, self.history['train_loss'])
        plt.show()
        return fig
    
    def compute_error_stats(self):
        ''' DESCRIPTION: This class function computes the mean and std values from logged history
                         Make sure all test_accuracy, train_loss, test_loss are of equal size
        '''
        mean, std = 0,0

        mean_test_accuracy = np.mean(self.history['accuracy'])
        mean_train_loss = np.mean(self.history['train_loss'])
        mean_test_loss = np.mean(self.history['test_loss'])
        std_test_accuracy = np.std(self.history['accuracy'])
        print("Mean Accuracy: {}, std: {}".format(mean_test_accuracy, std))
        return mean_test_accuracy, std
        
# ======================================================================================================
# Functions
# ======================================================================================================


def plot(x_index, y_index, data):
    
    
    formatter = plt.FuncFormatter(lambda i, *args: data.target_names[int(i)])
    plt.scatter(data.data[:, x_index], data.data[:, y_index], c=data.target)
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.xlabel(data.feature_names[x_index])
    plt.ylabel(data.feature_names[y_index])
    
# -----------------------------------------------------------------------------------------------

def evaluate_untrained_model(model, data, criterion = torch.nn.CrossEntropyLoss()):
    ''' DESCRIPTION: This function handles the evaluation of an untrained model on a dataset. It expects a python dictionary for data and return 
                     loss and accuracy.
                     
        ARGUMENTS: model (nn module): Learner model. nn module type
                   data (dictionary): The train/test data provided a dictionary 
                                      data = {'x_train':torch.tensor, 'x_test': torch.tensor, 'y_train':torch.tensor, 'y_test': torch.tensor}  
    
                   criterion (nn loss): NN loss function. i.e CrossEntropyLoss.
                   
        RETURNS: loss (float)
                 accuracy (float)
    '''
    
    # Handle Inputs
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    # ---|
    
    y_raw_pred=model(x_test)                # get raw, unormilized likelihood from model output
    y_prob = torch.softmax(y_raw_pred, 1)   # use softmax to get probability
    y_pred = torch.argmax(y_prob, axis=1)   # The predicted class is essentially the one with highest pred probability
    loss=criterion(y_raw_pred,y_test)       # calculate loss
    print(y_test.shape, y_pred.shape)
    test_accuracy = (sum(y_pred==y_test)/y_test.shape[0]).detach().numpy()  # count how many pred were off, divide by number of items
    print("Untrained Model's Performance on Test Data\n==========================================\nLoss: {} | Accuracy: {}".format(loss, test_accuracy))
    
    return loss, test_accuracy

# -----------------------------------------------------------------------------------------------

def evaluate_model(model, data, optimizer, lr = 0.1, criterion = torch.nn.CrossEntropyLoss(), 
                   number_of_epochs = 10000, print_interval = 100, debug= False, print_plot=True):
    ''' DESCRIPTION: This function should facilitate training of a given model on a given dataset. It performs Gradient Descent
                     To iterativaly update the weights of the learning, attempting to minimize training loss, at each epoch.
                     The function will also log training progress in the model's history varaible (which is a python dicitionary)
                     and plot the learning curves at the end of the training process.
                     
        ARGUMENTS: model (nn module): Learner model. nn module type
                   data (dictionary): The train/test data provided a dictionary 
                                      data = {'x_train':torch.tensor, 'x_test': torch.tensor, 'y_train':torch.tensor, 'y_test': torch.tensor}  
                   optimizer(nn optim): Chose optimizer, i.e adam, SGD etc
    
                   criterion (nn loss): NN loss function. i.e CrossEntropyLoss.
                   number_of_epochs (int): How many epochs the model should be evaluated for.
                   print_interval (int):   Per how many iters should the script print out info.
                   print_plot (boolean):   Selects whether the learning curves will be plotted.
        RETURNS: train_log_loss (float)
                 test_log_loss (float)
                 test_accuracy (float)
    '''    
    # Handle Inputs
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    # ---|
    
    # Main training loop
    for epoch in range(number_of_epochs): 
        y_prediction=model(x_train)          # make predictions
        loss=criterion(y_prediction,y_train) # calculate losses
        model.history['train_loss'].append(loss.item()) # log progress to model's history
        loss.backward()                      # obtain gradients
        optimizer.step()                     # update parameters
        optimizer.zero_grad()                # reset gradients
    
        
        y_prob = torch.softmax(model(x_test), 1)
        y_pred = torch.argmax(y_prob, axis=1)

        train_log_loss = criterion(model(x_train), y_train).detach().numpy()
        test_log_loss = criterion(model(x_test), y_test).detach().numpy()
        test_accuracy = (sum(y_pred==y_test)/y_test.shape[0]).detach().numpy()
        
        model.history['test_loss'].append(test_log_loss.item()) # log progress to model's history
        model.history['accuracy'].append(test_accuracy)         # log progress to model's history
        if (epoch+1)%print_interval == 0:                       # every print_interval iters print loss
            print('Epoch:', epoch+1,',loss=',loss.item())
    # end of main training loop
    #--- |
    
    # Print model parameters; only if debug is enabled.
    if debug == True:
        for param in model.named_parameters():
            print("Param = ",param)         
    # Print last epoch's loss and acc
    print("Train Log Loss = ", train_log_loss)
    print("Test Log Loss  = ", test_log_loss)
    print("Test Accuracy  = ", test_accuracy) 
    if print_plot == True:
        model.plot_learning_curves()
        
    # Do not change return types.
    return train_log_loss, test_log_loss, test_accuracy

# -----------------------------------------------------------------------------------------------

def evaluate_error_metrics(model, data, optimizer, criterion = torch.nn.CrossEntropyLoss(), 
                           number_of_experiments = 100, number_of_epochs = 30000, 
                           print_interval = 1000, debug = False):
    ''' DESCRIPTION: This function should initialize a model with different random parameters over several experiments and log its performance
                     during training. It should then compute the mean error and std of said error across all run experiments.
                     It should return those metrics
                     
        ARGUMENTS: model (nn module): Learner model. nn module type. To be used as a template for model initialization.
                   data (dictionary): The train/test data provided a dictionary 
                                      data = {'x_train':torch.tensor, 'x_test': torch.tensor, 'y_train':torch.tensor, 'y_test': torch.tensor}  
                   optimizer(nn optim): Chosen optimizer, i.e adam, SGD etc
    
                   criterion (nn loss): NN loss function. i.e CrossEntropyLoss.
                   number_of_experiments(int): The number of different models that will be trained and evaluated. The mean and std should be
                                               computed over all returned results.
                   number_of_epochs (int): How many epochs EACH model will be evaluated for.
                   print_interval (int):   Per how many iters should the script print out info.
                   
        RETURNS: mean_error (float)
                 error_std (float)
    ''' 
    mean_error, error_std = .0,.0
    # YOUR CODE HERE
    # ??????????????
    
    # You can create an optimizer given an existing one like this:
    # curr_optim = type(optimizer)(cur_model.parameters(),lr=lr)
    # the above creates a new optimizer that tracks the input cur_model's parameters; so input the model you want to train.
    
    # Write the loop that will  run for number_of_experiments iters
    
    # Instantiate a model like the given one and evaluate it calling the evaluate function (with the NEW optimizer and model)
    # Get and log the results. compute the mean and std of the ERROR (not the accuracy) over all the number_of_experiments results!
    
    # Compute the mean and std here after all the results are logged. Refer to the LinearPotentials class for insperation
    # on how to compute the metrics from a list or a dictionary.
    # ??????????????
    
    return mean_error, error_std

# -----------------------------------------------------------------------------------------------

def evaluate_with_model_regularization(model, data, optimizer, criterion = torch.nn.CrossEntropyLoss(), 
                                       reg_lambda = 0.01, reg_type = 'l1',
                                       number_of_epochs = 30000, print_interval = 1000, print_plot=True, debug = False):
    ''' DESCRIPTION: This function should facilitate training of a given model on a given dataset. It behaves itentically to eval
                     evaluate_model, whith the addition that it should also a add regulirization term to the training objective 
                     according to the input. It performs Gradient Descent
                     to iterativaly update the weights of the learning, attempting to minimize training loss, at each epoch.
                     The function will also log training progress in the model's history varaible (which is a python dicitionary)
                     and plot the learning curves at the end of the training process.
                     
        ARGUMENTS: model (nn module): Learner model. nn module type
                   data (dictionary): The train/test data provided a dictionary 
                                      data = {'x_train':torch.tensor, 'x_test': torch.tensor, 'y_train':torch.tensor, 'y_test': torch.tensor}  
                   optimizer(nn optim): Chosen optimizer, i.e adam, SGD etc
    
                   criterion (nn loss): NN loss function. i.e CrossEntropyLoss.
                   reg_lambda (float):  lamda parameter for the regulirization term
                   reg_type (string):   'l1' or 'l2': selector of l1 or l2 regularization
                   number_of_epochs (int): How many epochs the model should be evaluated for.
                   print_interval (int):   Per how many iters should the script print out info.
                   
        RETURNS: train_log_loss (float)
                 test_log_loss (float)
                 test_accuracy (float)
    '''    
    # Handle Inputs
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    # ---|
    
    # Main training loop
    for epoch in range(number_of_epochs): 
        y_prediction=model(x_train)          # make predictions
        loss=criterion(y_prediction,y_train) # calculate losses

        # Compute the L1 and L2 penalty of parameters and add to the loss
        # YOUR CODE HERE 
        # ??????????????
        # Compute L1 or L2 (depending on argument reg_type)
        # add the comptued regularization term to loss
        # name the reg term: l_penalty
        # ??????????????
        
        loss.backward()                      # obtain gradients
        optimizer.step()                     # update parameters
        optimizer.zero_grad()                # reset gradients
        if (epoch+1)%print_interval == 0:
            print("Epoch: {}, {}-loss: {}".format(epoch+1,reg_type, loss.item()))
        
        # Get predictions on the test set
        y_prob = torch.softmax(model(x_test), 1)
        y_pred = torch.argmax(y_prob, axis=1)
        train_log_loss = loss
        test_log_loss = (criterion(model(x_test), y_test) +l_penalty).detach().numpy()
        test_accuracy = (sum(y_pred==y_test)/y_test.shape[0]).detach().numpy()
        # Log all loss progress
        model.history['train_loss'].append(loss.item())
        model.history['test_loss'].append(test_log_loss.item())
        model.history['accuracy'].append(test_accuracy)
    # end of training loop    
    # ---|    
    # Print model parameters
    if debug == True:
        for param in model.named_parameters():
            print("Param = ",param) 
    if print_plot == True: 
        fig = model.plot_learning_curves()
    # Print last epoch's metrics
    print("Train Log Loss = ", train_log_loss)
    print("Test Log Loss  = ", test_log_loss)
    print("Test Accuracy  = ", test_accuracy) 
    
    # Do not change return types.
    return test_accuracy, model



# -----------------------------------------------------------------------------------------------
def learning_rate_iter_explore(model, data, optimizer, lr_range = [0.99],
                               criterion = torch.nn.CrossEntropyLoss(), number_of_epochs = 10000, 
                               print_interval = 1000, print_plot = True):
    ''' DESCRIPTION: This function should explore the trade off between learning rate and iterations of training.
                     It should iteratively train a model on a given learing rate and mark the best performance on the test
                     set for a given number of iterations. It should report the best learning rate-iteration combo.
                     
        ARGUMENTS: model (nn module): Learner model. nn module type
                   data (dictionary):  The train/test data provided a dictionary 
                                       data = {'x_train':torch.tensor, 'x_test': torch.tensor, 'y_train':torch.tensor, 'y_test': torch.tensor}  
                   optimizer(nn optim): Chose optimizer, i.e adam, SGD etc
                    
                   lr_range (list): list containg the learning rates to be explored. Can be a single value.
                   criterion (nn loss): NN loss function. i.e CrossEntropyLoss.
                   number_of_epochs (int): How many epochs each model should be evaluated for.
                   print_interval (int):   Per how many iters should the script print out info.
                   print_plot (boolean):   If selected, the learning curves for each trained model will be printed.
                   
        RETURNS: returnDict (python Dicitonary): Dictionary {'train_log_loss':[], 'test_log_loss':[], 'test_accuracy':[],
                                                             'learning_rates':[], 'iters_for_convergence':scalar} 
                                                Should only return the values, in the above format of the BEST model. ONLY.             
    '''
    # Input sanitization: make sure lr_range is a list!
    if not isinstance(lr_range, list):
        lr_range = [lr_range]
        
    best_lr, best_iters = 0,0
    
    # This PyThon Dicitonary holds the required metrics. Each key points to a python list that holds the apporpiate metric.
    # i.e key 'train_log_loss' points to a list that holds all train losses stored (each new value is appended to the end)
    resultsDict = {'train_log_loss':[], 'test_log_loss':[], 'test_accuracy':[], 'learning_rates':[]} 
    
    for i, lr in enumerate(lr_range):
        # Create a copy of the original model, with the same weights and evaluate it.
        cur_model = LinearPotentials(model.num_features, model.num_classes) # Instantiate a new model here and copy the desired parameters below.
        cur_model.load_state_dict(model.state_dict().copy())       # This is generaly how PyTorch handles parameter copy between models
        print("Evaluating Model with learning rate {}; Starting params: {}".format(lr, cur_model.linear.weight))
        curr_optim = type(optimizer)(cur_model.parameters(),lr=lr) # This is how we can make a new object based on the type of an input one in Python
        train_log_loss, test_log_loss, test_accuracy = evaluate_model(cur_model, data, curr_optim, 
                                                                      criterion = torch.nn.CrossEntropyLoss(), print_plot = print_plot,
                                                                      print_interval=print_interval, number_of_epochs = number_of_epochs)
        resultsDict['train_log_loss'].append(float(train_log_loss)) 
        resultsDict['test_log_loss'].append(float(test_log_loss))
        resultsDict['test_accuracy'].append(float(test_accuracy))
        resultsDict['learning_rates'].append(float(lr))
        resultsList.append([float(train_log_loss), float(test_log_loss), float(test_accuracy)])
    # YOUR CODE HERE    
    # ??????????????
    returnDict  = {} # make this have the same key-value structure as resultsDict plus an 'iters_for_convergence': scalar int and 'best_lr': scalar float
    # which will be the iteration number that this model could have converged and best lr you found (HINT: if the loss drop from one iter to the next is very small...)
    # remember cur_model.history is a dictionary containing all the training history of THAT model for all training epochs(see LinearPotenitals class)
    # return the best lr rate you found in term of performance as a dictionary Identical to resultsDict with the extra key-values mentioned above.
    # returnDIct should only have the info of the best model you found (all of its training history), in the format described above.
    # ??????????????
    
    return returnDict # remember to include key-values : 'iters_for_convergence': int scalar and 'best_lr': float scalar

# -----------------------------------------------------------------------------------------------

def svm_model_comparison(data, criterion = torch.nn.CrossEntropyLoss(), 
                   number_of_epochs = 10000, print_interval = 1000, print_plot = True):
    ''' DESCRIPTION: This function should fit an SVM on the train data and targets and test it
                     on the Y data and targets. Report accuracy on test set
                     
        RETURNS test_accuracy (float)
                
    '''
    my_svm_train_loss, my_svm_test_loss, my_svm_test_accuracy, sklearn_svm_test_accuracy = 0,0,0, 0
    
    # Handle Inputs
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    # Get data info
    num_features, num_classes = x_train.shape[1], y_train.max()+1
    # ---|
    
    # YOUR CODE HERE
    # ??????????????
    #  Create your model here and get its performance
    #model = LinearPotentials(num_features, num_classes, random_weights_init = True)
    #criterion = which loss?
    #optimizer = which optimizer?
    
    print("\nTraining Custom SVM to training data!!")
    print("========================================")
    #my_svm_train_loss, my_svm_test_loss, my_svm_test_accuracy = how to evaluate this mode? Do we have a tool?

    print("Custom SVM Test Accuracy  = ", my_svm_test_accuracy)
    # Instantiate an SVM based on libsvm
    #
    print("\nFitting sklearn's SVM to training data!!")
    print("==========================================")
    # Train model on train data
    # 
    # Need to transform output to PyTorch tensor (SVM returns ndarray), consult the pytorch manuals on how to conver ndarray to tensor
    # Find accuracy in the same way we did in the evaluate function. DOes sklearn's SVM return predicted classes probs or raw likelihoods over all classes?
    # ??????????????
    
    print("SVM Test Accuracy  = ", sklearn_svm_test_accuracy)
       
    return my_svm_test_accuracy, sklearn_svm_test_accuracy

# ======================================================================================================
# MAIN
# ======================================================================================================
def main():
    
    # Part 0
    # Load and visualize data.
    iris = sklearn.datasets.load_iris()
    x,y = iris.data,iris.target

    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    plot(0, 1, iris)
    plt.subplot(122)
    plot(2, 3, iris)
    plt.show()
    
    # split data into training and testing sets
    x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)

    # convert to tensors for Pytorch
    x_train = torch.from_numpy(x_train.astype(np.float32))
    x_test  = torch.from_numpy(x_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.int_))
    y_test  = torch.from_numpy(y_test.astype(np.int_))
    data    = dict(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    
    # Get data info
    num_features, num_classes = x_train.shape[1], y.max()+1
    # ---|
    
    # Part 1
    # Declare learner model, loss objective, optimizer
    # Initialize learner with random weights
    model = LinearPotentials(num_features, num_classes, random_weights_init = True)
    criterion = torch.nn.CrossEntropyLoss() # torch.nn.MultiMarginLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1, momentum=0.1) # torch.optim.Adam(model.parameters(), lr=0.1)
    # Func1_call
    # ---|
    
    # Part 2
    # ------------
    # Given a learning rate of 0.1 train on train data and report accuracy on test data
    # Func2_call 
    # ---|
    
    # Part 3
    # ------------
    # Evaluate 100 instances of the model above and report mean error rate and error std.
    # Func3_call 
    # ---|
    
    # Part 4 
    # A. Repeat steps 2,3 while adding L1 regularization to the objective
    # Func4_call 
    #evaluate_with_model_regularization(model, data, optimizer, criterion = torch.nn.CrossEntropyLoss(), number_of_epochs = 1000)
    # B. Repeat steps 2,3 while adding L2 regularization to the objective
    # ---|
    
    # Part 5
    # Explore different learning rates. For a given learning rate collection, report the lr-vs-iteration tradefoff, That is
    # for each learning rate, plot the test loss vs accuracy and report on where you believe the training has converged to a solution.
    # Func5_call 

    
    # Part 6
    # SVM deployment. Using sklearn's SVM implmentation, train an SVM with a radial basis function kernel on the training set and report 
    # its performance in terms of Accuracy on the test set
    # # Func6_call 
    
if __name__ == "__main__":
    main()