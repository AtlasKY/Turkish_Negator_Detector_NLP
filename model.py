import fasttext
import torch
import time
import random
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam, SGD
import numpy as np
import matplotlib.pyplot as plt

from net import Net


class NegationCheck(object):
    
    
    #Initialise the model and the fasttext object
    #Load the fasttext word embdding from the path provided 
    #that has ft_wv_dim dimension word vectors
    def __init__(self, ft_path, ft_wv_dim):
        self.ft = fasttext.load_model(ft_path)
        self.dim = ft_wv_dim
        self.model = Net(self.dim)
        self.model = self.model.float()
        self.model.eval()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.loss = MSELoss()
    
    
    #Load pre-trained weights from .pth or .pt file
    def load_model(self, path_to_model):
        self.model.load_state_dict(torch.load(path_to_model))
        return
    
    #Save the current weights to the path with .pth or .pt extension
    def save_model(self, path_to_save):
        torch.save(self.model.state_dict(), path_to_save)
        return
    
    #Given the input string, returns the vectorized representations obtained from the embeddings of the fasttext model
    #sentences: a batch of strings that are the inputs to the model
    #in_vector: [batch_size, num_inp_words, 100, 1] Tensor - reshape the vectors to 
    #           (100 x 1) matrices and stack them as channels
    def input_vectors(self, sentences: list):

        max_num_words = 4
        tokens = []
        batch_size = len(sentences)
        
        h, w = (self.dim,1)

        #loop over the batches to tokenize the inputs
        for i in range(batch_size):
            #Tokenize words using default fasttext tokenizer, which creates tokens 
            # by dividing splitting at word separating chars
            tokens.append(fasttext.tokenize(sentences[i]))

        #Create a matrix with batch_size batches, num token channels and 100x1 matrices to store the 100dim embeddings
        in_vector = np.zeros((batch_size, max_num_words, h, w))


        #cycle over the tokens and get their vectors, reshape them to 100x1 and store in the corresponding 
        #channel in the return variable
        
        #cycle over the entire batch
        for j in range(len(tokens)):

            #counter for tokens
            i = 0 
            
            #cycle over tokens
            for token in tokens[j]:
                
                #get the embedding for the single token
                vector = torch.tensor(self.ft[token].astype(np.double))
                
                #reshape it to desired dims
                vector = vector.reshape(h,w)
                
                #Store it in the input vectors matrix
                in_vector[j][i] = vector
                
                #increment the position of the word index within the given sentence
                #if it goes over the max word size, cut
                i=i+1
                if(i == max_num_words):
                    break

        #create a tensor object to return
        in_vector = torch.tensor(in_vector)

        return in_vector

    
    #Given a list of Turkish sentences, returns a list of True/False for their negativity
    #sentences: a list of sentence strings
    def is_negative(self, sentences):
        
        #get word vectors from the fasttext embeddings as torch.tensor
        sentence_vectors = self.input_vectors(sentences)
        
        #get the predictions from the model for the sentences
        outputs = self.model(sentence_vectors.float()).detach()
        
        #reshape the output from (n,1) to (n)
        outputs = outputs.reshape(outputs.shape[0])
        
        #get the output values as confidence scores
        confidences = outputs.clone().detach()
        
        #turn the tensor objects into numpy arrays
        confidences = confidences.numpy()
        outputs = outputs.numpy()
        
        #if the prediction is that the sentence is positive, then fix the confidence as 1-c
        for i in range(len(confidences)):
            if confidences[i] < 0.5:
                confidences[i] = 1 - confidences[i] 
        
        #turn the outputs into True/False values
        outputs = outputs > 0.5
        
        #return lists of outputs and confidences
        return outputs, confidences
    
    
    #reads the data in from a .txt file
    #Data Format:   0, words
    def data_from_txt(self, file_path):
        
        file = open(file_path, 'r')

        data_x = []
        data_y = []
        
        #n is the number of data points
        n = 0
        newline = '\n'
                   
        #Loop through the lines, and split at the comma for the inputs x and labels y
        for line in file:
            
            #store the x and y values in temp list
            temp = line.split(",")
            
            #strip the input string
            temp[1] = temp[1].strip()
            
            #if file ends with an empty line
            if temp[0] == '':
                break
            
            #append the Y and X values to the data lists
            data_y.append(float(temp[0]))
            
            if temp[1].endswith(newline):
                temp[1].replace(newline, '')
                   
            data_x.append(temp[1])

            n = n + 1

        file.close()

        return (data_x, data_y, n)
    
    
    #counts the number of parameters in th model for verbose output
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    
    #prints the model layers for verbose output
    def print_model(self):
        print(self.model)
        print("Total Parameters: ", count_parameters(self.model))
        return
    
    
    #Makes a torch.TensorDataset instance wtih X and Y
    def make_dataset(self, x, Y, n, verbose):
        
        #calls the input vector methods to get the word vectors for the string x
        X = self.input_vectors(x)
        
        #Casts the list Y as a tensor and makes it 2 dimensional
        Y = torch.tensor(Y)
        Y = Y.reshape(n, 1)
        
        if verbose:
            print(Y.shape)
            print(X.shape)
        
        dataset = TensorDataset(X, Y)

        return dataset
    
    
    #Trains the model with given train_data with batch_size for epoch with learning rate lr
    #and validates with given test_data at each step for book keeping
    def train(self, train_data, test_data, batch_size, epoch, lr=0.001, verbose=False):
        
        #Manual setting of the seed
        rand_seed = 40
        torch.manual_seed(rand_seed)
        np.random.seed(rand_seed)
        random.seed(rand_seed)

        #We use an Adam optimiser
        self.optimizer = Adam(self.model.parameters(), lr)
        
        #read X, Y and n from train and test data
        X, Y, n = train_data
        X_test, Y_test, n_test = test_data

        #Make TensorDatasets for train and test data
        data_train = self.make_dataset(X, Y, n, verbose)
        data_test = self.make_dataset(X_test, Y_test, n_test, verbose)
        
        #Create DataLOaders for both datasets
        train_loader = DataLoader(data_train, batch_size, shuffle = True)
        test_loader = DataLoader(data_test, batch_size=n_test, shuffle=False)

        #cast the model to a float model
        self.model = self.model.float()
        
        if verbose:
            print_model()
        
        #initialise lists for keep track of the values at each epoch for logging
        train_losses = []
        train_weights = []
        test_accs = []
        test_losses = []

        #train for epoch
        for e in range(epoch):

            #train for one epoch and test the values of the epoch
            train_loss, weight = self.train_epoch(train_loader)
            test_acc, test_loss = self.test_epoch(test_loader)
                
            #log the losses and accuracy ratings
            train_losses.append(train_loss)
            test_accs.append(test_acc)
            test_losses.append(test_loss)
            train_weights.append(weight)
            
            
            print("Epoch: ", e+1, "\tTrain Loss: %.3f" %train_loss.item(), 
                  "\tTest Loss: %.3f" %test_loss, "\tTest Acc: %.2f" %test_acc)

        #Plot the logs when the training is done
        plt.plot(train_losses, label="Training Losses")
        plt.plot(test_losses, label="Test Losses")
        plt.legend()
        plt.show() 
        plt.plot(test_accs, label="Test Accuracy")
        plt.legend()
        plt.show()
        
        #Put the model in evaluation mode
        self.model.eval()
        
        return

    
    #Does one step/epoch of the training of the model
    def train_epoch(self, train_loader):
        
        #set the model to training mode
        self.model.train()
    
        #training loss of the alst batch of his epoch
        last_loss = 0

        for i, data in enumerate(train_loader, 0):    

            inputs, labels = data

            self.optimizer.zero_grad()
                        
            X_out = self.model(inputs.float())

            loss_tr = self.loss(X_out, labels)

            loss_tr.backward()

            self.optimizer.step()

            last_loss += loss_tr

        s = torch.sum(self.model.out_layers[7].weight.data)

        return last_loss, s
    
    #For one epoch of training, calcualte the performance of that epoch's weights on the test data
    def test_epoch(self, test_loader):
        
        self.model.eval()
    
        correct = 0
        total = 0
        test_loss = 0

        with torch.no_grad():
            for data in test_loader:

                sents, labels = data

                outputs = self.model(sents.float())

                test_loss = self.loss(outputs, labels)

                outputs = outputs>=0.5

                total += labels.size(0)
                correct += (outputs==labels).sum().item()

        accuracy = 100*correct/total

        return accuracy, test_loss
    
    #given a test set of x and y values, returns predictions from the model, 
    #whether the prediction is correct and confidence output
    def test_acc(self, test_x, test_y, n):
    
        inputs = self.input_vectors(test_x)
        outputs = self.model(inputs.float())
        labels = torch.tensor(test_y)
        
        correct = 0
        total = n
        
        outputs = outputs.reshape(outputs.shape[0])
        
        predicts = outputs>=0.5
        
        correct = (predicts == labels)
        
        num_correct = correct.sum()
        
        accuracy = 100*num_correct/total

        print("Accuracy on the validation set of ", total, " items is: ", accuracy.item())

        return correct, predicts, outputs
    
    
    #A test method that prints out Type I and Type II error percentages of the model on a given dataset
    def test_confusion_matrix(self, path_to_file):
        
        x, y, n = self.data_from_txt(path_to_file)
        x_pos = [x[i] for i in range(len(y)) if y[i]==0]
        y_pos = [y[i] for i in range(len(y)) if y[i]==0]
        n_pos = len(x_pos)
        x_neg = [x[i] for i in range(len(y)) if y[i]==1]
        y_neg = [y[i] for i in range(len(y)) if y[i]==1]
        n_neg = len(x_neg)
    
        #correct, predictions, outputs
        cr_pos, _, _ = self.test_acc(x_pos, y_pos, n_pos)
        cr_neg, _, _ = self.test_acc(x_neg, y_neg, n_neg)
    
        labels = torch.tensor(y)
        
        #calculate positive identification accuracy
        corr_pos = sum(cr_pos)
        pos_acc = corr_pos/n_pos*100
        
        #calculate negative identification accuracy
        corr_neg = sum(cr_neg)
        neg_acc = corr_neg/n_neg*100
        
        print("Positive Accuracy: ", pos_acc.item(), "\tCorrectly Identified ", corr_pos.item(), " out of %d positives\n" %n_pos.item(),
              "Negative Accuracy: ", neg_acc.item(), "\tCorrectly Identified ", corr_neg.item(), " out of %d negatives" %n_neg.item())
    
    #A test method to identify the incorrectly classified test cases and print them
    def test_validate(self, path_to_file):
        
        x, y, n = self.data_from_txt(path_to_file)
        correct, preds, outs = self.test_acc(x, y, n)

        labels = torch.tensor(y)
        false_ind = [i for i in range(len(x)) if correct[i]==False]
        false_sents = [x[i] for i in range(len(x)) if correct[i]==False]
        false_preds = [int(preds[i]) for i in range(len(x)) if correct[i]==False]
        labs = [int(y[i]) for i in range(len(x)) if correct[i]==False]
        f_outs = [float(outs[i]) for i in range(len(x)) if correct[i]==False]
        falses = [false_ind, false_sents, labs, false_preds, f_outs]
        falses = list(map(list, zip(*falses)))
        
        for i in range(len(false_ind)):
            print("Ind: ", falses[i][0], "\tSent: ", falses[i][1], "Label: ", 
                  falses[i][2], "Model Pred: %d " %falses[i][3], "%0.2f" %falses[i][4])
        print(false_sents)
        
        return
    
    #A test method to check the distribution and the size of the input data
    def input_stats(self, path_to_file):
        
        _, y, n = self.data_from_txt(path_to_file)
        
        num_negs = np.sum(y)
        
        percent_neg = num_negs/n
        
        return percent_neg, n
    
    
    #A test method to measure the response time of the model to a given input
    def response_time(self, path_to_file):
                
        x, y, n = self.data_from_txt(path_to_file)

        start_time = time.time_ns()

        inputs = self.input_vectors(x)
        outputs = self.model(inputs.float())
        labels = torch.tensor(y)
        
        end_time = time.time_ns()
        
        print("Processed ", n, "items in ", (end_time-start_time)/1000000, " ms")