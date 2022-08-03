import mygrad as mg
# import torch
from torch import tensor
import torch.nn as nn
relu = nn.functional.relu
import torch.nn as nn
import torch.nn.functional as F
import torch
# import torchvision
import torchvision.transforms as transforms
import numpy as np
class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.pool = nn.MaxPool2d(2,2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 16, 5)
        self.conv4 = nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(11664, 1166)
        self.fc2 = nn.Linear(1166,116)
        self.fc3 = nn.Linear(116, 12)
        self.fc4 = nn.Linear(12, 2)
        

        
        for layer in (self.conv1, self.conv2, self.conv3, self.conv4, self.fc1, self.fc2, self.fc3, self.fc4):
            nn.init.xavier_normal_(layer.weight, np.sqrt(2)).type(torch.cuda.FloatTensor)
            nn.init.constant_(layer.bias, 0).type(torch.cuda.FloatTensor)
    def forward(self, x):
        # print("forward")
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
def accuracy(predictions, truth):
    """
    Returns the mean classification accuracy for a batch of predictions.
    
    Parameters
    ----------
    predictions : Union[numpy.ndarray, mg.Tensor], shape=(M, D)
        The scores for D classes, for a batch of M data points
    truth : numpy.ndarray, shape=(M,)
        The true labels for each datum in the batch: each label is an
        integer in [0, D)
    
    Returns
    -------
    float
    """
    # print(predictions.size())
    # print(truth.size())
    if isinstance(predictions, mg.Tensor):
        predictions = predictions.data
    return np.mean((torch.argmax(predictions, dim=1) == truth).cpu().numpy())

def train_model(model: Model, train_data, test_data, X, Y, optim, epochs=100, batch_size=50, plotter=None):
    for epoch_cnt in range(epochs):
        idxs = np.arange(train_data.size()[0])  # -> array([0, 1, ..., 9999])
        np.random.shuffle(idxs)  
        
        for batch_cnt in range(0, train_data.size()[0]//batch_size):
            batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
            batch = train_data[batch_indices]  # random batch of our training data
            
            prediction = model(batch)
        
            truth = test_data[batch_indices]
            # print(prediction, truth)
            # Although its name does not indicate this, the `cross_entropy` loss
            # here also *includes a softmax* before computing the actual cross-entropy.
            loss = nn.functional.cross_entropy(prediction, truth)
            loss.backward()
            
            optim.step()
            
            # Unlike in MyGrad, after you perform a gradient-based step with your optimizer, you
            # must explicitly delete/zero-out the gradients of your model's parameters
            # once you are done with them. MyGrad handles this for us, but PyTorch does not.
            optim.zero_grad()  
            if plotter is not None:
                acc = accuracy(prediction, truth)
                plotter.set_train_batch({"loss" : loss.item(),
                                        "accuracy" : acc},
                                        batch_size=batch_size, plot=True)
        
        # This context manager simply signals to pytorch that we will not be 
        # computing any gradients (since we are only evaluating our model on
        # test data, not training on it). This will allow PyTorch to optimize
        # its computation
        if plotter is not None:
            with torch.autograd.no_grad():
                for batch_cnt in range(0, X.size()[0]//batch_size):
                    idxs = np.arange(X.size()[0])
                    batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
                    batch = X[batch_indices] 
                    prediction = model(batch)
                    truth = Y[batch_indices]
                    acc = accuracy(prediction, truth)
                    plotter.set_test_batch({ "accuracy" : acc},
                                            batch_size=batch_size)
            plotter.set_train_epoch()
            plotter.set_test_epoch()  