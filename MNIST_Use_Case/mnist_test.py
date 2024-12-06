from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
#from torchvision.models.feature_extraction import create_feature_extractor
import os
import pickle
import logging

DIRECTORY = 'verrou'

#logger = logging.getLogger(__name__)
#logging.basicConfig(filename=f"/mnist/fuzzy_mnist_test2_{os.environ['TASK_ID']}.log", level=logging.INFO)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class Net_Embed(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        pickle.dump(x, open(f"/mnist/embeddings/{DIRECTORY}/conv1_{os.environ['TASK_ID']}.pkl", 'wb'))
        x = F.relu(x)
        x = self.conv2(x)
        pickle.dump(x, open(f"/mnist/embeddings/{DIRECTORY}/conv2_{os.environ['TASK_ID']}.pkl", 'wb'))
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        pickle.dump(x, open(f"/mnist/embeddings/{DIRECTORY}/pool_{os.environ['TASK_ID']}.pkl", 'wb'))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        pickle.dump(x, open(f"/mnist/embeddings/{DIRECTORY}/fc1_{os.environ['TASK_ID']}.pkl", 'wb'))
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        pickle.dump(x, open(f"/mnist/embeddings/{DIRECTORY}/fc2_{os.environ['TASK_ID']}.pkl", 'wb'))
        output = F.log_softmax(x, dim=1)
        return output

def get_test_embeddings(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    metrics = {'output': [], 'pred':[], 'target':[], 'loss': []}
    #metrics = {'pred':[], 'target':[] }
    with torch.no_grad():
        for data, target in test_loader:
            #logger.info(f'Before {data.shape}')
            data, target = data.to(device), target.to(device)
            output = model(data)
            batch_loss = F.nll_loss(output, target, reduction='sum').item() 
            test_loss += batch_loss  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            break


    test_loss /= len(test_loader.dataset)

    # print(f'\nTest set: Average loss: {test_loss}')
    print('\nTest set: Average loss: {:}, Accuracy: {}/{} ({:}%)\n'.format(
       test_loss, correct, len(test_loader.dataset),
       100. * correct / len(test_loader.dataset)))

    return metrics


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    metrics = {'output': [], 'pred':[], 'target':[], 'loss': []}
    #metrics = {'pred':[], 'target':[] }
    with torch.no_grad():
        for data, target in test_loader:
            #logger.info(f'Before {data.shape}')
            data, target = data.to(device), target.to(device)
            output = model(data)
            batch_loss = F.nll_loss(output, target, reduction='sum').item() 
            test_loss += batch_loss  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            metrics['loss'].append( batch_loss/data.size(0) )
            metrics['output'].append(output)
            metrics['pred'].append(pred)
            metrics['target'].append(target)
            #logger.info(f'After {data.shape}')

    test_loss /= len(test_loader.dataset)

    # print(f'\nTest set: Average loss: {test_loss}')
    print('\nTest set: Average loss: {:}, Accuracy: {}/{} ({:}%)\n'.format(
       test_loss, correct, len(test_loader.dataset),
       100. * correct / len(test_loader.dataset)))

    return metrics


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='For loading the model')
    parser.add_argument('--get-embeddings', action='store_true', default=False,
                        help='For loading the model')
    parser.add_argument('--model-path', type=str)

    #logger = logging.getLogger(__name__)
    #logging.basicConfig(filename='test.log', encoding='utf-8', level=logging.INFO)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
   
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    print(len(dataset2))
    if args.get_embeddings:
        model = Net_Embed().to(device)
    
    else:
        model = Net().to(device)
    
    if args.load_model:
        model.load_state_dict(torch.load(args.model_path))
    
    if args.get_embeddings:
        _ = get_test_embeddings(model, device, test_loader)

    else:
        metrics = test(model, device, test_loader)
        
        pickle.dump(metrics, open(f"/mnist/{DIRECTORY}/test_metrics_{os.environ['TASK_ID']}.pkl", 'wb'))

    #GET MODEL EMBEDDINGS
    # model2 = create_feature_extractor(model, return_nodes={'conv1':'conv1'}).to(device)
    # model3 = create_feature_extractor(model, return_nodes={'fc2':'fc2'}).to(device)

    # for data, target in test_loader:
    #     intermediate_outputs1 = model2(data)
    #     intermediate_outputs2 = model3(data)
    #     break

    # pickle.dump(intermediate_outputs1, open(f'conv_embed_{os.environ["TASK_ID"]}', 'wb'))
    # pickle.dump(intermediate_outputs2, open(f'fc_embed_{os.environ["TASK_ID"]}', 'wb'))

    # metrics = test(model, device, test_loader)
    # pickle.dump(metrics, open(f'mnist_results/rr/mnist_results_{os.environ["TASK_ID"]}.pkl', 'wb'))



if __name__ == '__main__':
    main()
