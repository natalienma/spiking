# Same network/training loop as boxmodel_MNIST.py, but for the SHD (Spiking Heidelberg
# Digits) dataset -- spoken digit audio converted to spike events, instead of static images.

import snntorch as snn

import tonic
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

# dataloader arguments
batch_size = 128
data_path = './data'
TIME_WINDOW_US = 5000  # 5ms bins -- SHD event timestamps are in microseconds

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# SHD gives a variable-length list of (channel, timestamp) events per sample, not a fixed
# tensor. ToFrame bins events into time_window-sized windows -> (timesteps, channels) counts;
# we binarize since we only care whether a channel fired at all in that bin.
sensor_size = tonic.datasets.SHD.sensor_size
to_frame = tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=TIME_WINDOW_US)


def transform(events):
    frames = torch.from_numpy(to_frame(events)).squeeze(1).float()
    return (frames > 0).float()


shd_train = tonic.datasets.SHD(save_to=data_path, train=True, transform=transform)
shd_test = tonic.datasets.SHD(save_to=data_path, train=False, transform=transform)

# samples have different numbers of timesteps (different word durations) -- pad the shorter
# ones in each batch out to the longest, instead of MNIST's fixed-size images.
pad_collate = tonic.collation.PadTensors(batch_first=True)
train_loader = DataLoader(shd_train, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=pad_collate)
test_loader = DataLoader(shd_test, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=pad_collate)

# Network Architecture
num_inputs = sensor_size[0]  # 700 input channels
num_hidden = 1000
num_outputs = 20  # 20 spoken digit classes (0-9 in English and German)

# Temporal Dynamics
beta = 0.95

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        # x: (batch, timesteps, num_inputs) -- timesteps varies per batch (padded to the
        # longest sample in that batch), unlike MNIST's fixed num_steps.
        num_steps = x.shape[1]

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

# pass data into the network, sum the spikes over time
# and compare the neuron with the highest number of spikes
# with the target
def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(batch_size, data.shape[1], -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")



def train_printer():
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")

net = Net().to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

num_epochs = 1
loss_hist = []
test_loss_hist = []
counter = 0

# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    # Minibatch training loop
    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)
        num_steps = data.shape[1]

        # forward pass
        net.train()
        spk_rec, mem_rec = net(data.view(batch_size, num_steps, -1))

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)
            test_num_steps = test_data.shape[1]

            # Test set forward pass
            test_spk, test_mem = net(test_data.view(batch_size, test_num_steps, -1))

            # Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(test_num_steps):
                test_loss += loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy
            if counter % 50 == 0:
                train_printer()
            counter += 1
            iter_counter += 1
