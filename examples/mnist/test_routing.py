import numpy as np
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.route import Route_data
from bindsnet.analysis.plotting import (
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.network import Network
from bindsnet.network.nodes import Input

# Build a simple two-layer, input-output network.
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import LIFNodes
from bindsnet.network.topology import Connection


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=500)
parser.add_argument("--n_epochs", type=int, default=4000)
parser.add_argument("--examples", type=int, default=219)
parser.add_argument("--examples_test", type=int, default=201)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=64)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--device_id", type=int, default=0)
parser.set_defaults(plot=False, gpu=True, train=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
examples = args.examples
examples_test = args.examples_test
n_workers = args.n_workers
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu
device_id = args.device_id

np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

# Sets up Gpu use
if gpu and torch.cuda.is_available():
    torch.cuda.set_device(device_id)
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.manual_seed(seed)
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        # h = int(input_size/2)
        self.linear_1 = nn.Linear(input_size, num_classes)
        # self.linear_1 = nn.Linear(input_size, h)
        # self.linear_2 = nn.Linear(h, num_classes)

    def forward(self, x):
        out = torch.sigmoid(self.linear_1(x.float().view(-1)))
        # out = torch.sigmoid(self.linear_2(out))
        return out
#  create network
network = Network(dt=dt)
inpt = Input(10, shape=(1, 10, 1))
network.add_layer(inpt, name="I")
output = LIFNodes(n_neurons, thresh=-52 + np.random.randn(n_neurons).astype(float))
network.add_layer(output, name="O")
C1 = Connection(source=inpt, target=output, w=0.5 * torch.randn(inpt.n, output.n))
C2 = Connection(source=output, target=output, w=0.5 * torch.randn(output.n, output.n))

network.add_connection(C1, source="I", target="O")
network.add_connection(C2, source="O", target="O")

spikes = {}
for l in network.layers:
    spikes[l] = Monitor(network.layers[l], ["s"], time=time)
    network.add_monitor(spikes[l], name="%s_spikes" % l)

voltages = {"O": Monitor(network.layers["O"], ["v"], time=time)}
network.add_monitor(voltages["O"], name="O_voltages")

# Directs network to GPU
if gpu:
    network.to("cuda")

#get  training_dataset& test_dataset

# dataset = Route_data(csv_file='./route/TTC_train.csv',
#                                     root_dir ='./route', time = time, dt = dt)
dataset = Route_data(csv_file='./route/TTC_train.csv',
                                    root_dir ='./route', time = time, dt = dt)
# dataset_t = Route_data(csv_file='./route/TTC_test2.csv',
#                                     root_dir='./route', time = time, dt = dt)
dataset_t = Route_data(csv_file='./route/TTC_test2.csv',
                                  root_dir='./route', time = time, dt = dt)
inpt_axes = None
inpt_ims = None
spike_axes = None
spike_ims = None
weights_im = None
weights_im2 = None
voltage_ims = None
voltage_axes = None

# Create a dataloader to iterate and batch data
# dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=gpu
# )
dataloader_t = torch.utils.data.DataLoader(
    dataset_t, batch_size=1, shuffle=True, num_workers=0, pin_memory=gpu
)

n_iters = examples_test

test_pairs = []
pbar = tqdm(enumerate(dataloader_t))
for (i, dataPoint) in pbar:
    # if i > n_iters:
    #     break
    datum = dataPoint["train_data"].view(time, 1, 1, 10, 1).to(device_id)
    label = dataPoint["label"]
    pbar.set_description_str("Testing progress: (%d / %d)" % (i+1, n_iters))

    network.run(inputs={"I": datum}, time=250, input_time_dim=1)
    test_pairs.append([spikes["O"].get("s").sum(0), label, dataPoint['each_index']])

    if plot:

        spike_ims, spike_axes = plot_spikes(
            {layer: spikes[layer].get("s").view(250,-1 ) for layer in spikes},
            axes=spike_axes,
            ims=spike_ims,
        )
        voltage_ims, voltage_axes = plot_voltages(
            {layer: voltages[layer].get("v").view(250, -1) for layer in voltages},
            ims=voltage_ims,
            axes=voltage_axes,
        )
        weights_im2 = plot_weights(C2.w, im=weights_im2, wmin=-2, wmax=2)

        plt.pause(1e-8)
    network.reset_state_variables()
model = NN(n_neurons, 4).to(device_id)
model.load_state_dict(torch.load("Liner.pth"))
# Test the Model
correct, total = 0, 0
for s, label, each_index in test_pairs:
    outputs = model(s)
    _, predicted = torch.max(outputs.data.unsqueeze(0), 1)
    total += 1
    citydict = {0 : "Houston", 1 : "Madrid", 2 : "Goldstone", 3 : "Canberra"}
    if  predicted == label.long().to(device_id):
        print("0:Houston, 1:Madrid, 2: Goldstone, 3:Canberra and the prediction is city: %s" % citydict[predicted.item()])
    else:
        print("number: %d is wrong, and predict %s as %s " % (each_index, citydict[label.long().to(device_id).item()], citydict[predicted.item()]))
    correct += int(predicted == label.long().to(device_id))

print(
    "\n Accuracy of the model on %d test  %.2f %%"
    % (total, 100 * correct / total)
)