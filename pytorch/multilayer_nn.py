import torch
import torch.nn.functional as F
import math

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            # ReLU introduces non-linearity by applying max(0, x)
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs)
        )
    
    def forward(self, x):
        logits = self.layers(x)
        return logits

if __name__ == "__main__":
    torch.manual_seed(123)
    model = NeuralNetwork(50, 3)
    print(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable model parameters: {num_params}")

    print(model.layers[0].weight.shape)
    print(model.layers[0].bias.shape)
    print(model.layers[0].weight)

    num_params_count = 0
    for i in range(len(model.layers)):
        if isinstance(model.layers[i], torch.nn.Linear):
            weights = math.prod(model.layers[i].weight.shape)
            bias = model.layers[i].bias.shape[0]
            print(f"Layer {i} has {weights} weights, {bias} and bias values.")
            num_params_count += (weights + bias)

    print(f"Total number of trainable model parameters: {num_params} and counted {num_params_count}")

    # forward pass
    X = torch.randn((1, 50))
    with torch.no_grad():
        out = model(X)
        out = torch.softmax(model(X), dim=1) 
    print(torch.sum(out))