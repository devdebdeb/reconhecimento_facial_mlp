import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        super(MLP, self).__init__()

        layers = []
        in_features = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    input_size = 4096  # Exemplo
    hidden_sizes = [2048, 512, 128]
    num_classes = 5

    model = MLP(input_size, hidden_sizes, num_classes)
    print(model)

    x = torch.randn(32, input_size)
    out = model(x)
    print("Saída do modelo:", out.shape)
