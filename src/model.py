import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[1024, 512, 256, 64], num_classes=4, dropout_rate=0.3):
        super(MLP, self).__init__()
        
        # Validação de parâmetros
        if not isinstance(hidden_sizes, list) or len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes deve ser uma lista não vazia")
            
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError("dropout_rate deve estar entre 0 e 1")
            
        # Camadas dinâmicas baseadas em hidden_sizes
        layers = []
        prev_size = input_size
        
        for i, h_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            
            # Aplicar dropout apenas se não for a última camada
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout_rate))
                
            prev_size = h_size
            
        # Camada de saída
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.layers = nn.Sequential(*layers)
        
        # Inicialização de pesos
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        return self.layers(x)
        
    def get_summary(self):
        """Retorna um resumo da arquitetura"""
        summary = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                summary.append(f"Linear(in={layer.in_features}, out={layer.out_features})")
            elif isinstance(layer, nn.Dropout):
                summary.append(f"Dropout(p={layer.p})")
            elif isinstance(layer, nn.ReLU):
                summary.append("ReLU()")
        return "\n".join(summary)

if __name__ == "__main__":
    model = MLP(2304, [1024, 512, 256, 64], 4)
    print("Resumo do modelo:")
    print(model.get_summary())
    print(f"Total de parâmetros: {sum(p.numel() for p in model.parameters())}")