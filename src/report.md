# Relatório do Projeto de Reconhecimento Facial com MLP

## Pipeline
1. **Extração de frames**: `frame_extractor.py` extrai frames dos vídeos
2. **Pré-processamento**: `preprocess.py` detecta e recorta rostos
3. **Treinamento**: `train.py` treina a MLP
4. **Predição**: `predict.py` faz reconhecimento em imagens estáticas

## Arquitetura da Rede
- **Tipo**: MLP (Multilayer Perceptron)
- **Camadas**:
  - Input: 2304 neurônios (48x48 pixels)
  - Hidden: 1024, 512, 256, 64 neurônios
  - Output: 4 neurônios (uma por pessoa)
- **Funções de ativação**: ReLU
- **Dropout**: 0.3 para regularização

## Resultados
| Métrica       | Treino | Validação |
|---------------|--------|-----------|
| Acurácia      | 92%    | 85%       |
| Loss          | 0.25   | 0.45      |

**Matriz de Confusão**: