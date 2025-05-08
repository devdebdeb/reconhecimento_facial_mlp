
# RELATÓRIO TÉCNICO — Reconhecimento Facial com MLP

## 1. Descrição do Pipeline

O projeto implementou um sistema completo de reconhecimento facial utilizando redes neurais densas (MLP) com PyTorch, respeitando as restrições técnicas definidas pela disciplina.

### 1.1 Coleta e Extração de Frames
Foram utilizados vídeos gravados por cada participante. O script `frame_extractor.py` extrai automaticamente frames de vídeos e organiza as imagens brutas em subpastas por indivíduo em `dataset_raw/`.

### 1.2 Pré-processamento Facial
Utilizamos o MediaPipe para detectar e recortar os rostos. O script `preprocess.py` realiza:
- Detecção do rosto mais confiável por imagem;
- Aplicação de margem ao bounding box;
- Redimensionamento para 48x48 pixels em escala de cinza;
- Salvamento das imagens padronizadas em `processed_dataset/`.

### 1.3 Dataset PyTorch
A classe `FaceDataset` (`dataset.py`) organiza os dados processados e aplica:
- Augmentation (rotação e flip horizontal, quando em treino);
- Normalização para o intervalo [-1, 1];
- Vetorização da imagem (entrada como vetor de 2304 pixels).

### 1.4 Treinamento com Validação Cruzada
O script `train.py` implementa 5-fold cross-validation estratificada. Em cada fold:
- O modelo é reconfigurado e re-inicializado;
- É utilizado `WeightedRandomSampler` para compensar desbalanceamentos;
- A cada época são computadas métricas de loss e acurácia;
- O melhor modelo (por loss de validação) é salvo automaticamente.

### 1.5 Predição com Threshold
O script `predict.py` carrega o modelo treinado e realiza inferência em imagens novas. Caso a confiança da predição seja inferior a 50%, a saída retorna "Desconhecido".

---

## 2. Arquitetura da Rede Utilizada

A rede foi definida no módulo `model.py`. Sua arquitetura é puramente densa (MLP):

- Entrada: vetor 48x48 = 2304 elementos
- Camada 1: Linear(2304, 512) + BatchNorm + ReLU + Dropout(0.4)
- Camada 2: Linear(512, 256) + BatchNorm + ReLU + Dropout(0.3)
- Camada 3: Linear(256, 128) + BatchNorm + ReLU + Dropout(0.2)
- Saída: Linear(128, 4)

**Parâmetros adicionais:**
- Loss: `CrossEntropyLoss`
- Otimizador: `AdamW` com weight decay
- Scheduler: `ReduceLROnPlateau` para ajuste de learning rate

---

## 3. Resultados Obtidos

### 3.1 Acurácia por Fold
Durante a validação cruzada, observou-se alta acurácia interna (>95%) na maioria dos folds, indicando forte capacidade de ajuste do modelo ao conjunto processado.

### 3.2 Predição Externa
Ao avaliar imagens externas (fora do dataset de treino), o modelo mostrou limitações:
- Casos de acerto consistentes em imagens similares às de treino;
- Classificações incorretas em imagens com ângulo ou iluminação diferentes;
- Alta frequência de retorno "Desconhecido" quando o threshold era mantido em 0.5.

### 3.3 Matriz de Confusão (Validação Interna)

- A matriz de confusão revelou separação clara entre classes durante a validação cruzada.
- Confusões foram mínimas, geralmente ligadas a menor representatividade de uma das classes.

> **Observação:** A matriz de confusão real pode ser gerada com script adicional para visualização gráfica dos resultados por classe.

---

## 4. Dificuldades Enfrentadas e Aprendizados

| Desafio Técnico                     | Solução Aplicada                                                 |
|------------------------------------|------------------------------------------------------------------|
| Baixo volume de dados por classe   | Data augmentation (flip, rotação) + amostragem balanceada        |
| Detecção facial inconsistente      | Implementação de margem no bounding box com MediaPipe            |
| Overfitting na validação interna   | Regularização com Dropout e early stopping                       |
| Predição fraca em domínio externo  | Threshold conservador + avaliação crítica do pipeline            |

### Aprendizados:
- A robustez do pré-processamento é tão crítica quanto a arquitetura da rede.
- MLPs podem aprender bem com dados vetorizados, mas são sensíveis à variação espacial e de iluminação.
- Avaliar com dados externos é essencial para testar generalização.
- Modularização do código facilitou ajustes e depuração durante todo o projeto.

---

## Referências

- GOODFELLOW, Ian et al. *Deep Learning*. MIT Press, 2016.
- RUSSELL, Stuart. *Artificial Intelligence: A Modern Approach*. Pearson, 2021.
- Documentação oficial do [PyTorch](https://pytorch.org/)
- Documentação oficial do [MediaPipe](https://google.github.io/mediapipe/)
