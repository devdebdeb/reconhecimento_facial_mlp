# README — Reconhecimento Facial com MLP

Sistema de reconhecimento facial utilizando redes neurais densas (MLP), desenvolvido com PyTorch como parte do Projeto 1 da disciplina de Visão Computacional (PUC-SP).

---

## 📌 Objetivo

Construir um sistema de reconhecimento facial a partir de vídeos dos participantes, utilizando uma rede MLP treinada com imagens vetorizadas, sem uso de CNNs ou modelos pré-treinados.

---

## 🧠 Arquitetura da Rede

* Entrada: vetor de 48×48 pixels (2304 features)
* Camadas ocultas:

  * Linear(2304 → 512) + BatchNorm + ReLU + Dropout(0.4)
  * Linear(512 → 256) + BatchNorm + ReLU + Dropout(0.3)
  * Linear(256 → 128) + BatchNorm + ReLU + Dropout(0.2)
* Saída: Linear(128 → número de participantes)
* Função de perda: CrossEntropyLoss
* Otimizador: AdamW
* Scheduler: ReduceLROnPlateau

---

## 🧪 Pipeline de Execução

### 1. Extração de Frames dos Vídeos

```bash
python src/frame_extractor.py
```

* Entrada: pasta `videos/`
* Saída: imagens em `dataset_raw/`

### 2. Detecção e Recorte Facial com MediaPipe

```bash
python src/preprocess.py
```

* Entrada: `dataset_raw/`
* Saída: rostos em escala de cinza, salvos em `processed_dataset/`

### 3. Treinamento do Modelo MLP

```bash
python src/train.py
```

* Entrada: `processed_dataset/`
* Saída: pesos salvos na pasta `models/`

### 4. Predição em Imagens Novas

```bash
python src/predict.py
```

* Entrada: imagens em `minhas_fotos/`
* Saída: nome da pessoa ou “Desconhecido” com base no threshold de confiança

---

## 🗂 Estrutura de Diretórios

```
├── dataset_raw/             # Imagens extraídas dos vídeos (input bruto)

├── processed_dataset/       # Rostos detectados e normalizados

├── models/                  # Pesos dos modelos MLP treinados

├── minhas_fotos/            # Imagens de teste para predição

├── videos/                  # Vídeos de entrada

├── src/                     # Scripts-fonte

│   ├── dataset.py

│   ├── frame_extractor.py

│   ├── preprocess.py

│   ├── train.py

│   ├── predict.py

│   ├── model.py

│   └── evaluate.py (opcional)

├── requirements.txt

├── README.md

└── RELATORIO_MLP.md
```

---

## ⚠️ Requisitos e Execução

### Ambiente Virtual (recomendado)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

---

## 👤 Integrantes

* André Messina
* Cesar Sibila
* Enzo Takida 
* Willian Dias

---

## 📄 Licença

Projeto acadêmico — uso não comercial.
