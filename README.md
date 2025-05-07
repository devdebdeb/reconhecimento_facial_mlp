Reconhecimento Facial com MLP

Sistema de reconhecimento facial utilizando redes neurais densas (MLP), desenvolvido em PyTorch, com foco didático e estrutura modular.

📌 Objetivo

Reconhecer imagens faciais sem usar CNNs, embeddings pré-treinados ou transfer learning.

🧠 Arquitetura

Entrada: vetor de pixels 48x48

MLP com 3 camadas ocultas (512 → 256 → 128)

Saída para 4 classes

⚙️ Execução

### Extrair frames dos vídeos
python src/frame_extractor.py

### Preprocessamento dos rostos
python src/preprocess.py

### Treinar o modelo
python src/train.py

### Realizar predições em imagens novas
python src/predict.py

📁 Estrutura

├── src/

│   ├── dataset.py

│   ├── frame_extractor.py

│   ├── preprocess.py

│   ├── train.py

│   ├── predict.py

│   └── model.py

├── models/

├── dataset_raw/

├── processed_dataset/

├── minhas_fotos/

├── README.md

└── RELATORIO.md

👥 Integrantes

André Messina
César Sibila
Enzo Takida
Willian Dias

🔒 Licença

Projeto acadêmico, uso livre com créditos.
