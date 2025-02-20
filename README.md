# NLP Sequence Classification on PLOD-CW Dataset

## Project Overview
This project is an individual coursework submission for the **Natural Language Processing (NLP)** module at the University of Surrey. The primary objective is **sequence classification and labeling** for abbreviation detection in scientific texts using the **PLOD-CW dataset**.

The project explores different **data preprocessing techniques**, NLP models, **hyperparameter tuning**, and **performance evaluations** to determine the most effective approach for abbreviation detection.

---

## Features
- **Sequence Classification and Labeling**: Identifies abbreviations in scientific texts.
- **Bi-LSTM and CRF-LSTM Models**: Implements and compares deep learning models for entity recognition.
- **BERT Tokenization vs. Traditional Tokenization**: Evaluates the impact of different tokenization methods.
- **Hyperparameter Tuning**: Uses Grid Search and Random Search for optimization.
- **Performance Evaluation**: Analyzes results using F1-score, precision, recall, and confusion matrices.

---

## Dataset: PLOD-CW
The **PLOD Dataset** is designed for abbreviation detection within scientific texts, particularly **acronyms (AC) and their long forms (LF)**. The dataset is structured as follows:
- **Tokens**: Words in the text.
- **POS Tags**: Part-of-Speech tags obtained from SpaCy.
- **NER Tags**: Labels for abbreviations and long forms.

**Data Splits**:
- **Training Set**: 79.3%
- **Validation Set**: 9.4%
- **Test Set**: 11.3%

---

## Technologies Used
- **Programming Language**: Python
- **Libraries & Frameworks**:
  - TensorFlow, PyTorch
  - Transformers (Hugging Face BERT)
  - FastAPI (for potential deployment)
  - Scikit-learn (evaluation metrics)
  - Keras-Tuner (hyperparameter tuning)

---

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed.

### Steps
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script for training the model:
   ```bash
   python train.py
   ```
5. Evaluate the model:
   ```bash
   python evaluate.py
   ```

---

## Model Experiments & Results

### 1. Tokenization Methods
- **Traditional Tokenization (NLTK-based)**
- **BERT Tokenization (Subword-based)**

**Findings**:
- **BERT tokenization improved performance**, handling scientific abbreviations more effectively.
- **F1-score improved from 0.23 (Traditional) to 0.79 (BERT).**

### 2. Model Comparison
| Model       | Accuracy | F1-score |
|------------|----------|----------|
| Bi-LSTM    | 86.2%    | 0.34     |
| CRF-LSTM   | 87.5%    | 0.41     |

- **CRF-LSTM outperformed Bi-LSTM**, as CRF better captured label dependencies.
- **Hyperparameter tuning further improved CRF-LSTM performance.**

### 3. Optimizers & Loss Functions
| Loss Function | Optimizer | F1-score |
|--------------|----------|----------|
| Categorical Crossentropy | Adam | **0.84** |
| Mean Squared Error | Adam | 0.50 |
| Categorical Crossentropy | SGD | 0.70 |

- **Adam optimizer with categorical crossentropy performed best.**

### 4. Hyperparameter Optimization
- **Grid Search**:
  - Best setup: **1 Bi-LSTM layer, 32 units, learning rate 0.01**
  - **Test Accuracy: 86.38%, F1-score: 0.8356**

- **Random Search**:
  - Best setup: **3 LSTM layers, mixed unit sizes**
  - **Test Accuracy: 86.52%, F1-score: 0.8283**

- **Conclusion**: Grid Search was more efficient, while Random Search explored deeper architectures.

---

## Best Model for Deployment
**Final Selected Model**: **Bi-LSTM with Grid Search Optimized Hyperparameters**

- **1 LSTM layer, 32 units**
- **Learning rate: 0.01**
- **Best Test Accuracy: 86.38%**
- **F1-score: 0.84**
- **Efficient and balanced performance**

---

## Future Work & Improvements
- **Improve class balance**: Implement **SMOTE or weighted loss**.
- **Enhance token representation**: Use **pre-trained embeddings (Word2Vec, Glove, BERT fine-tuning).**
- **More advanced models**: Explore **Transformer-based architectures** (e.g., **BERT, RoBERTa**).
- **Better hyperparameter tuning**: **Bayesian Optimization** over Grid/Random Search.

---

## Author
- **Saksham Ashwini Rai**  
  - MSc Artificial Intelligence, University of Surrey  

---

## License
This project is for educational purposes and is licensed under **MIT License**.

---

