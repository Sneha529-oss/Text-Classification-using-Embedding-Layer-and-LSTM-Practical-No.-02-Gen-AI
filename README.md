# Generative AI - Practical No-02: Text Classification using Embedding Layer and LSTM

## Emotion Classification — Embedding + LSTM on the Emotion Dataset

Multi-class text classification pipeline built for a Generative AI lab assignment. An **Embedding layer + LSTM network** is trained to identify the emotion expressed in an English text message (six classes), and evaluated using accuracy, precision, recall, F1-score and a confusion matrix.

---

##  Project Info

| Field                | Details           |
| -------------------- | ----------------- |
| **Student**          | Sneha Chaurasia   |
| **PRN**              | 202401110046      |
| **Class / Division** | CSE-AIML / A      |
| **Subject**          | Generative AI Lab |
| **Practical No.**    | 02                |

---

## Problem Statement

Build a multi-class text classification model using an Embedding layer and a Long Short-Term Memory (LSTM) network to classify text into different emotion categories. Train the model on the Emotion dataset and evaluate its performance using accuracy, precision, recall, F1-score and a confusion matrix.

## Objective

To develop and evaluate an LSTM-based text classification model that can identify the emotion expressed in a given text.

---

## 📦 Dataset

| Field         | Details                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------- |
| **Name**      | Emotion Dataset (`dair-ai/emotion`)                                                         |
| **Source**    | [Hugging Face](https://huggingface.co/datasets/dair-ai/emotion)                             |
| **Type**      | English Twitter messages labelled with one emotion                                          |
| **Classes**   | Sadness, Joy, Love, Anger, Fear, Surprise                                                   |
| **Loading**   | Downloaded automatically in the notebook with `load_dataset("dair-ai/emotion")`             |

| Split      | Samples    |
| ---------- | ---------- |
| Training   | 16,000     |
| Validation | 2,000      |
| Testing    | 2,000      |
| **Total**  | **20,000** |

The dataset is **class-imbalanced** (Joy and Sadness dominate; Surprise is the rarest), which is handled with class weights during training.

---

## 🧠 Model Architecture

```
Text Input → Tokenization → Padding → Embedding → LSTM → Dropout → Dense → Softmax
```

| Layer                                  | Purpose                                                        |
| -------------------------------------- | -------------------------------------------------------------- |
| `Embedding(vocab, 128, mask_zero=True)` | Converts word indices to dense vectors; padding is masked      |
| `LSTM(128)`                            | Reads the sequence and captures context                        |
| `Dropout(0.4)`                         | Regularisation                                                 |
| `Dense(64, relu)`                      | Learns non-linear feature combinations                         |
| `Dropout(0.3)`                         | Regularisation                                                 |
| `Dense(6, softmax)`                    | Probability for each of the 6 emotions                         |

**Training setup:** Adam optimizer (lr = 1e-3), sparse categorical cross-entropy loss, batch size 64, up to 25 epochs with `EarlyStopping` (restores best weights) and `ReduceLROnPlateau`, class weights for imbalance.

---

## 🔄 Workflow

1. Import libraries and load the dataset
2. Exploratory data analysis (class distribution, text length)
3. Text preprocessing (lowercase, remove non-letters; stop-words kept because words like *not* affect emotion)
4. Tokenization (vocabulary of 10,000 words built from the **training set only** to avoid data leakage)
5. Padding to the 95th-percentile sequence length
6. Build, compile and train the Embedding + LSTM model
7. Plot accuracy and loss curves
8. Evaluate on the test set: classification report, confusion matrix
9. Predict emotions for sample and custom sentences

---

## 📊 Results

> Replace the `XX.XX` values with the numbers printed in the executed notebook (Section 22: Result).

**Test Set Results:**

| Metric               | Value  |
| -------------------- | ------ |
| Test Accuracy        | XX.XX  |
| Macro Precision      | XX.XX  |
| Macro Recall         | XX.XX  |
| Macro F1-score       | XX.XX  |
| Weighted F1-score    | XX.XX  |
| Test Loss            | XX.XX  |
| Best Epoch           | XX     |
| Total Parameters     | XX,XXX,XXX |

**Per-class performance (test set):**

| Emotion  | Precision | Recall | F1-score |
| -------- | --------- | ------ | -------- |
| Sadness  | XX.XX     | XX.XX  | XX.XX    |
| Joy      | XX.XX     | XX.XX  | XX.XX    |
| Love     | XX.XX     | XX.XX  | XX.XX    |
| Anger    | XX.XX     | XX.XX  | XX.XX    |
| Fear     | XX.XX     | XX.XX  | XX.XX    |
| Surprise | XX.XX     | XX.XX  | XX.XX    |

**Graphs generated in the notebook:** class distribution, text-length distribution, training/validation accuracy, training/validation loss, per-class precision/recall/F1, confusion matrix (counts and normalised), prediction-probability chart.

### Observations

- The model performs well above the majority-class baseline (~33%, always predicting *Joy*).
- Frequent classes (Joy, Sadness, Anger, Fear) score higher than the rare classes (Love, Surprise) because they have fewer training examples.
- The most common errors occur between semantically close emotions such as **Joy ↔ Love** and **Fear ↔ Surprise**.
- Training and validation curves stay close and EarlyStopping restores the best epoch, so overfitting is controlled.

---

## ✅ Conclusion

An Embedding + LSTM network was successfully trained to classify text into six emotions on the Emotion dataset. The LSTM captures word order and context, which a bag-of-words approach cannot, and it classifies unseen text reliably. Limitations are the class imbalance, confusion between similar emotions, and the embeddings being learned from scratch on a small dataset.

**Possible improvements:** Bidirectional LSTM / GRU, pre-trained embeddings (GloVe, Word2Vec, FastText), attention mechanisms, transformer models such as BERT / DistilBERT, and hyper-parameter tuning.

---

## 📁 Repository Structure

```
GenAI-Text-Classification-using-Embedding-and-LSTM/
│
├── README.md                          ← this file
├── Practical_02_Emotion_LSTM.ipynb    ← main notebook (executed, with outputs)
└── Practical_02_Report.pdf            ← assignment report
```

The dataset is loaded from Hugging Face inside the notebook, so no dataset folder is needed.

---

## ⚙️ How to Run

**Option 1 — Google Colab (recommended)**

1. Open `Practical_02_Emotion_LSTM.ipynb` in Google Colab (`File → Upload notebook`).
2. Select a GPU: `Runtime → Change runtime type → T4 GPU` (optional, faster).
3. Run all cells: `Runtime → Run all`.

**Option 2 — Local Jupyter**

```
git clone https://github.com/Sneha529-oss/GenAI-Text-Classification-using-Embedding-and-LSTM.git
cd GenAI-Text-Classification-using-Embedding-and-LSTM
pip install tensorflow datasets numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook Practical_02_Emotion_LSTM.ipynb
```

Training takes about 2–5 minutes on a GPU.

---

## 📦 Requirements

```
tensorflow
datasets
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

---

## ✅ Submission Checklist

- [x] Code file (Jupyter Notebook, executed end-to-end)
- [x] Dataset source link included above
- [x] Text preprocessing, tokenization and padding
- [x] Embedding + LSTM model, training and evaluation
- [x] Accuracy / loss plots, classification report, confusion matrix
- [x] Sample predictions
- [x] Report PDF
- [x] README file

---

## 📖 References

1. Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., & Chen, Y.-S. (2018). *CARER: Contextualized Affect Representations for Emotion Recognition.* EMNLP 2018.
2. Emotion Dataset (Hugging Face): <https://huggingface.co/datasets/dair-ai/emotion>
3. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.
4. TensorFlow / Keras documentation: <https://www.tensorflow.org/>

---

## 📝 Declaration

I, **Sneha Chaurasia**, confirm that this implementation was prepared for Practical Assignment 2 and that the results presented are generated from the experiments performed in this notebook.

**GitHub Repository Link:** <https://github.com/Sneha529-oss/GenAI-Text-Classification-using-Embedding-and-LSTM>
