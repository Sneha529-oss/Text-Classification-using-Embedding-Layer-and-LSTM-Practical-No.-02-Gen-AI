# Generative AI - Practical No-02: Text Classification using Embedding Layer and LSTM

## Emotion Classification — Embedding + LSTM on the Emotion Dataset

Multi-class text classification pipeline built for a Generative AI lab assignment. An **Embedding layer + LSTM network** is trained to identify the emotion expressed in an English text message (six classes), and evaluated using accuracy, precision, recall, F1-score and a confusion matrix.

---

## Project Info

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

## Dataset

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

## Model Architecture

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

**Training setup:** Adam optimizer (lr = 1e-3), sparse categorical cross-entropy loss, batch size 64, up to 25 epochs with `EarlyStopping` (restores best weights) and `ReduceLROnPlateau`, class weights for imbalance. In the final run training stopped at epoch 7 and the best weights (epoch 3) were restored.

---

## Workflow

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

## Results

**Training summary:** Training ran on a GPU and stopped early at epoch 7 (EarlyStopping, patience = 4). The weights from the **best epoch (epoch 3, lowest validation loss)** were restored and used for all evaluation below. The model has **1,420,230 trainable parameters (5.42 MB)**, with a vocabulary of 10,000 words and a padded sequence length of 41.

**Loss and accuracy:**

| Split      | Loss   | Accuracy |
| ---------- | ------ | -------- |
| Train      | 0.1412 | 0.9544   |
| Validation | 0.3287 | 0.9075   |
| Test       | 0.3564 | **0.8905** |

**Test Set Results:**

| Metric             | Value  |
| ------------------ | ------ |
| Test Accuracy      | 0.8905 |
| Macro Precision    | 0.8335 |
| Macro Recall       | 0.8867 |
| Macro F1-score     | 0.8535 |
| Weighted Precision | 0.9003 |
| Weighted Recall    | 0.8905 |
| Weighted F1-score  | 0.8930 |
| Test Loss          | 0.3564 |

**Classification Report (test set, 2,000 samples):**

| Emotion  | Precision | Recall | F1-score | Support |
| -------- | --------- | ------ | -------- | ------- |
| Sadness  | 0.9625    | 0.8847 | 0.9220   | 581     |
| Joy      | 0.9251    | 0.9065 | 0.9157   | 695     |
| Love     | 0.7241    | 0.9245 | 0.8122   | 159     |
| Anger    | 0.8750    | 0.8909 | 0.8829   | 275     |
| Fear     | 0.9034    | 0.8348 | 0.8677   | 224     |
| Surprise | 0.6105    | 0.8788 | 0.7205   | 66      |
| **Macro avg**    | 0.8335 | 0.8867 | 0.8535 | 2000 |
| **Weighted avg** | 0.9003 | 0.8905 | 0.8930 | 2000 |

**Top 5 misclassifications (actual → predicted):**

| Actual  | Predicted | Samples |
| ------- | --------- | ------- |
| Joy     | Love      | 50      |
| Sadness | Joy       | 26      |
| Sadness | Anger     | 23      |
| Fear    | Surprise  | 22      |
| Sadness | Fear      | 11      |

**Graphs generated in the notebook:** class distribution, text-length distribution, training/validation accuracy, training/validation loss, per-class precision/recall/F1, confusion matrix (counts and normalised), prediction-probability chart.

### Observations

- The model reaches **89.05% test accuracy**, far above the majority-class baseline of about 33% (always predicting *Joy*).
- **Sadness (F1 0.922) and Joy (F1 0.916)** are predicted best. **Surprise (F1 0.721)** and **Love (F1 0.812)** are hardest because they have the fewest training examples (3.6% and 8.2%).
- Surprise and Love have **high recall but lower precision** (0.88 vs 0.61 and 0.92 vs 0.72). This is the effect of class weights: minority classes are found more often, at the cost of some false positives. This is why macro recall (0.887) is higher than macro precision (0.834).
- The main confusions are between semantically close emotions: **Joy → Love (50 samples)** and **Fear → Surprise (22 samples)**. Sadness is also sometimes confused with Joy, Anger and Fear.
- The model **starts to overfit after epoch 3**: training accuracy keeps rising (to about 98%) while validation loss increases. EarlyStopping with `restore_best_weights` handles this, and the small gap between validation (90.75%) and test (89.05%) accuracy shows the restored model generalises well.
- Custom sentence tests were mostly correct, but "i adore you and i want to spend my whole life with you" was predicted as Joy (72%) instead of Love, which matches the Joy/Love confusion above.

---

## ✅ Conclusion

An Embedding + LSTM network was successfully trained to classify text into six emotions on the Emotion dataset, reaching **89.05% test accuracy, 0.8535 macro F1 and 0.8930 weighted F1**. The LSTM captures word order and context, which a bag-of-words approach cannot, and it classifies unseen text reliably. Class weights improved recall on rare classes, and EarlyStopping limited overfitting by restoring the best epoch.

**Limitations:** class imbalance (Surprise and Love have lower precision), confusion between similar emotions (Joy/Love, Fear/Surprise), and embeddings learned from scratch on a small dataset.

**Possible improvements:** Bidirectional LSTM / GRU, pre-trained embeddings (GloVe, Word2Vec, FastText), attention mechanisms, transformer models such as BERT / DistilBERT, and hyper-parameter tuning.

---

## 📁 Repository Structure

```
GenAI-Text-Classification-using-Embedding-and-LSTM/
│
├── README.md                          ← this file
├── 202401110046_Sneha_Practical_02_LSTM.ipynb    ← main notebook (executed, with outputs)
└── Practical_02_Report.pdf            ← assignment report
```

The dataset is loaded from Hugging Face inside the notebook, so no dataset folder is needed.

---

## ⚙️ How to Run

**Option 1 — Google Colab (recommended)**

1. Open `202401110046_Sneha_Practical_02_LSTM.ipynb` in Google Colab (`File → Upload notebook`).
2. Select a GPU: `Runtime → Change runtime type → T4 GPU` (optional, faster).
3. Run all cells: `Runtime → Run all`.

**Option 2 — Local Jupyter**

```
git clone https://github.com/Sneha529-oss/GenAI-Text-Classification-using-Embedding-and-LSTM.git
cd GenAI-Text-Classification-using-Embedding-and-LSTM
pip install tensorflow datasets numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook 202401110046_Sneha_Practical_02_LSTM.ipynb
```

Training takes about one minute on a GPU (7 epochs, ~2 s per epoch).

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

## Declaration

I, **Sneha Chaurasia**, confirm that this implementation was prepared for Practical Assignment 2 and that the results presented are generated from the experiments performed in this notebook.

**GitHub Repository Link:** <https://github.com/Sneha529-oss/Text-Classification-using-Embedding-Layer-and-LSTM-Practical-No.-02-Gen-AI>
