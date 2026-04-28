# Kazakh Punctuation Restoration — WISH Hackathon

This repository contains a solution for the **WISH Hackathon** (Kaggle competition) focused on restoring punctuation marks in Kazakh language texts.

## 📌 Task Overview
The goal is to take lowercase, punctuation-stripped text (common in Automatic Speech Recognition output) and predict the correct punctuation marks to make the text readable and processable for downstream NLP tasks.

**Task Type:** Token Classification
**Evaluation Metric:** Macro F1-score (over `COMMA`, `PERIOD`, and `QUESTION`)

### Labels:
For each word, the model predicts one of the following:
* `O`: No punctuation
* `COMMA`: Comma follows (`,`)
* `PERIOD`: Period or Exclamation mark follows (`.`, `!`)
* `QUESTION`: Question mark follows (`?`)

**Example:**
* *Input:* `сәлем менің атым бақыт қалың қалай`
* *Output:* `COMMA O O PERIOD O QUESTION`
* *Result:* `Сәлем, менің атым Бахыт. Қалың қалай?`

## 🛠 Tech Stack
* **Language:** Python
* **Frameworks:** `transformers`, `torch`, `datasets`
* **Base Model:** `xlm-roberta-base`

## 🚀 Key Implementation Details

1.  **Data Augmentation:**
    To improve performance, the training set was augmented with the `kz-transformers/multidomain-kazakh-dataset`, adding approximately 40,000 sentences to help the model learn diverse linguistic structures.

2.  **Preprocessing:**
    * Used `XLMRobertaTokenizer` with subword alignment.
    * Mapped labels to tokens while handling subword splitting using a `-100` ignore index for loss calculation.

3.  **Training Strategy:**
    * Fine-tuned `xlm-roberta-base` using the Hugging Face `Trainer` API.
    * **Hyperparameters:** 2e-5 learning rate, 16 batch size, and weight decay of 0.01.

4.  **Linguistic Post-processing (Heuristics):**
    To boost the Macro F1-score, custom heuristics were applied:
    * **Question Detection:** Identified question particles (e.g., `ма`, `ме`, `ба`, `бе`) to force-assign the `QUESTION` label at sentence endings.
    * **Conjunction Rules:** Applied rules for common conjunctions (e.g., `бірақ`, `алайда`, `өйткені`) where a `COMMA` is grammatically required.

## 📊 Evaluation
The model is evaluated using **Macro F1-score** across the three punctuation classes. The `O` class is excluded from scoring as it represents ~85% of the data.

$$Macro F1 = \frac{F1_{COMMA} + F1_{PERIOD} + F1_{QUESTION}}{3}$$

## 📂 Project Structure
* `genielablast.ipynb`: Core notebook containing data loading, training, and inference.
* `submission_big_model.csv`: Final prediction file for the competition.

## 👥 Authors
Work was done individually in group
Developed for the WISH Hackathon competition.
