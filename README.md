🧠 Jigsaw Rule Violation Classifier (DeBERTa-v3 Fine-Tuning)

This project fine-tunes the DeBERTa-v3-base transformer model to detect rule violations in Reddit comments for the Jigsaw Agile Community Rules Kaggle competition.

The model predicts whether a given Reddit comment breaks a specific rule, based on text and contextual rule information.

🚀 Project Overview

This notebook automates:

Loading competition datasets using KaggleHub

Cleaning and preprocessing Reddit comments

Building contextual model inputs (Rule + Comment)

Tokenizing text with DeBERTa-v3-base

Training and evaluating a sequence classification model

Generating a submission.csv file for Kaggle

🧩 Tech Stack
Category	Tools & Libraries
Framework	Transformers
, Datasets
, PyTorch

Model	DeBERTa-v3-base

Data Handling	pandas, NumPy, scikit-learn
Automation	KaggleHub
Metrics	ROC-AUC, Accuracy
⚙️ Features

✅ Downloads and loads the Jigsaw competition dataset and DeBERTa-v3 model
✅ Cleans noisy Reddit text (links, markdown, special chars)
✅ Combines rule and comment for contextual understanding
✅ Fine-tunes the model with custom metrics (AUC + accuracy)
✅ Automatically saves best-performing model
✅ Generates a submission.csv file ready for Kaggle upload

📁 Project Structure
submission_csv.py        # Main training + submission generation script
submission.csv           # (Generated output file)

🧠 Model Details

Base Model: DeBERTa-v3-base

Task: Binary Classification (Rule Violation Detection)

Input Format:

Rule: [RULE_TEXT] [SEP] Comment: [COMMENT_TEXT]


Loss Function: Binary Cross-Entropy

Optimizer: AdamW

Metrics: ROC-AUC, Accuracy

⚡ How to Run

Clone this repository:

git clone https://github.com/<your-username>/jigsaw-rule-violation-classifier.git
cd jigsaw-rule-violation-classifier


Install dependencies:

pip install transformers tokenizers datasets accelerate pyarrow==15.0.0


Log in to Kaggle:

import kagglehub
kagglehub.login()


Run the script:

python submission_csv.py


After training, your submission file will be generated as:

submission.csv

🧾 Example Output
Train shape: (20000, 3)
✅ Text cleaning done! Example:
0    This is an example Reddit comment...
1    Another cleaned comment example...
Train split: (18000, 2), Valid split: (2000, 2)
✅ Submission file created successfully!

🏆 Results
Metric	Validation
ROC-AUC	~0.91
Accuracy	~0.84

(Your results may vary depending on random seed and hardware)

📜 License

This project is released under the MIT License.

👨‍💻 Author

Almizan Khan

Kaggle: @almizankhan

GitHub: @almizankhan
