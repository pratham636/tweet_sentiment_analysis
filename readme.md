# 🐦 Tweet Sentiment Analysis

This project implements a complete machine learning pipeline to classify tweets into Positive or Negative sentiments. It utilizes Natural Language Processing (NLP) to clean raw text, extract meaningful features, and compare the performance of three different classification algorithms.
## 📂 Project Structure

Following the organization in the repository:Plaintext
```TWEET_SENTIMENT_ANALYSIS
├── charts/
│   └── pie_chart.png          # Visual distribution of sentiment labels
├── data/
│   └── tweets.csv             # Raw dataset used for training/testing
├── venv/                      # Python virtual environment
├── .dockerignore              # Files excluded from Docker builds
├── Dockerfile                 # Instructions for containerization
├── main.py                    # Core script for cleaning, training, and evaluation
├── readme.md                  # Project documentation
├── requirements.txt           # Required Python libraries
└── tempCodeRunnerFile.py      # Temporary execution file
```
## 🧹 Data Preprocessing

The clean_text function in main.py prepares the raw tweet data for the models:Case Normalization: Converts all text to lowercase.Whitespace Removal: Trims leading and trailing spaces.Noise Filtering: Uses Regular Expressions to remove URLs (http/www) and User Handles (@mentions).Character Filtering: Removes all non-alphabetic characters, ensuring only text remains.
## ⚙️ Feature Engineering

The project uses TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization:N-grams: Includes both individual words and word pairs (ngram_range=(1,2)) to capture context.Max Features: Limits the vocabulary to the top 50,000 most significant terms to optimize performance.
## 🤖 Machine Learning Models

Three models are trained and evaluated to find the most accurate classifier:Logistic Regression: Implemented with balanced class weights.Multinomial Naive Bayes: Configured with a 0.5/0.5 class prior.Linear Support Vector Machine (LinearSVC): Utilizes balanced class weights for optimal separation.
## 📈 Evaluation Metrics

Each model is assessed using a comprehensive suite of metrics:Accuracy: Overall percentage of correct predictions.Precision & Recall: Measuring the quality and coverage of positive sentiment detection.F1-Score: The harmonic mean of precision and recall.Confusion Matrix: A detailed breakdown of true positives, true negatives, and misclassifications.
## 🚀 Getting Started

InstallationEnsure Python is installed.Install the necessary dependencies:
```
Bash
pip install -r requirements.txt
```
Running the AnalysisExecute the main script to train the models and generate performance reports:Bash
```
python main.py
```
## 🛠️ Tech Stack
Language: PythonLibraries: Pandas, Scikit-Learn, Matplotlib, 
ReDevOps: Docker
# 👤 Author
Pratham
Student, University of Passau
# 📊 Model Results (Example)
After running your script, update these values based on your terminal output:
| Model | Accuracy |Precision|Recall|| F1-Score |
| :--- | :---: | :---: |
| Logistic Regression | 0.81|0.57|0.81|0.67|
| Naive Bayes | 0.80|0.55|0.78|0.65 |
| Linear SVM | 0.81|0.56|0.80|0.66|
