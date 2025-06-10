# FakeTweetDetection
# DeepFake Tweet Detection System

## Introduction
This project is a deep learning-based system designed to detect machine-generated tweets (deepfake texts). The rapid evolution of deepfake technology poses significant risks to public opinion and misinformation. This system identifies bot-generated tweets to ensure the integrity of social media platforms.

---

## Objective
Develop a Convolutional Neural Network (CNN) model using FastText embeddings to classify tweets as either human-authored or bot-generated. The project also includes a Django-based web application for real-time predictions.

---

## Features
- **Real-Time Detection**: Classifies tweets as "Normal" or "Bot Fake" within seconds.
- **Model Performance**: CNN-based architecture achieving 93% accuracy.
- **Feedback Mechanism**: Allows users to provide feedback for model improvement.
- **Scalable Architecture**: Supports deployment on local or cloud platforms.

---

## System Architecture
The pipeline includes:
1. **Data Collection**: Tweets stored in a database for processing.
2. **Preprocessing**: Cleaning, tokenization, and standardization.
3. **Embedding**: FastText embeddings for semantic representation.
4. **Classification**: CNN-based model for detection.
5. **Real-Time Predictions**: Outputs results in under 2-3 seconds.

![System Architecture](https://github.com/ChanduPT/FakeTweetDetection/blob/main/DeepApp/static/images/investor2.jpg)

---

## Tools and Technologies
- **Programming**: Python
- **Libraries**: Pandas, NumPy, TensorFlow/Keras, NLTK
- **Web Framework**: Django
- **Visualization**: Matplotlib
- **Data Storage**: SQLite
- **Deployment**: Local server or cloud platforms (AWS, Google Cloud)

---

## Dataset
- **Source**: [TweepFake Dataset on Kaggle](https://www.kaggle.com/datasets/mtesconi/twitter-deep-fake-text)
- **Composition**: 25,572 tweets (50% human-generated, 50% bot-generated)
- **Bot Techniques**: Markov Chains, RNNs, LSTMs, GPT-2, Gpt-3, Gpt-4

---

## Usage
1. Start the server by running `run.bat`.
2. Navigate to `http://127.0.0.1:8000/index.html` in a browser.
3. Sign in to access the detection tool.
4. Enter a tweet and click "Submit" for classification results.

---


## Future Work
- Incorporate advanced models like BERT or RoBERTa.
- Expand detection capabilities to other social media platforms.
- Utilize quantum NLP for enhanced detection accuracy.
---

