import base64
import io
import os
import pickle
import csv
from string import punctuation
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D, BatchNormalization
from keras.models import Sequential, load_model, Model
from keras.utils import to_categorical
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer  # loading tfidf vector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from django.shortcuts import render, redirect
from django.contrib import messages
import emoji
import re
import contractions
import logging

global username

# define object to remove stop words and other text processing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
accuracy, precision, recall, fscore = [], [], [], []
X, Y, sc, tfidf_vectorizer = None, None, None, None

# Function to expand contractions in text
def expand_contractions(text):
    return contractions.fix(text)

# Function to remove URLs
def remove_urls(text):
    return re.sub(r'http\S+|www\S+', '', text)

# Function to convert emojis to text
def emoji_to_text(text):
    return emoji.demojize(text)

# define function to clean text by removing stop words and other special symbols
def cleanText(doc):
    doc = expand_contractions(doc)  # Expand contractions
    doc = remove_urls(doc)          # Remove URLs
    doc = emoji_to_text(doc)        # Remove emojis
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [ps.stem(token) for token in tokens]
    return ' '.join(tokens)

# Preprocessing dataset
if os.path.exists("model/X.npy"):
    with open('model/tfidf.pckl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    X = np.load("model/X.npy")
    Y = np.load("model/Y.npy")
else:
    dataset = pd.read_csv("/Users/Downloads/DeepFake2/Dataset/combined_tweepfake_gpt4_dataset.csv", sep=";")
    dataset = dataset.dropna()
    # print(np.unique(dataset['account.type'], return_counts=True))
    # print(dataset)
    # dataset = dataset.values
    X, Y = [], []
    for i in range(len(dataset)):
        tweet = dataset.iloc[i, 1]
        tweet = tweet.strip("\n").strip().lower()
        label = dataset.iloc[i, 2]
        tweet = cleanText(tweet)  # clean description
        X.append(tweet)
        Y.append(1 if label == 'bot' else 0)
        # print(str(i) + " " + str(len(tweet)) + " " + str(label))
    X = np.asarray(X)
    Y = np.asarray(Y)
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=True,
                                        max_features=1000, ngram_range=(1,2))
    X = tfidf_vectorizer.fit_transform(X).toarray()
    f = open('model/tfidf.pckl', 'wb')
    pickle.dump(tfidf_vectorizer, f)
    f.close()
    np.save("model/X", X)
    np.save("model/Y", Y)

sc = StandardScaler()
X = sc.fit_transform(X)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)  # split dataset into train and test

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

# Training ML models
models = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
}

for name, model in models.items():
    model.fit(X_train[:1000], y_train[:1000])
    predict = model.predict(X_test[:200])
    calculateMetrics(name, predict, y_test[:200])

X_train1 = np.reshape(X_train, (X_train.shape[0], 50, 10, 2))
X_test1 = np.reshape(X_test, (X_test.shape[0], 50, 10, 2))
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)

cnn_model = Sequential()
cnn_model.add(
    Convolution2D(32, (3, 3), input_shape=(X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Convolution2D(64, (3, 3), activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(units=256, activation='relu'))
cnn_model.add(Dense(units=y_train1.shape[1], activation='sigmoid'))
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
if not os.path.exists("model/cnn_weights.hdf5"):
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose=1, save_best_only=True)
    hist = cnn_model.fit(X_train1, y_train1, batch_size=8, epochs=50, validation_data=(X_test1, y_test1),
                         callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
else:
    cnn_model.load_weights("model/cnn_weights.hdf5")

predict = cnn_model.predict(X_test1)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test1, axis=1)
calculateMetrics("CNN Algorithm", predict, y_test1)

hybrid_model = Model(cnn_model.inputs, cnn_model.layers[-2].output)  # create mobilenet  model
hybrid_features = hybrid_model.predict(X_test1)  # extracting mobilenet features
print(hybrid_features.shape)
Y = y_test
X_train, X_test, y_train, y_test = train_test_split(hybrid_features, Y, test_size=0.2)
rf = RandomForestClassifier() # create random forest object
rf.fit(X_train, y_train)  # train on mobileenet features
predict = rf.predict(X_test)  # perform prediction on test data
calculateMetrics("Extension Hybrid CNN", predict, y_test)  # call function to calculate accuracy and other metrics

# logger = logging.getLogger('django') # To check logs

def LoadDataset(request):
    if request.method == 'GET':
        dataset = pd.read_csv("/Users/Downloads/DeepFake2/Dataset/combined_tweepfake_gpt4_dataset.csv", sep=";")
        # logger.info(f"Dataset shape: {dataset.shape}")
        dataset = dataset.dropna()
        dataset = dataset.values
        output = ''
        output += ('<table border=1 align=center width=100%><tr><th><font size="" color="black">Username</th><th><font '
                   'size="" color="black">Tweet</th><th><font size="" color="black">Account Type</th>')
        output += '</tr>'
        for i in range(0, 100):
            output += '<tr><td><font size="" color="black">' + dataset[i, 0] + "</td>"
            output += '<td><font size="" color="black">' + dataset[i, 1] + "</td>"
            output += '<td><font size="" color="black">' + dataset[i, 2] + "</td></tr>"
        context = {'data': output}
        return render(request, 'ViewOutput.html', context)

def FastText(request):
    if request.method == 'GET':
        global X
        context = {'data': str(X)}
        return render(request, 'ViewOutput.html', context)

def TrainML(request):
    if request.method == 'GET':
        global precision, recall, fscore, accuracy
        algorithms = ['Naive Bayes', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting',
                      'Propose CNN', 'Extension Hybrid CNN']

        metrics = ['Precision', 'Recall', 'F1 Score', 'Accuracy']

        # Generate tabular data for rendering
        table_data = [
            f"<tr><td><font size='' color='black'>{algorithms[i]}</td>"
            f"<td><font size='' color='black'>{accuracy[i]}</td>"
            f"<td><font size='' color='black'>{precision[i]}</td>"
            f"<td><font size='' color='black'>{recall[i]}</td>"
            f"<td><font size='' color='black'>{fscore[i]}</td></tr>"
            for i in range(len(algorithms))
        ]
        table_html = (
                "<table border=1 align=center width=100%>"
                "<tr><th><font size='' color='black'>Algorithm Name</th>"
                "<th><font size='' color='black'>Accuracy</th>"
                "<th><font size='' color='black'>Precision</th>"
                "<th><font size='' color='black'>Recall</th>"
                "<th><font size='' color='black'>F1 Score</th></tr>"
                + "".join(table_data) + "</table><br/>"
        )

        # Flattening all metrics into a single list for DataFrame creation
        data = [
            [algorithms[i], metrics[j], [precision, recall, fscore, accuracy][j][i]]
            for i in range(len(algorithms))
            for j in range(len(metrics))
        ]

        # Create DataFrame for plotting
        df = pd.DataFrame(data, columns=['Algorithms', 'Metrics', 'Value'])

        # Generate performance graph
        matplotlib.use('Agg')  # Use non-GUI backend for rendering
        df.pivot_table(index="Algorithms", columns="Metrics", values="Value").plot(kind='bar', figsize=(8, 4))
        plt.title("All Algorithms Performance Graph")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        # Render HTML with table and graph
        context = {'data': table_html, 'img': img_b64}
        return render(request, 'ViewGraph.html', context)


def DetectFakeAction(request):
    if request.method == 'POST':
        global username, hybrid_model, tfidf_vectorizer, sc
        tweet = request.POST.get('t1', False)
        data = tweet.strip().lower()
        data = cleanText(data)
        temp = [data]
        temp = tfidf_vectorizer.transform(temp).toarray()
        dl_model = load_model("model/cnn_weights.hdf5")
        temp = sc.transform(temp)
        temp = np.reshape(temp, (temp.shape[0], 50, 10, 2))
        predict = dl_model.predict(temp)
        predict = np.argmax(predict)
        print(predict)
        output = "Normal"
        if predict == 1:
            output = "Bot Fake"
        context = {'data': 'Given Tweet Detected as : ' + output}
        return render(request, 'DetectFake.html', context)

def DetectFake(request):
    if request.method == 'GET':
        return render(request, 'DetectFake.html', {})

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})

def UserLogin(request):
    if request.method == 'GET':
        return render(request, 'UserLogin.html', {})

def Register(request):
    if request.method == 'GET':
        return render(request, 'Register.html', {})

def UserLoginAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        status = "UserLogin.html"
        context = {'data': 'Invalid login details'}
        if "admin" == username and "password" == password:
            context = {'data': "Welcome " + username}
            status = 'UserScreen.html'
        return render(request, status, context)

def FeedbackPage(request):
    return render(request, 'Feedback.html')  # Render the feedback form page

def SubmitFeedback(request):
    if request.method == 'POST':
        screen_name = request.POST.get('screen_name', '').strip()
        tweet = request.POST.get('tweet', '').strip()
        category = request.POST.get('category', '').strip()
        source = request.POST.get('source', '').strip()

        # Validate inputs
        if not screen_name or not tweet or not category:
            messages.error(request, "Invalid feedback! Please fill in all required fields.")
            return redirect('DetectFake')

        """
        # Load existing dataset and check for duplicates
        dataset_path = '/Users/Downloads/DeepFake2/Dataset/combined_tweepfake_gpt4_dataset.csv'
        try:
            existing_data = pd.read_csv(dataset_path, sep=';')
        except Exception as e:
            messages.error(request, f"Could not read the dataset: {e}")
            return redirect('DetectFake')

        if tweet in existing_data['text'].values:
            messages.warning(request, "This tweet already exists in the dataset. Thank you!")
            return redirect('DetectFake')
        """

        # use either of the approaches 
        # Map feedback to dataset columns
        account_type = "human" if category.lower() == "human" else "bot"
        class_type = "human" if category.lower() == "human" else source

        dataset_path = '/Users/Downloads/DeepFake2/Dataset/feedback.csv'

        # Append feedback to the dataset
        try:
            with open(dataset_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow([screen_name, tweet, account_type, class_type])
            messages.success(request, "Thank you for your feedback!.")
            return redirect('DetectFake')  # Redirect back to DetectFake page
        except Exception as e:
            messages.error(request, f"An error occurred while saving feedback: {e}")
            return redirect('DetectFake')
