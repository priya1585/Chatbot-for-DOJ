import pandas as pd

"""load ths csv file with the exact location of the file and add the name of the csv file at the end of the 
   location address(....../name_of_csv_file.csv)"""

data = pd.read_csv(r'C:\Users\MR\Desktop\Smart India Hackthon\SampleBot\data.csv')

data.dropna(inplace=True)# Drop any rows with NaN values

# Convert the questions and answers to lists
questions = data['Question'].tolist()# tolist() function converts a series into a list
answers = data['Answer'].tolist()
images = data['Image'].tolist()

# Print the loaded data
print(questions)
print(answers)

import nltk

nltk.download('punkt') #This tokenizer divides a text into a list of sentences
nltk.download('punkt_tab')
nltk.download('stopwords')

import sklearn
"""contains a lot of efficient tools for machine learning and statistical modeling including classification, 
regression, clustering and dimensionality reduction."""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Preprocess function to clean and tokenize text
def preprocess(text):
    print(text)
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    print(tokens)
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    print(filtered_tokens)
    return ' '.join(filtered_tokens)

# Preprocess the questions
processed_questions = [preprocess(question) for question in questions]
print(processed_questions)

# Correct usage of TfidfVectorizer
vectorizer = TfidfVectorizer() #why vectorizer? To map words or phrases from vocabulary to a corresponding vector of real numbers
question_vectors = vectorizer.fit_transform(processed_questions)
print(question_vectors)

# Function to find the best matching answer
def get_answer(user_question):
    # Preprocess the user's question
    user_question = preprocess(user_question)
    print(user_question)

    # Vectorize the user's question
    user_vector = vectorizer.transform([user_question])
    print(user_vector)

    # Compute cosine similarity between user question and data questions
    similarities = cosine_similarity(user_vector, question_vectors)
    print(similarities)

    # Find the index of the most similar question
    closest_match_index = np.argmax(similarities)
    print(closest_match_index)

    # If the best match has a very low similarity score, the chatbot might not understand the question
    if similarities[0, closest_match_index] < 0.1:
        return "Sorry, I don't understand your question."

    # Return the answer corresponding to the best match
    return answers[closest_match_index]

from PIL import Image
import requests
import matplotlib.pyplot as plt

def display_answer_with_image(index):
    # Display the answer
    print(f"Answer: {answers[index]}")

    # Load and display the image
    img_path = images[index]
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.show()

# Example function to get answer and image based on the user's question
def get_answer(user_question):
    for i, question in enumerate(questions):
        if user_question.lower() in question.lower():
            display_answer_with_image(i)
            return

    print("Sorry, I don't understand your question.")

# Start the chat loop
def chat():
    print("DoJ Chatbot: How can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("DoJ Chatbot: Goodbye!")
            break

        get_answer(user_input)

# Start the chatbot
chat()
