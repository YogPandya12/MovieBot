import random
import json
import numpy
import torch

from src.components.train_model.model import NeuralNet
from src.components.train_model.nltk_utils import bag_of_words, tokenize, stem

def chat(given_statement):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('artifacts\intents.json', 'r') as json_data:
        intents = json.load(json_data)

    FILE = "artifacts\data.pth"
    data = torch.load(FILE, map_location=device)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    # Genre keywords dictionary
    genre_keywords = {
        "action": "Action",
        "horror": "Horror",
        "thriller": "Thriller",
        "western": "Western",
        "drama": "Drama",
        "fantasy": "Fantasy",
        "adventure": "Adventure",
        "romance": "Romance",
        "comedy": "Comedy",
        "mystery": "Mystery",
        "sci-fi": "Sci-Fi",
        "animation": "Animation",
        "supernatural": "Supernatural",
        "religious": "Religious",
        "biography": "Biography",
        "suspense": "Suspense"
    }

    while True:
        sentence = given_statement
        if sentence.lower() == "quit":
            break

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
    
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    response = random.choice(intent['responses'])
                    if '{movie_title}' in response:
                        # Detect genre preference from the user's query
                        detected_genre = None
                        for word in sentence:
                            word_stemmed = stem(word)
                            if word_stemmed in genre_keywords:
                                detected_genre = genre_keywords[word_stemmed]
                                break
                    
                        # If a genre is detected, choose a movie from that genre
                        if detected_genre and detected_genre in intents['genres']:
                            movie = random.choice(intents['genres'][detected_genre])
                            response = response.replace('{movie_title}', f"{movie['title']} ({movie['language']})")
                        else:
                            genre = random.choice(list(intents['genres'].keys()))
                            movie = random.choice(intents['genres'][genre])
                            response = response.replace('{movie_title}', f"{movie['title']} ({movie['language']})")
                    return (f"{response}")
        else:
            return (f"I do not understand...")
