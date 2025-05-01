MovieBot
MovieBot is a conversational movie recommendation system that suggests movies based on the user's preferred genre.​

Features
Interactive chatbot interface for movie recommendations

Genre-based movie suggestion mechanism

Modular codebase with separate components for training and inference

Dockerized setup for easy deployment​
GitHub

Installation
Clone the repository:​

bash
Copy
Edit
git clone https://github.com/YogPandya12/MovieBot.git
cd MovieBot
Install the required dependencies:​

bash
Copy
Edit
pip install -r requirements.txt
Run the application:​

bash
Copy
Edit
python app.py
Usage
Once the application is running, you can interact with the bot through the provided interface. Simply input your preferred movie genre, and the bot will suggest movies accordingly.​

Project Structure
app.py / application.py: Main application files to run the bot

inference_pipeline.py: Handles the inference logic for movie recommendations

train_model_pipeline.py: Contains the training pipeline for the recommendation model

templates/ and static/: Frontend templates and static files for the web interface

Dockerfile and docker-compose.yml: Docker configurations for containerized deployment​
GitHub
