# MovieBot

MovieBot is a conversational movie recommendation system that suggests movies based on the user's preferred genre.

## Features

- Interactive chatbot interface for movie recommendations
- Genre-based movie suggestion mechanism
- Modular codebase with separate components for training and inference
- Dockerized setup for easy deployment

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YogPandya12/MovieBot.git
   cd MovieBot
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

## Usage

Once the application is running, you can interact with the bot through the provided interface. Simply input your preferred movie genre, and the bot will suggest movies accordingly.

## Project Structure

- `app.py` / `application.py`: Main application files to run the bot
- `inference_pipeline.py`: Handles the inference logic for movie recommendations
- `train_model_pipeline.py`: Contains the training pipeline for the recommendation model
- `templates/` and `static/`: Frontend templates and static files for the web interface
- `Dockerfile` and `docker-compose.yml`: Docker configurations for containerized deployment

## License

This project is licensed under the MIT License.
