

from inference_pipeline import chat

def main():
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        reply = chat(user_input)
        print(f"Bot: {reply}")

if __name__ == "__main__":
    main()
