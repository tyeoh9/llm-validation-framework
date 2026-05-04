"""Example: stream a chatbot response in the terminal."""

from Chatbot import Chatbot

if __name__ == "__main__":
    bot = Chatbot()

    question = input("Ask something: ").strip()
    if not question:
        print("No question provided.")
        raise SystemExit

    print("\nResponse:")
    for token in bot.stream(question):
        print(token, end="", flush=True)
    print()
