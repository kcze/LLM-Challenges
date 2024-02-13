from llm import llm

def main():

    print("Starting the conversation. Type 'quit' to exit.")

    api_key = ""
    # I didn't want to just put the API key in the code for security reasons
    # So I made the .api file and added it in the .gitignore
    with open(".api", "r") as file:
        api_key = file.read()
    chat_helper = llm(api_key)

    while True:
        user_input = input("\n\033[92mYou:\033[0m ")
        if user_input.lower() == 'quit':
            break

        try:
            response = chat_helper.send_message(user_input)
            print("\n\033[95mAI:", response, "\033[0m")

        except ValueError as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    main()