from openai import OpenAI
import tiktoken

class Message:
    # Static tokenizer to count the number of tokens
    tokenizer = tiktoken.encoding_for_model("gpt-4")

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        #NOTE it would be good to reconsider the choice of putting mutable is_summarized variable
        self.is_summarized = False # Whether the message has been summarized and added to the memories
        self.token_count = len(Message.tokenizer.encode(content)) # the number of tokens in the message

    def get(self) -> dict[str, str]:
        """
        Returns the message as a dictionary in OpenAI format.

        Returns:
            {"role": str, "content": str}
        """
        return {"role": self.role, "content": self.content}

# Memory management: summarization
class llm:
    # I added the api param, I'm aware that this could break other's people code so I made it optional
    def __init__(self, api: str = ""):
        self.full_message_history = [] # This is the full conversation history https://platform.openai.com/docs/api-reference/chat/object . 
        self.memory = ""
        self.client = OpenAI(api_key=api)
        self.DEBUG = True # Set this to True to see the context window being sent to the LLM.
        self.max_message_tokens = 128 # The maximum number of tokens for messages in the context, excluding memory
        # Overflowing messages will be summarized and added to the memory
        self.max_memory_tokens = 128 # The maximum number of tokens for the memory
        if self.client.api_key == '':
            raise ValueError("\033[91m Please enter the OpenAI API key which was provided in the challenge email into llm constructor.\033[0m")

    def manage_context_window(self):
        """
        #~#~#~# CONTEXT WINDOW MANAGEMENT CHALLENGE #~#~#~#
        This function creates the context window to be sent to the LLM. The context window is managed list of messages, and constitutes all the information the LLM knows about the conversation.

        Returns:
            list: The context window to be sent to the LLM.
        """
        context = []
        # First add messages from the full history from the newest to the context till the max_message_tokens is reached
        message_tokens = 0
        last_index = 0  # To keep track of where we left off

        # Iterate over the messages in reversed order
        for index, message in enumerate(reversed(self.full_message_history)):
            message_tokens += message.token_count
            if message_tokens > self.max_message_tokens:
                message_tokens -= message.token_count
                last_index = len(self.full_message_history) - 1 - index
                break
            context.insert(0, message.get())
        print(f"# Message tokens: {message_tokens}")

        # Summarize the rest of unsummarized messages
        # Need to iterate from the beginning so the messages are summarized in the correct order
        to_summarize = []
        for message in self.full_message_history[:last_index]:
            if not message.is_summarized:
                to_summarize.append(message)

        if len(to_summarize) > 0:
            self.memory = self.summarize(self.memory, to_summarize)

        # Add the memory to the beginning of the context
        if len(self.memory) > 0:
            print(f"# Memory tokens: {len(Message.tokenizer.encode(self.memory))}")
            print(f"{self.memory}")
            print("#################")
            context.insert(0, {"role": "system", "content": self.memory})
                
        return context

    def summarize(self, memory: str, new_messages: list[Message]) -> str:
        all_messages = ""
        for message in new_messages:
            all_messages += f"[{message.role}] {message.content}\n"
            message.is_summarized = True

        # Summarize the messages
        # Using research-based "expert", "please", "important for my carrer", not using negatives
        # Could be extended using chain-of-thought and think before summarizing
        # The most important is beginning and end.
        messages = [
            {"role": "system", "content": """You are an expert summarizer.
            Please summarize the following messages into a single message, this is important for my career.
            Minimise the length of the summary but keep all the important information.
            You are a helpful assistant so please include all important details about the user and main points of the conversation.
            You are provided the current summary and the new messages.
            Respond with the summary only."""},
            {"role": "user", "content": f"{self.memory} + {all_messages}"},
        ]

        for i in range(3):
            summary = self.gpt4_conversation(messages).choices[0].message.content
            tokens = Message.tokenizer.encode(summary)
            # If summary is short enough, break
            if len(tokens) <= self.max_memory_tokens:
                break
            else:
                # Otherwise, regenerate
                print(f"!!! Summary too long: {len(tokens)} tokens")
                messages.append({"role": "assistant", "content": summary})
                messages.append({"role": "user", "content": "The summary is too long, please try again and make it shorter."})
                # If it's still too long after 3 tries, truncate it
                # That isn't good but better than regenerating indefinitely
                if i == 2:
                    summary = Message.tokenizer.decode(tokens[:self.max_memory_tokens])
                    print(f"!!! Summary cut off: {summary} tokens")
                    break
            
        return summary

    def send_message(self, prompt: str, role: str = 'user', json_response: bool = False):
        """
        This function adds the provided prompt to the existing message history, creating a context window for the LLM. 
        This context window is forwarded to the LLM. The received response from the LLM is then appended to the message history and returned.

        Args:
            prompt (str): The message content to be sent to the LLM.
            role (str, optional): The role of the speaker. Defaults to 'user'.
                - 'user': Represents the user speaking.
                - 'assistant': Represents the AI assistant speaking.
                - 'system': Represents instructions or context for the AI.
            json_response (bool, optional): Specifies whether the response should be returned as JSON. Defaults to False. If True, schema must be specified.

        Returns:
            str: The AI's response to the message.

        Raises:
            ValueError: If an invalid role is provided.

        Note:
            - The `context_window` is currently using a primitive way of managing conversation history, simply keeping only the last 4 messages.

        Example:
            >>> helper = OpenAIHelper()
            >>> response = helper.send_message("Hello, how can I assist you?")
            >>> print(response)
            "Sure, I can help you with that!"
        """
        if role == 'user':
            # self.full_message_history.append({'role': 'user', 'content': prompt})
            self.full_message_history.append(Message('user', prompt))
        elif role == 'assistant':
            # self.full_message_history.append({'role': 'assistant', 'content': prompt})
            self.full_message_history.append(Message('assistant', prompt))
        elif role == 'system':
            # self.full_message_history.append({'role': 'system', 'content': prompt})
            self.full_message_history.append(Message('system', prompt))
        else:
            raise ValueError("Invalid role provided. Valid roles are 'user', 'assistant', or 'system'.")

        context_window = self.manage_context_window()

        if self.DEBUG:
            print(f"\033[91m  Context sent to LLM:\n  {context_window} \033[0m")

        # Send the message to the LLM and get the response.
        response = self.gpt4_conversation(context_window)

        ai_message = response.choices[0].message.content

        self.full_message_history.append(Message('assistant', ai_message))
        return ai_message

    #~#~#~# Methods for interacting with OpenAI's Chat Completions EndPoint #~#~#~#
    def gpt4_conversation(self, messages: list, json_response: bool = False, model: str = "gpt-4-1106-preview"):
        """
        Initiates a conversation with the GPT-4 language model using the specified parameters.

        This method sends a list of messages to the GPT-4 model and retrieves the model's response. It allows configuration 
        of the model and response format.

        Args:
            messages (list): A list of message objects (dicts) that form the conversation history (context window).
            json_response (bool, optional): If True, forces the response to be in JSON format. Requires a defined response schema somewhere in the messages.
                                            Defaults to False.
            model (str, optional): The specific GPT-4 model version to be used for the conversation. Defaults to "gpt-4-1106-preview".

        Returns:
            response: The response from the GPT-4 model. Format: https://platform.openai.com/docs/api-reference/chat/object

        Raises:
            ValueError: If the combined token count of the response and context exceeds the 4096 token limit.

        Note:
            - The 'temperature' parameter controls the randomness of the model's responses. A higher value increases randomness.
            - The 'max_tokens' parameter sets the limit for the response token count. Exceeding this limit triggers a ValueError.
            - The method currently imposes a shorter context window limit for this specific implementation.
        
        Example:
            >>> response = gpt4_conversation([{'role': 'user', 'content': 'Hello, AI!'}])
            >>> print(response)
        """
        # Initialize the conversation with the GPT-4 model
        response = self.client.chat.completions.create(
            model=model,  # Specifies the GPT-4 model version
            temperature=0.79,  # Sets the AI's creativity level. Higher values increase randomness.
            max_tokens=4096,  # Sets the maximum number of tokens in the AI's response.
            response_format={"type": "json_object"} if json_response else None,  # Optional JSON response format
            messages=messages  # The conversation history to be sent to the model
        )

        # Check if the total token usage exceeds the limit
        # DO NOT REMOVE THIS - This is a requirement for the challenge - you can make it smaller if you like
        if response.usage.total_tokens > 4096:
            raise ValueError("CHALLENGE CONTEXT WINDOW EXCEEDED: The context window now exceeds the 4096 token limit. Please try again with a shorter prompt.")

        return response

    #NOTE this was outside of the class!!!
    def gpt4_one_shot(self, system_prompt: str, user_prompt: str, json_response: bool = False, model: str = "gpt-4-1106-preview"):
        """
        Executes a one-shot completion with the GPT-4 language model, using both a system and a user prompt.

        This method is designed for scenarios where a single interaction with the GPT-4 model is required, rather than a 
        continuous conversation. It allows specifying both a system-level and a user-level prompt to guide the model's response.

        Args:
            system_prompt (str): The system-level prompt that sets the context or instructions for the model.
            user_prompt (str): The user's input or question to the model.
            json_response (bool, optional): If True, forces the response to be in JSON format. Requires a defined response schema.
                                            Defaults to False.
            model (str, optional): The specific GPT-4 model version to be used for the completion. Defaults to "gpt-4-1106-preview".

        Returns:
            str: The content of the model's response message as a string.

        Raises:
            ValueError: If the combined token count of the response and prompts exceeds the 4096 token limit.

        Note:
            - The 'temperature' parameter influences the model's creativity and unpredictability.
            - The 'max_tokens' parameter sets a limit on the response size.
            - This method is suitable for tasks like generating content, answering questions, or other one-off tasks.
        
        Example:
            >>> response = gpt4_one_shot("Always respond in French.", "Tell a one scentence poem about a robot's adventure.")
            >>> print(response)
            "Un robot solitaire, vers les étoiles il vole, son aventure commence, un rêve qui se dévoile."
        """
        # Initialize a one-shot completion with the GPT-4 model
        response = self.client.chat.completions.create(
            model=model,  # Specifies the GPT-4 model version
            temperature=0.79,  # Sets the AI's creativity level
            max_tokens=4096,  # Limits the response token count
            response_format={"type": "json_object"} if json_response else None,  # Optional JSON response format
            messages=[
                {"role": "system", "content": system_prompt},  # System-level context or instruction
                {"role": "user", "content": user_prompt}  # User input or question
            ]
        )

        # Check if the total token usage exceeds the limit
        # DO NOT CHANGE THIS - This is a requirement for the challenge.
        if response.usage.total_tokens > 4096:
            raise ValueError("CHALLENGE CONTEXT WINDOW EXCEEDED: The context window now exceeds the 4096 token limit. Please try again with a shorter prompt.")

        return response.choices[0].message.content
