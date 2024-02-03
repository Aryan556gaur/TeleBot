from dotenv import load_dotenv
import os
from aiogram import Bot, Dispatcher, executor, types
import openai
import sys
import transformers
import torch
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

class Reference:
    '''
    A class to store previously response from the chatGPT API
    '''

    def __init__(self) -> None:
        self.response = ""


load_dotenv()
# openai.api_key = os.getenv("OpenAI_API_KEY")  

reference = Reference()

TOKEN = os.getenv("TOKEN")

#model name
MODEL_NAME = "karakuri-ai/karakuri-lm-70b-chat-v0.1"


# Initialize bot and dispatcher
bot = Bot(token=TOKEN)
dispatcher = Dispatcher(bot)


def clear_past():
    """A function to clear the previous conversation and context.
    """
    reference.response = ""



@dispatcher.message_handler(commands=['start'])
async def welcome(message: types.Message):
    """
    This handler receives messages with `/start` or  `/help `command
    """
    await message.reply("Hi\nI am Tele Bot!\Created by Aryan. How can i assist you?")



@dispatcher.message_handler(commands=['clear'])
async def clear(message: types.Message):
    """
    A handler to clear the previous conversation and context.
    """
    clear_past()
    await message.reply("I've cleared the past conversation and context.")



@dispatcher.message_handler(commands=['help'])
async def helper(message: types.Message):
    """
    A handler to display the help menu.
    """
    help_command = """
    Hi There, I'm chatGPT Telegram bot created by Aryan! Please follow these commands - 
    /start - to start the conversation
    /clear - to clear the past conversation and context.
    /help - to get this help menu.
    I hope this helps. :)
    """
    await message.reply(help_command)


# @dispatcher.message_handler()
# async def echo(message: types.Message):
#     """
#     This will retrun echo
#     """
#     if message.text.lower() in ["hi", "hello", "how are you", "whats up"]:
#         reply_text = "Hi! Aryan here... I'm doing well. How about you?"
#         await message.answer(reply_text)
#     else:
#         await message.answer(message.text)


@dispatcher.message_handler()
async def chat_blenderbot(message: types.Message):

    if message.text.lower() in ["hi", "hello", "how are you", "whats up"]:
        reply_text = "Hi! Aryan here... I'm doing well. How about you?"
        await message.answer(reply_text)
    else:
        """
        A handler to process the user's input and generate a response using BlenderBot model.
        """
        print(f">>> USER: \n\t{message.text}")
        
        # Tokenize the user's input
        inputs = tokenizer([message.text], return_tensors="pt", max_length=512, truncation=True)

        # Generate response from the model
        response_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=150,  # adjust as needed
            num_return_sequences=1,
            early_stopping=True
        )

        # Decode the response and send it
        reference.response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        print(f">>> BlenderBot: \n\t{reference.response}")
        await bot.send_message(chat_id=message.chat.id, text=reference.response)


if __name__ == "__main__":
    executor.start_polling(dispatcher, skip_updates=False)