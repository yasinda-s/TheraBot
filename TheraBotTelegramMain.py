# import APT KEY from Constants.py and Telegram packages
import Constants as Keys
from telegram.ext import *
import TheraBotTelegramCode as Tb

updater = Updater(Keys.API_KEY, use_context=True)

# this function is called when the '/start' command is entered by the user
# it will greet the user by calling the user's name
def start_command(update, context):
    name = update.message.chat.first_name
    update.message.reply_text('Hi ' + name + '!')
    update.message.reply_text('I am TheraBot!')
    update.message.reply_text('Nice to meet you 😊')
    update.message.reply_text('You can always type quit to end the conversation!')

# this function is called when the '/help' command is entered by the user
# it will prompt message to guide the users what to do next
def help_command(update, context):
    update.message.reply_text('You can use /start to begin')
    update.message.reply_text("You can type 'quit' to end")

# this function receives user's text messages and convert the messages to lowercases
# it will pass the text messages to the responses function in TheraBotTelegramCode.py for further processing
# and reply to the users
def handle_message(update, context):
    user_id = update.message.chat.id
    text = str(update.message.text).lower()
    response = Tb.responses(text, user_id)
    update.message.reply_text(response)

# this function handles error
def error(update, context):
    print(f"Update{update} caused error {context.error}")

# this function makes the chatbot to start running on Telegram
def main():
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start_command))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(MessageHandler(Filters.text, handle_message))
    dp.add_error_handler(error)
    updater.start_polling()
    updater.idle()

main()
