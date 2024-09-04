from aiogram import Bot, Dispatcher, types
from aiogram.types import BotCommand
from langchain_openai import ChatOpenAI
import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from graph import graph

load_dotenv(find_dotenv())

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")

# Инициализация бота
bot = Bot(token=TG_BOT_TOKEN)
dp = Dispatcher()

# Настройка модели GPT-4 с помощью langchain_openai
llm = ChatOpenAI(model="gpt-4o")

emojis = {
    "web_search": "🌍🔍",
    "retrieve": "📚🔍",
    "grade_documents": "📚⬆️",
    "generate": "🧠",
    "transform_query": "🔄",
}

@dp.message()
async def handle_message(message: types.Message):
    user_message = message.text
    bot_username = (await bot.get_me()).username
    
    # Проверяем, было ли упоминание бота в канале или сообщение отправлено напрямую
    if message.chat.type in ('group', 'supergroup', 'channel'):
        # Проверяем упоминание
        if not (f'@{bot_username}' in user_message):
            return  # Игнорируем сообщение, если бот не был упомянут

        # Удаляем упоминание из текста сообщения, чтобы его не обрабатывала модель
        user_message = user_message.replace(f'@{bot_username}', '').strip()

    # Предполагаем, что llm.stream поддерживает стриминг ответа
    answer = await message.answer("Обрабатываю ваш запрос...")
    inputs = {"question": user_message}
    for output in graph.stream(inputs):
        for key, value in output.items():
            # print(f"Finished running: {key} {value[0:100]}:")
            await answer.edit_text(f"Current step - {key} " + emojis.get(key, "") + "...")
    await answer.edit_text(value["generation"])

async def main():
    # Регистрация обработчиков
    dp.message.register(handle_message)
    
    # Запуск бота
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
