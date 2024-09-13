import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, types
from dotenv import find_dotenv, load_dotenv

from graph_2 import graph

load_dotenv(find_dotenv())

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")

# Инициализация бота
bot = Bot(token=TG_BOT_TOKEN)
dp = Dispatcher()


@dp.message()
async def handle_message(message: types.Message):
    user_message = message.text
    if not user_message:
        return
    bot_username = (await bot.get_me()).username

    # Проверяем, было ли упоминание бота в канале или сообщение отправлено напрямую
    if message.chat.type in ("group", "supergroup", "channel"):
        if not (f"@{bot_username}" in user_message):
            return  # Игнорируем сообщение, если бот не был упомянут

        # Удаляем упоминание из текста сообщения, чтобы его не обрабатывала модель
        user_message = user_message.replace(f"@{bot_username}", "").strip()

    if user_message.startswith("/start"):
        await message.answer("Я готов к работе")
        return
    if user_message.startswith("/"):
        return
    if user_message.strip() == "":
        return

    try:
        logging.warning(
            f"User request: {user_message}, from {message.from_user.id} {message.from_user.username}"
        )

        # Предполагаем, что llm.stream поддерживает стриминг ответа
        answer = await message.answer("Обрабатываю ваш запрос...")
        inputs = {"question": user_message}
        last_step = None
        for output in graph.stream(inputs):
            for key, value in output.items():
                if key == last_step:
                    continue
                await answer.edit_text(
                    f"Current step - {key} " + "..."
                )
                last_step = key
        await answer.delete()
        await message.answer(value["generation"], parse_mode='Markdown')
    except Exception as e:
        logging.error(f"Error processing user request: {e}", exc_info=True)
        await message.answer(
            f"Произошла ошибка {e} при обработке вашего запроса. Пожалуйста, попробуйте еще раз."
        )


async def main():
    # Регистрация обработчиков
    dp.message.register(handle_message)

    # Запуск бота
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())