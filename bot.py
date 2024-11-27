import asyncio
import os
from aiogram import Bot, Dispatcher, types
from graph_4 import graph
from dotenv import load_dotenv, find_dotenv
import logging

load_dotenv(find_dotenv())

bot = Bot(token=os.getenv("TG_BOT_TOKEN"))
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
    if user_message.startswith("/") or user_message.strip() == "":
        return

    try:
        logging.warning(
            "User request: %s, from %s %s",
            user_message,
            message.from_user.id,
            message.from_user.username,
        )

        answer = await message.answer("Обрабатываю ваш запрос...")
        inputs = {"question": user_message}
        last_step, value = None, None
        async for output in graph.astream(inputs):
            for key, value in output.items():
                if key == last_step:
                    continue
                await answer.edit_text(f"Текущий шаг - {key} " + "...")
                last_step = key
        await answer.delete()
        if value:
            await message.answer(value["generation"], parse_mode="Markdown")
    except Exception as e:
        logging.error("Error processing user request: %s", e, exc_info=True)
        await message.answer(f"Произошла ошибка {e}. Пожалуйста, попробуйте еще раз.")


async def main():
    dp.message.register(handle_message)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
