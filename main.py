import os
import io
import uuid
import base64
from dataclasses import dataclass
from typing import Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, BufferedInputFile, Update
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
PROXY_API_KEY = os.getenv("PROXY_API_KEY")
BOT_USERNAME = os.getenv("BOT_USERNAME")
APP_BASE_URL = os.getenv("APP_BASE_URL")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET")



# =========================
# НАСТРОЙКИ И ПАМЯТЬ
# =========================
app = FastAPI()
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

DATA: Dict[str, dict] = {}


@dataclass
class Generation:
    preview_id: str
    input_bytes: bytes
    preview_bytes: Optional[bytes] = None
    status: str = "uploaded"


def make_preview(image_bytes: bytes) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    image.thumbnail((1200, 1200))

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    text = "MarketGen AI • Preview"
    font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = max(20, (image.size[0] - text_w) // 2)
    y = image.size[1] - text_h - 30

    draw.rounded_rectangle(
        (x - 18, y - 12, x + text_w + 18, y + text_h + 12),
        radius=18,
        fill=(0, 0, 0, 130),
    )
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)

    merged = Image.alpha_composite(image, overlay)
    buf = io.BytesIO()
    merged.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@dp.message(CommandStart())
async def cmd_start(message: Message):
    text = (
        "Привет! Это тестовый бот MarketGen AI.\n\n"
        "Команды:\n"
        "/start — старт\n"
        "/seo — бесплатное SEO-описание\n\n"
        "Можете отправить фото — я сделаю preview."
    )
    await message.answer(text)


@dp.message(Command("seo"))
async def cmd_seo(message: Message):
    await message.answer(
        "SEO-описание бесплатно:\n\n"
        "Название: Товар для маркетплейса\n\n"
        "Характеристики:\n"
        "- Качественный товар\n"
        "- Удобен в использовании\n"
        "- Подходит для ежедневного применения\n\n"
        "Описание: Это тестовый SEO-блок для проверки работы бота.\n\n"
        "LSI: товар, маркетплейс, описание, характеристики"
    )


@dp.message(F.photo)
async def handle_photo(message: Message):
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    content = await bot.download_file(file.file_path)

    image_bytes = content.read()
    preview_id = uuid.uuid4().hex[:12]

    DATA[preview_id] = {
        "status": "uploaded",
        "input_bytes": image_bytes,
        "preview_bytes": None,
    }

    preview_bytes = make_preview(image_bytes)
    DATA[preview_id]["preview_bytes"] = preview_bytes
    DATA[preview_id]["status"] = "preview_ready"

    file_obj = BufferedInputFile(preview_bytes, filename=f"{preview_id}_preview.jpg")
    await message.answer_photo(
        photo=file_obj,
        caption=f"Готово. Preview ID: {preview_id}"
    )


@app.get("/")
async def root():
    return HTMLResponse(
        """
        <html>
        <head><title>MarketGen AI</title></head>
        <body style="font-family: Arial; padding: 40px;">
            <h1>MarketGen AI bot is running</h1>
            <p>Use Telegram bot to test it.</p>
        </body>
        </html>
        """
    )


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    image_bytes = await file.read()
    preview_id = uuid.uuid4().hex[:12]

    DATA[preview_id] = {
        "status": "uploaded",
        "input_bytes": image_bytes,
        "preview_bytes": None,
    }

    return {
        "success": True,
        "preview_id": preview_id,
        "telegram_link": f"https://t.me/{BOT_USERNAME}?start=preview_{preview_id}",
    }


@app.post("/api/generate-preview/{preview_id}")
async def api_generate_preview(preview_id: str):
    item = DATA.get(preview_id)
    if not item:
        return JSONResponse({"error": "not found"}, status_code=404)

    preview_bytes = make_preview(item["input_bytes"])
    item["preview_bytes"] = preview_bytes
    item["status"] = "preview_ready"
    return {"success": True, "status": item["status"]}


@app.get("/api/status/{preview_id}")
async def api_status(preview_id: str):
    item = DATA.get(preview_id)
    if not item:
        return JSONResponse({"error": "not found"}, status_code=404)

    return {
        "preview_id": preview_id,
        "status": item["status"],
        "has_preview": item["preview_bytes"] is not None,
    }


@app.post("/telegram/webhook")
async def telegram_webhook(update: dict):
    telegram_update = Update.model_validate(update)
    await dp.feed_update(bot, telegram_update)
    return {"ok": True}


def setup_webhook():
    webhook_url = f"{APP_BASE_URL}/telegram/webhook"
    return bot.set_webhook(webhook_url, secret_token=TELEGRAM_WEBHOOK_SECRET)


# @app.on_event("startup")
# async def on_startup():
#    await setup_webhook()


@app.on_event("shutdown")
async def on_shutdown():
    await bot.session.close()