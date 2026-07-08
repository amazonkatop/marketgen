import os
import io
import re
import uuid
import json
import base64
import asyncio
import textwrap
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Form
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont

from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart, CommandObject, Command
from aiogram.types import Message, BufferedInputFile, Update
from aiogram.utils.deep_linking import create_start_link

from openai import AsyncOpenAI

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("marketgen-bot")

BOT_TOKEN = os.getenv("BOT_TOKEN")
PROXY_API_KEY = os.getenv("PROXY_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
BOT_USERNAME = os.getenv("BOT_USERNAME")
APP_BASE_URL = os.getenv("APP_BASE_URL")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
ALLOWED_API_TOKEN = os.getenv("ALLOWED_API_TOKEN", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN не найден в переменных окружения")

if not APP_BASE_URL:
    raise RuntimeError("APP_BASE_URL не найден в переменных окружения")

app = FastAPI(title="MarketGen AI")
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

ai_client: Optional[AsyncOpenAI] = None
if PROXY_API_KEY:
    ai_client = AsyncOpenAI(
        api_key=PROXY_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

DATA: Dict[str, "Generation"] = {}


@dataclass
class Generation:
    preview_id: str
    marketplace: str
    description: str
    input_bytes: bytes
    filename: str
    mime_type: str
    preview_bytes: Optional[bytes] = None
    final_image_bytes: Optional[bytes] = None
    seo_text: Optional[str] = None
    facts_json: Optional[dict] = None
    infographic_labels: Optional[List[str]] = None
    status: str = "uploaded"
    error: Optional[str] = None


def normalize_marketplace(value: str) -> str:
    value = (value or "").strip().lower()
    aliases = {
        "wb": "wildberries",
        "wildberries": "wildberries",
        "ozon": "ozon",
    }
    return aliases.get(value, "")


def ensure_authorized(x_api_token: Optional[str]):
    if ALLOWED_API_TOKEN and x_api_token != ALLOWED_API_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")


def resize_and_center(image: Image.Image, target: tuple[int, int]) -> Image.Image:
    image = image.convert("RGBA")
    canvas = Image.new("RGBA", target, (248, 248, 248, 255))

    ratio = min(target[0] / image.width, target[1] / image.height)
    new_size = (
        max(1, int(image.width * ratio)),
        max(1, int(image.height * ratio)),
    )
    resized = image.resize(new_size)

    x = (target[0] - resized.width) // 2
    y = (target[1] - resized.height) // 2
    canvas.alpha_composite(resized, (x, y))
    return canvas


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    words = text.split()
    lines = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines[:3]


def try_remove_background_simple(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    bg = Image.new("RGBA", image.size, (255, 255, 255, 0))

    corners = [
        image.getpixel((0, 0)),
        image.getpixel((image.width - 1, 0)),
        image.getpixel((0, image.height - 1)),
        image.getpixel((image.width - 1, image.height - 1)),
    ]
    avg = tuple(sum(p[i] for p in corners) // 4 for i in range(4))

    result = Image.new("RGBA", image.size)
    for y in range(image.height):
        for x in range(image.width):
            px = image.getpixel((x, y))
            diff = abs(px[0] - avg[0]) + abs(px[1] - avg[1]) + abs(px[2] - avg[2])
            if diff < 45:
                result.putpixel((x, y), (255, 255, 255, 0))
            else:
                result.putpixel((x, y), px)

    bg.alpha_composite(result)
    return bg


def marketplace_canvas(marketplace: str) -> tuple[int, int]:
    if marketplace == "wildberries":
        return (900, 1200)
    return (1200, 1200)


def make_preview(image_bytes: bytes, marketplace: str = "generic") -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    target = marketplace_canvas(marketplace if marketplace in {"wildberries", "ozon"} else "ozon")
    canvas = resize_and_center(image, target)

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    badge = f"MarketGen AI • {marketplace.title()} Preview"
    bbox = draw.textbbox((0, 0), badge, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    bx = max(20, (canvas.size[0] - text_w) // 2)
    by = canvas.size[1] - text_h - 40

    draw.rounded_rectangle(
        (bx - 18, by - 12, bx + text_w + 18, by + text_h + 12),
        radius=18,
        fill=(0, 0, 0, 145),
    )
    draw.text((bx, by), badge, fill=(255, 255, 255, 255), font=font)

    merged = Image.alpha_composite(canvas, overlay)
    buf = io.BytesIO()
    merged.convert("RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def render_infographic(
    image_bytes: bytes,
    marketplace: str,
    labels: List[str],
    title: str | None = None,
) -> bytes:
    target = marketplace_canvas(marketplace)
    base = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    product = try_remove_background_simple(base)
    product = resize_and_center(product, (int(target[0] * 0.72), int(target[1] * 0.72)))

    canvas = Image.new("RGBA", target, (250, 250, 248, 255))
    shadow = Image.new("RGBA", target, (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)

    cx = (target[0] - product.width) // 2
    cy = int(target[1] * 0.18)

    sdraw.ellipse(
        (
            cx + 80,
            cy + product.height - 40,
            cx + product.width - 80,
            cy + product.height + 35,
        ),
        fill=(0, 0, 0, 40),
    )
    canvas = Image.alpha_composite(canvas, shadow)
    canvas.alpha_composite(product, (cx, cy))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    title = (title or "").strip()[:64]

    if title:
        lines = wrap_text(draw, title, font, target[0] - 120)
        y = 36
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = (target[0] - tw) // 2
            draw.rounded_rectangle(
                (tx - 12, y - 8, tx + tw + 12, y + th + 8),
                radius=12,
                fill=(255, 255, 255, 220),
                outline=(225, 225, 225, 255),
            )
            draw.text((tx, y), line, fill=(20, 20, 20, 255), font=font)
            y += th + 18

    cleaned = [re.sub(r"\s+", " ", (x or "").strip())[:28] for x in labels if (x or "").strip()]
    cleaned = cleaned[:5]

    positions = [
        (34, 180),
        (target[0] - 250, 220),
        (34, target[1] - 240),
        (target[0] - 250, target[1] - 280),
        ((target[0] - 220) // 2, target[1] - 160),
    ]

    for i, text in enumerate(cleaned):
        x, y = positions[i]
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.rounded_rectangle(
            (x, y, x + tw + 32, y + th + 22),
            radius=16,
            fill=(17, 17, 17, 235),
        )
        draw.text((x + 16, y + 11), text, fill=(255, 255, 255, 255), font=font)

    out = io.BytesIO()
    canvas.convert("RGB").save(out, format="JPEG", quality=93)
    return out.getvalue()


def image_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


async def extract_facts_and_seo(item: Generation):
    if not ai_client:
        item.status = "failed"
        item.error = "AI client is not configured"
        return

    item.status = "analyzing"

    try:
        image_data_url = image_to_data_url(item.input_bytes, item.mime_type)

        system_prompt = (
            "Ты помощник селлера маркетплейсов Wildberries и Ozon. "
            "Пиши строго по-русски. "
            "Нельзя галлюцинировать. "
            "Используй только факты из описания пользователя и визуально очевидные признаки на фото. "
            "Если факт не подтвержден текстом пользователя или явно не виден на фото, не выдумывай его. "
            "Если характеристика не подтверждена, помечай ее как 'не уточнено' во внутреннем анализе, "
            "но не добавляй ее в итоговый продающий текст как факт."
        )

        user_text = f"""
Проанализируй фото товара и описание пользователя.

Маркетплейс: {item.marketplace}

Описание пользователя:
{item.description}

Верни результат СТРОГО в JSON формате по этой схеме:
{{
  "confirmed_facts": [
    "..."
  ],
  "uncertain_facts": [
    "..."
  ],
  "infographic_labels": [
    "короткая плашка 1",
    "короткая плашка 2",
    "короткая плашка 3"
  ],
  "product_title": "заголовок товара",
  "characteristics": [
    "характеристика 1",
    "характеристика 2"
  ],
  "sales_description": "продающее описание",
  "seo_keywords": [
    "ключ 1",
    "ключ 2"
  ],
  "lsi_phrases": [
    "lsi 1",
    "lsi 2"
  ]
}}

Ограничения:
- infographic_labels: от 3 до 5 коротких плашек, максимум 4 слова в каждой.
- product_title: без выдуманного бренда, без неподтвержденной совместимости.
- characteristics: только подтвержденные факты.
- sales_description: 600-1000 символов, без фантазий.
- seo_keywords и lsi_phrases: только релевантные подтвержденному товару.
- Если описание скудное, лучше сделать более сдержанный текст, чем выдумать детали.
- Верни только JSON, без markdown и без пояснений.
""".strip()

        response = await ai_client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                    ],
                },
            ],
        )

        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            item.status = "failed"
            item.error = "AI returned empty JSON"
            return

        parsed = json.loads(raw)
        item.facts_json = parsed

        labels = parsed.get("infographic_labels") or []
        title = (parsed.get("product_title") or "").strip()

        characteristics = parsed.get("characteristics") or []
        sales_description = (parsed.get("sales_description") or "").strip()
        seo_keywords = parsed.get("seo_keywords") or []
        lsi_phrases = parsed.get("lsi_phrases") or []
        confirmed_facts = parsed.get("confirmed_facts") or []
        uncertain_facts = parsed.get("uncertain_facts") or []

        seo_text = []
        seo_text.append("1. Название товара")
        seo_text.append(title or "Не удалось сформировать название без риска искажения фактов.")
        seo_text.append("")
        seo_text.append("2. Подтвержденные факты")
        seo_text.extend(f"- {x}" for x in confirmed_facts[:15] if str(x).strip())
        seo_text.append("")
        seo_text.append("3. Характеристики")
        seo_text.extend(f"- {x}" for x in characteristics[:15] if str(x).strip())
        seo_text.append("")
        seo_text.append("4. Продающее описание")
        seo_text.append(sales_description or "Описание не сформировано.")
        seo_text.append("")
        seo_text.append("5. SEO-ключи")
        seo_text.append(", ".join([str(x).strip() for x in seo_keywords if str(x).strip()]) or "Нет данных")
        seo_text.append("")
        seo_text.append("6. LSI-фразы")
        seo_text.append(", ".join([str(x).strip() for x in lsi_phrases if str(x).strip()]) or "Нет данных")

        if uncertain_facts:
            seo_text.append("")
            seo_text.append("7. Что не было подтверждено")
            seo_text.extend(f"- {x}" for x in uncertain_facts[:10] if str(x).strip())

        item.seo_text = "\n".join(seo_text).strip()
        item.infographic_labels = labels[:5]
        item.final_image_bytes = render_infographic(
            image_bytes=item.input_bytes,
            marketplace=item.marketplace,
            labels=item.infographic_labels or ["Фото товара", "На основе фактов", "Под формат площадки"],
            title=title,
        )
        item.status = "ready"

    except Exception as e:
        logger.exception("AI generation error")
        item.status = "failed"
        item.error = str(e)


@app.on_event("startup")
async def on_startup():
    webhook_url = f"{APP_BASE_URL.rstrip('/')}/telegram/webhook"
    await bot.set_webhook(
        webhook_url,
        secret_token=TELEGRAM_WEBHOOK_SECRET or None,
    )
    logger.info("Webhook set to %s", webhook_url)


@app.on_event("shutdown")
async def on_shutdown():
    await bot.delete_webhook(drop_pending_updates=False)
    await bot.session.close()


@app.get("/health")
async def health():
    return {
        "ok": True,
        "bot_username": BOT_USERNAME,
        "openai_configured": bool(PROXY_API_KEY),
        "tasks_in_memory": len(DATA),
    }


@app.post("/telegram/webhook")
async def telegram_webhook(
    update: dict,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
):
    if TELEGRAM_WEBHOOK_SECRET and x_telegram_bot_api_secret_token != TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="forbidden")

    telegram_update = Update.model_validate(update)
    await dp.feed_update(bot, telegram_update)
    return {"ok": True}


@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    description: str = Form(...),
    marketplace: str = Form(...),
    x_api_token: str | None = Header(default=None),
):
    ensure_authorized(x_api_token)

    if not BOT_USERNAME:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "BOT_USERNAME is not configured"},
        )

    content = await file.read()
    description = (description or "").strip()
    marketplace = normalize_marketplace(marketplace)

    if not content:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "empty file"},
        )

    if len(description) < 5:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "description is too short"},
        )

    if marketplace not in {"wildberries", "ozon"}:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "unsupported marketplace"},
        )

    content_type = (file.content_type or "").lower()
    if content_type not in {"image/jpeg", "image/jpg", "image/png", "image/webp"}:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "unsupported file type"},
        )

    preview_id = uuid.uuid4().hex[:12]
    preview_bytes = make_preview(content, marketplace=marketplace)

    item = Generation(
        preview_id=preview_id,
        marketplace=marketplace,
        description=description,
        input_bytes=content,
        filename=file.filename or f"{preview_id}.jpg",
        mime_type=content_type if content_type != "image/jpg" else "image/jpeg",
        preview_bytes=preview_bytes,
        status="uploaded",
    )
    DATA[preview_id] = item

    deep_link = await create_start_link(bot, f"preview_{preview_id}", encode=False)
    asyncio.create_task(extract_facts_and_seo(item))

    return {
        "ok": True,
        "preview_id": preview_id,
        "telegram_link": deep_link,
        "status": item.status,
    }


@app.get("/api/status/{preview_id}")
async def api_status(preview_id: str, x_api_token: str | None = Header(default=None)):
    ensure_authorized(x_api_token)

    item = DATA.get(preview_id)
    if not item:
        return JSONResponse(status_code=404, content={"ok": False, "error": "not found"})

    return {
        "ok": True,
        "preview_id": item.preview_id,
        "status": item.status,
        "marketplace": item.marketplace,
        "has_preview": bool(item.preview_bytes),
        "has_final": bool(item.final_image_bytes),
        "has_seo": bool(item.seo_text),
        "error": item.error,
    }


@dp.message(CommandStart(deep_link=True))
async def cmd_start_with_payload(message: Message, command: CommandObject):
    args = (command.args or "").strip()

    if not args.startswith("preview_"):
        await message.answer("Некорректный параметр запуска.")
        return

    preview_id = args.replace("preview_", "", 1)
    item = DATA.get(preview_id)

    if not item:
        await message.answer(
            "Сессия не найдена. Возможно, сервис был перезапущен, и временные данные очищены."
        )
        return

    if item.preview_bytes:
        preview_photo = BufferedInputFile(item.preview_bytes, filename=f"{preview_id}_preview.jpg")
        await message.answer_photo(
            photo=preview_photo,
            caption=(
                f"Задача получена.\n"
                f"Маркетплейс: {item.marketplace.title()}\n"
                f"Статус: {item.status}"
            ),
        )

    if item.status in {"uploaded", "analyzing"}:
        await message.answer(
            "Генерация еще выполняется. Откройте бота чуть позже."
        )
        return

    if item.status == "failed":
        await message.answer(
            f"Генерация завершилась ошибкой: {item.error or 'неизвестная ошибка'}"
        )
        return

    if item.final_image_bytes:
        final_photo = BufferedInputFile(item.final_image_bytes, filename=f"{preview_id}_final.jpg")
        await message.answer_photo(
            photo=final_photo,
            caption="Готово! Вот карточка-превью товара.",
        )

    if item.seo_text:
        chunks = textwrap.wrap(
            item.seo_text,
            width=3500,
            break_long_words=False,
            break_on_hyphens=False,
        )
        for chunk in chunks:
            await message.answer(chunk)


@dp.message(CommandStart())
async def cmd_start_plain(message: Message):
    text = (
        "Привет! Это бот MarketGen AI.\n\n"
        "Как это работает:\n"
        "1. На сайте загрузите фото товара\n"
        "2. Добавьте описание своими словами\n"
        "3. Выберите Wildberries или Ozon\n"
        "4. Получите результат здесь, в Telegram\n\n"
        "Фото в бот тоже можно отправить — я сделаю тестовое preview."
    )
    await message.answer(text)


@dp.message(Command("seo"))
async def cmd_seo(message: Message):
    await message.answer(
        "Команда /seo больше не является основным сценарием.\n"
        "Теперь генерация работает корректно только в связке: фото + описание + marketplace через сайт."
    )


@dp.message(lambda message: bool(message.photo))
async def handle_photo(message: Message):
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    file_bytes_io = io.BytesIO()
    await bot.download_file(file.file_path, destination=file_bytes_io)

    preview_bytes = make_preview(file_bytes_io.getvalue(), marketplace="ozon")
    result = BufferedInputFile(preview_bytes, filename="preview.jpg")

    await message.answer_photo(
        photo=result,
        caption=(
            "Готово! Вот тестовое preview.\n"
            "Для полноценной карточки и SEO-описания используйте загрузку через сайт."
        ),
    )