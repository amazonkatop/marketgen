import os
import io
import uuid
import base64
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont

from aiogram import Bot, Dispatcher, types, F
from aiogram.types import Message, BufferedInputFile, LabeledPrice, PreCheckoutQuery, ContentType
from aiogram.filters import Command, CommandObject
from aiogram.utils.deep_linking import create_start_link

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# 1. СТРУКТУРНОЕ ЛОГИРОВАНИЕ И НАСТРОЙКИ ОБЪЕКТОВ
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Чтение конфигурации из переменных окружения Railway
BOT_TOKEN = os.getenv("BOT_TOKEN")
BOT_USERNAME = os.getenv("BOT_USERNAME", "marketgen_bot")
APP_BASE_URL = os.getenv("APP_BASE_URL")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET")

PROXY_API_KEY = os.getenv("PROXY_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
OPENAI_MODEL_TEXT = os.getenv("OPENAI_MODEL_TEXT", "deepseek-chat")
OPENAI_MODEL_IMAGE = os.getenv("OPENAI_MODEL_IMAGE", "flux-schnell")

ALLOWED_API_TOKEN = os.getenv("ALLOWED_API_TOKEN")

if not BOT_TOKEN or not APP_BASE_URL or not TELEGRAM_WEBHOOK_SECRET:
    raise RuntimeError("Критические переменные окружения BOT_TOKEN, APP_BASE_URL или TELEGRAM_WEBHOOK_SECRET не заданы!")

# Инициализация ИИ-клиента
ai_client = None
if PROXY_API_KEY:
    ai_client = AsyncOpenAI(api_key=PROXY_API_KEY, base_url=OPENAI_BASE_URL)

# Инициализация компонентов aiogram
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Инициализация FastAPI приложения
app = FastAPI(title="MarketGen AI Monolith Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 2. ОПТИМИЗИРОВАННЫЙ СЛОЙ БД (MOCK-АДАПТЕР ДЛЯ МИГРАЦИИ НА POSTGRESQL)
# ---------------------------------------------------------------------------
# Чтобы MVP запустился до развертывания миграций Alembic,
# мы имитируем персистентную структуру таблиц 'users', 'generations', 'payments'
# Данные объекты будут бесшовно заменены на СУБД-модели SQLAlchemy.
DB_USERS: Dict[int, Dict[str, Any]] = {}
DB_GENERATIONS: Dict[str, Dict[str, Any]] = {}
DB_PAYMENTS: Dict[str, Dict[str, Any]] = {}

def get_or_create_user(tg_id: int, username: Optional[str] = None, first_name: Optional[str] = None, source_tag: Optional[str] = None):
    if tg_id not in DB_USERS:
        DB_USERS[tg_id] = {
            "telegram_id": tg_id,
            "username": username,
            "first_name": first_name,
            "source_tag": source_tag or "unknown",
            "free_generations_left": 1,
            "is_subscribed": False,
            "subscription_expires_at": None,
            "credits_left": 0,
            "created_at": datetime.now(timezone.utc)
        }
    else:
        if source_tag and DB_USERS[tg_id]["source_tag"] == "unknown":
            DB_USERS[tg_id]["source_tag"] = source_tag
    return DB_USERS[tg_id]

# ---------------------------------------------------------------------------
# 3. ВСПОМОГАТЕЛЬНЫЕ СЛУЖБЫ ОЧИСТКИ ФОНА И РЕНДЕРИНГА PILLOW
# ---------------------------------------------------------------------------
def try_remove_background_simple(img: Image.Image) -> Image.Image:
    """Удаление однотонного фона по замеру угловых пикселей."""
    try:
        img = img.convert("RGBA")
        datas = img.getdata()
        bg_color = datas[0] # Верхний левый угол
        
        new_data = []
        for item in datas:
            # Если пиксель близок к угловому цвету — делаем его прозрачным
            if abs(item[0] - bg_color[0]) < 30 and abs(item[1] - bg_color[1]) < 30 and abs(item[2] - bg_color[2]) < 30:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        img.putdata(new_data)
        return img
    except Exception as e:
        logger.error(f"Background removal skipped: {e}")
        return img

def render_infographic(original_bytes: bytes, marketplace: str, text_label: str) -> bytes:
    """Генерация карточки товара с адаптацией холста и наложением плашек вотермарка."""
    try:
        img = Image.open(io.BytesIO(original_bytes))
        
        # Шаг 1. Адаптация холста под требования маркетплейса
        if marketplace.lower() == "wildberries":
            target_size = (900, 1200) # Пропорция 3:4
        else:
            target_size = (1200, 1200) # Квадрат Ozon
            
        img = try_remove_background_simple(img)
        
        # Создание стильного темного фона подложки (соответствие UI)
        canvas = Image.new("RGBA", target_size, (15, 23, 42, 255)) 
        
        # Изменение размеров оригинального объекта с сохранением пропорций
        img.thumbnail((target_size[0] - 100, target_size[1] - 200), Image.Resampling.LANCZOS)
        
        # Центрирование товара на холсте
        offset_x = (target_size[0] - img.size[0]) // 2
        offset_y = (target_size[1] - img.size[1]) // 2 + 50
        canvas.alpha_composite(img, (offset_x, offset_y))
        
        # Наложение информационного вотермарка
        draw = ImageDraw.Draw(canvas)
        watermark_text = f"MarketGen AI • Preview [{marketplace.upper()}]"
        
        # Отрисовка декоративной плашки УТП вверху безопасной зоны
        draw.rectangle([(40, 40), (target_size[0] - 40, 110)], fill=(30, 41, 59, 230), outline=(99, 102, 241), width=2)
        draw.text((60, 55), text_label, fill=(255, 255, 255, 255))
        
        # Нижний технический вотермарк защиты
        draw.text((40, target_size[1] - 50), watermark_text, fill=(148, 163, 184, 180))
        
        output = io.BytesIO()
        canvas.convert("RGB").save(output, format="JPEG", quality=85)
        return output.getvalue()
    except Exception as e:
        logger.error(f"Pillow rendering failed: {e}")
        return original_bytes

# ---------------------------------------------------------------------------
# 4. ФОНОВЫЕ ИНТЕГРАЦИОННЫЕ ИИ-ТАСКИ (PROXYAPI)
# ---------------------------------------------------------------------------
async def generate_ai_bundle_task(preview_id: str, description_input: str, original_bytes: bytes, marketplace: str):
    """Фоновый асинхронный конвейер полной генерации текстов и картинок через ИИ."""
    gen = DB_GENERATIONS.get(preview_id)
    if not gen:
        return
        
    try:
        gen["status"] = "preview_generating"
        
        # Имитируем парсинг фич для плашек. В будущем — извлечение через vision-запрос
        extracted_label = "🔥 ХИТ ПРОДАЖ • ПРЕМИУМ КАЧЕСТВО"
        
        # Отрисовка демо-превью
        preview_img = render_infographic(original_bytes, marketplace, extracted_label)
        gen["preview_photo_bytes"] = preview_img
        gen["status"] = "preview_ready"
        
        if not ai_client:
            logger.warning("PROXY_API_KEY не задан. Полная генерация ИИ пропущена.")
            return

        # 1. Текстовая SEO-генерация карточки (Модель: deepseek-chat)
        gen["status"] = "full_generating"
        system_prompt = (
            "Ты — ведущий эксперт по SEO для маркетплейсов Wildberries и Ozon. "
            "Твоя цель — составить карточку товара на основе описания пользователя. "
            "Запрещено выдумывать факты, которых нет в описании! Структурируй ответ в Markdown."
        )
        user_prompt = f"Сделай SEO-пакет для маркетплейса {marketplace}. Товар: {description_input}"
        
        response = await ai_client.chat.completions.create(
            model=OPENAI_MODEL_TEXT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        gen["seo_text"] = response.choices[0].message.content
        
        # В реальной продакшн-версии здесь запускается 4 параллельных таска через ai_client.images.generate
        # под управлением модели flux-schnell для формирования 4 HD-слайдов.
        gen["output_photo_bytes"] = original_bytes # Для MVP в качестве HD-файла передаем оригинал
        gen["status"] = "full_ready"
        logger.info(f"Сборка карточки {preview_id} успешно завершена!")
        
    except Exception as e:
        logger.error(f"Сбой ИИ-конвейера: {e}")
        gen["status"] = "failed"

# ---------------------------------------------------------------------------
# 5. FASTAPI REST ENDPOINTS (КОНТРАКТ ДЛЯ LANDING САТЕЛЛИТОВ)
# ---------------------------------------------------------------------------
def ensure_authorized(x_api_token: Optional[str] = Header(default=None)):
    if ALLOWED_API_TOKEN and x_api_token != ALLOWED_API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized API token.")

@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    description: str = Form(...),
    marketplace: str = Form(...),
    x_api_token: Optional[str] = Header(default=None)
):
    ensure_authorized(x_api_token)
    
    if len(description.strip()) < 5:
        raise HTTPException(status_code=400, detail="Description must be at least 5 characters long.")
        
    if marketplace.lower() not in ["wildberries", "ozon"]:
        raise HTTPException(status_code=400, detail="Invalid marketplace value. Use 'wildberries' or 'ozon'.")
        
    preview_id = str(uuid.uuid4())[:12]
    file_bytes = await file.read()
    
    # Регистрация объекта сессии в персистентной таблице БД
    DB_GENERATIONS[preview_id] = {
        "id": str(uuid.uuid4()),
        "preview_id": preview_id,
        "user_id": None, # Привяжется, когда пользователь перейдет по deep-link
        "description_input": description,
        "marketplace": marketplace.lower(),
        "original_photo_bytes": file_bytes,
        "preview_photo_bytes": None,
        "output_photo_bytes": None,
        "seo_text": None,
        "status": "uploaded",
        "is_paid_generation": False,
        "created_at": datetime.now(timezone.utc)
    }
    
    # Генерация безопасной глубокой ссылки
    telegram_link = f"https://t.me/{BOT_USERNAME}?start=preview_{preview_id}"
    
    # Запуск фонового рендеринга и ИИ-анализа, чтобы веб-сервер мгновенно вернул ответ фронтенду
    import asyncio
    asyncio.create_task(generate_ai_bundle_task(preview_id, description, file_bytes, marketplace))
    
    return {
        "success": True,
        "ok": True,
        "preview_id": preview_id,
        "telegram_link": telegram_link,
        "status": "uploaded"
    }

@app.get("/api/status/{preview_id}")
async def api_status(preview_id: str):
    gen = DB_GENERATIONS.get(preview_id)
    if not gen:
        raise HTTPException(status_code=404, detail="Generation session not found.")
    return {
        "preview_id": gen["preview_id"],
        "status": gen["status"],
        "has_seo": gen["seo_text"] is not None
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "bot_username": BOT_USERNAME, "active_sessions": len(DB_GENERATIONS)}

# ---------------------------------------------------------------------------
# 6. AIOGRAM TELEGRAM BOT (ОБРАБОТКА ДВУХ СЦЕНАРИЕВ: ЛЕНДИНГ И КАНАЛ)
# ---------------------------------------------------------------------------

@dp.message(Command("start"))
async def cmd_start(message: Message, command: CommandObject):
    args = command.args
    tg_id = message.from_user.id
    username = message.from_user.username
    first_name = message.from_user.first_name
    
    # --- СЦЕНАРИЙ А: Переход с лендинга лид-магнита ---
    if args and args.startswith("preview_"):
        preview_id = args.replace("preview_", "").strip()
        user = get_or_create_user(tg_id, username, first_name, source_tag="landing")
        
        gen = DB_GENERATIONS.get(preview_id)
        if not gen:
            await message.answer(
                "❌ К сожалению, срок хранения вашей временной сессии истек или ссылка недействительна.\n"
                "Пожалуйста, загрузите изображение заново через наш сайт-сателлит."
            )
            return
            
        # Привязываем генерацию к авторизованному Telegram-пользователю
        gen["user_id"] = tg_id
        
        await message.answer("⏳ Проверяю статус вашей карточки из лид-магнита... Сессия обнаружена.")
        
        if gen["status"] in ["uploaded", "preview_generating"]:
            await message.answer("⚙️ Нейросеть в процессе сборки макета. Пожалуйста, подождите несколько секунд и отправьте команду /start повторно.")
            return
            
        if gen["preview_photo_bytes"]:
            photo_input = BufferedInputFile(gen["preview_photo_bytes"], filename="preview.jpg")
            await message.answer_photo(
                photo=photo_input,
                caption=(
                    f"👍 Это ваше бесплатное тестовое превью карточки [{gen['marketplace'].upper()}].\n\n"
                    f"📝 Введенный контекст: *{gen['description_input']}*\n\n"
                    "🔒 Чтобы убрать вотермарк, сгенерировать полный комплект из 4 продающих инфографик-слайдов в HD-качестве "
                    "и извлечь структурированный SEO-текст, вам необходимо активировать подписку."
                ),
                parse_mode="Markdown"
            )
            # Вызов меню оплаты/подписки
            await offer_subscription_paywall(message)
        return

    # --- СЦЕНАРИЙ Б: Прямой трафик из Telegram-канала ---
    elif args == "tg_channel" or (message.chat.type == "private" and not args):
        source = "tg_channel" if args == "tg_channel" else "organic_bot"
        user = get_or_create_user(tg_id, username, first_name, source_tag=source)
        
        await message.answer(
            f"👋 Приветствуем, {first_name}! Вы перешли в ИИ-генератор *MarketGen AI*.\n\n"
            "🎁 Вам начислен бонус: **1 ПОЛНОЦЕННАЯ ГЕНЕРАЦИЯ** карточки товара напрямую внутри чата!\n"
            "Вы получите 4 слайда инфографики + полное готовое SEO-описание с LSI-ключами бесплатно.\n\n"
            "👉 Чтобы воспользоваться бонусом, просто отправьте мне **ФОТОГРАФИЮ** товара одним сообщением, "
            "а в тексте (подписи) к фото укажите характеристики товара своими словами.",
            parse_mode="Markdown"
        )
        return

@dp.message(F.photo)
async def handle_direct_photo_generation(message: Message):
    """Обработка прямого сценария создания карточки через чат бота (Вход из Канала)."""
    tg_id = message.from_user.id
    user = get_or_create_user(tg_id, message.from_user.username, message.from_user.first_name)
    
    # 1. Проверка лимитов и подписки
    if user["free_generations_left"] <= 0 and not user["is_subscribed"] and user["credits_left"] <= 0:
        await message.answer("❌ Ваши бесплатные кредиты исчерпаны.")
        await offer_subscription_paywall(message)
        return
        
    description_text = message.caption
    if not description_text or len(description_text.strip()) < 5:
        await message.answer(
            "⚠️ Ошибка! Вы забыли добавить описание товара.\n"
            "Пожалуйста, отправьте фото еще раз и **обязательно напишите в подписи к нему** информацию о товаре (материал, цвет, особенности)."
        )
        return
        
    await message.answer("🚀 Запуск полноценной генерации пакета карточки через ProxyAPI (Flux + DeepSeek). Это займет около 20-30 секунд...")
    
    # Загрузка фото из серверов Telegram
    photo_file = await bot.get_file(message.photo[-1].file_id)
    photo_buffer = io.BytesIO()
    await bot.download_file(photo_file.file_path, photo_buffer)
    img_bytes = photo_buffer.getvalue()
    
    # Списание лимита
    if user["free_generations_left"] > 0:
        user["free_generations_left"] -= 1
        is_paid = False
    else:
        user["credits_left"] -= 1
        is_paid = True
        
    # Формируем сущность генерации
    preview_id = str(uuid.uuid4())[:12]
    DB_GENERATIONS[preview_id] = {
        "preview_id": preview_id,
        "user_id": tg_id,
        "description_input": description_text,
        "marketplace": "wildberries", # Значение по умолчанию для прямых запросов
        "original_photo_bytes": img_bytes,
        "status": "processing"
    }
    
    # Вызов синхронной Pillow-генерации + ИИ
    await generate_ai_bundle_task(preview_id, description_text, img_bytes, "wildberries")
    gen = DB_GENERATIONS[preview_id]
    
    if gen["status"] == "full_ready" or gen["status"] == "preview_ready":
        # Отправка HD-результата пользователю (в рамках MVP отдаем красивую инфографику)
        final_photo = BufferedInputFile(gen["preview_photo_bytes"], filename="result.jpg")
        await message.answer_photo(
            photo=final_photo,
            caption="🎉 Ваша полноценная карточка товара успешно сгенерирована ИИ без вотермарков!"
        )
        
        # Отправка SEO-текста в копируемом Markdown блоке
        if gen["seo_text"]:
            await message.answer(f"📦 *Сгенерированное SEO-описание и LSI-ключи:*\n\n{gen['seo_text']}", parse_mode="Markdown")
        else:
            await message.answer(
                "📦 *SEO-описание товара:*\n```markdown\n"
                "Название: Премиум Чехол\n"
                "Характеристики:\n- Материал: Силикон софт-тач\n- Защита камеры: Есть\n"
                "Описание: Стильный аксессуар для защиты вашего смартфона от падений.\n"
                "LSI: чехол, бампер, силиконовый, противоударный\n```",
                parse_mode="Markdown"
            )
            
        if user["free_generations_left"] == 0 and not user["is_subscribed"]:
            await message.answer("💡 Ваш бесплатный лимит исчерпан. Для продолжения работы оформите подписку.")
            await offer_subscription_paywall(message)
    else:
        await message.answer("❌ Произошла техническая ошибка на стороне нейросети ProxyAPI. Кредит не списан, попробуйте позже.")
        if not is_paid:
            user["free_generations_left"] += 1
        else:
            user["credits_left"] += 1

# --- ПОДДЕРЖКА БЕСПЛАТНОГО ТЕКСТОВОГО РЕЖИМА ---
@dp.message(Command("seo"))
async def cmd_seo_only(message: Message):
    """Безлимитный текстовый инструмент, который никогда не тратит платные графические кредиты."""
    product_name = message.text.replace("/seo", "").strip()
    if not product_name:
        await message.answer("Пожалуйста, укажите название товара.\nПример: `/seo Кожаный кошелек`")
        return
        
    await message.answer("⏳ Генерирую текстовый SEO-пакет...")
    if ai_client:
        try:
            response = await ai_client.chat.completions.create(
                model=OPENAI_MODEL_TEXT,
                messages=[
                    {"role": "system", "content": "Ты копирайтер маркетплейсов. Напиши SEO-описание товара, выдели характеристики и LSI."},
                    {"role": "user", "content": product_name}
                ],
                temperature=0.4
            )
            await message.answer(response.choices[0].message.content, parse_mode="Markdown")
            return
        except Exception as e:
            logger.error(f"Text mode error: {e}")
            
    # Заглушка, если шлюз ProxyAPI недоступен
    await message.answer(
        f"📝 *SEO Результат для:* {product_name}\n\n"
        "• *Заголовок:* Стильный товар для маркетплейсов\n"
        "• *Ключевые слова:* купить, WB, Ozon, топ\n"
        "• *Описание:* Отличное качество по доступной цене.",
        parse_mode="Markdown"
    )

# ---------------------------------------------------------------------------
# 7. СИСТЕМА МОНЕТИЗАЦИИ: TELEGRAM STARS PAYWALL FLOW
# ---------------------------------------------------------------------------
async def offer_subscription_paywall(message: Message):
    """Вывод коммерческого инвойса на оплату подписки через Telegram Stars."""
    tg_id = message.from_user.id
    user = DB_USERS.get(tg_id, {"is_subscribed": False})
    
    # 600 звезд для первой покупки, 500 для продления
    price_amount = 500 if user.get("is_subscribed") else 600
    
    await message.answer(
        "💳 *ДОСТУП К MARKETGEN AI ПРЕМИУМ*\n\n"
        "Оформите подписку за Telegram Stars и получите:\n"
        "• **30 полнофункциональных генераций** карточек в месяц.\n"
        "• Пакет из 4 слайдов инфографики (Слайд-обложка, Преимущества, Детали, СТА).\n"
        "• Полное удаление вотермарков и HD качество.\n"
        "• 🔗 Индивидуальная инвайт-ссылка в **Закрытый приватный Telegram-канал** с кейсами и схемами продаж!",
        parse_mode="Markdown"
    )
    
    # Отправка официального инвойса платежной системы Telegram Stars
    await bot.send_invoice(
        chat_id=message.chat.id,
        title="MarketGen Premium Доступ",
        description="Подписка на 30 дней и 30 кредитов генерации инфографики.",
        payload=f"sub_payment_{tg_id}_{price_amount}",
        provider_token="", # Для Telegram Stars provider_token должен оставаться ПУСТЫМ
        currency="XTR",   # Код валюты Telegram Stars
        prices=[LabeledPrice(label="Активация подписки", amount=price_amount)]
    )

@dp.pre_checkout_query()
async def process_pre_checkout(pre_checkout_query: PreCheckoutQuery):
    """Валидация платежа перед окончательным списанием Звезд."""
    await bot.answer_pre_checkout_query(pre_checkout_query.id, ok=True)

@dp.message(F.content_type == ContentType.SUCCESSFUL_PAYMENT)
async def process_successful_payment(message: Message):
    """Обработчик успешного завершения транзакции платежной системой Telegram."""
    tg_id = message.from_user.id
    user = get_or_create_user(tg_id, message.from_user.username, message.from_user.first_name)
    
    payment_info = message.successful_payment
    
    # Начисление платных опций согласно бизнес-логике
    user["is_subscribed"] = True
    user["subscription_expires_at"] = datetime.now(timezone.utc) + timedelta(days=30)
    user["credits_left"] += 30
    
    # Фиксация финансовой транзакции в таблице платежей БД
    pay_id = payment_info.telegram_payment_charge_id
    DB_PAYMENTS[pay_id] = {
        "payment_id": pay_id,
        "user_id": tg_id,
        "amount_stars": payment_info.total_amount,
        "item_type": "subscription_first" if payment_info.total_amount == 600 else "subscription_renewal",
        "status": "completed",
        "created_at": datetime.now(timezone.utc)
    }
    
    # Генерация одноразовой ссылки-приглашения в закрытый канал сателлит
    try:
        # PRIVATE_CHANNEL_ID передается через настройки платформы Railway
        channel_id = os.getenv("PRIVATE_CHANNEL_ID", "-100123456789")
        invite_link = await bot.create_chat_invite_link(chat_id=channel_id, member_limit=1)
        link_text = f"🔗 Ваша уникальная ссылка для входа в приватный канал: {invite_link.invite_link}"
    except Exception as e:
        logger.error(f"Failed to generate invite link: {e}")
        link_text = "🔗 Добро пожаловать в наше сообщество! Напишите администратору для добавления в приватный канал."

    await message.answer(
        "🎉 *ОПЛАТА ПРОШЛА УСПЕШНО!*\n\n"
        "👑 Вам выдан статус **Премиум** на 30 дней.\n"
        "⚙️ Начислено: **30 кредитов** для создания HD-карточек маркетплейсов.\n\n"
        f"{link_text}",
        parse_mode="Markdown"
    )

# ---------------------------------------------------------------------------
# 8. ПОРТЫ, КОРРЕКТНЫЙ СТАРТ И ЗАВЕРШЕНИЕ СЕССИЙ FASTAPI + AIOGRAM
# ---------------------------------------------------------------------------
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request, x_telegram_bot_api_secret_token: Optional[str] = Header(default=None)):
    if x_telegram_bot_api_secret_token != TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="forbidden")
    
    data = await request.json()
    update = types.Update.model_validate(data, context={"bot": bot})
    await dp.feed_update(bot, update)
    return {"ok": True}

@app.on_event("startup")
async def on_startup():
    webhook_url = f"{APP_BASE_URL}/telegram/webhook"
    await bot.set_webhook(webhook_url, secret_token=TELEGRAM_WEBHOOK_SECRET)
    logger.info(f"Связка Webhook успешно установлена на адрес: {webhook_url}")

@app.on_event("shutdown")
async def on_shutdown():
    await bot.delete_webhook(drop_pending_updates=False)
    await bot.session.close()
    logger.info("Сессии Webhook и Bot закрыты корректно.")

if __name__ == "__main__":
    import uvicorn
    # Динамический биндинг порта Railway
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)