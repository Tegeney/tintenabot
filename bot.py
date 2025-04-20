import logging
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
    CallbackQueryHandler
)
import io
import os
from dotenv import load_dotenv

# Configure logging for Render
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables - works with Render's environment variables
load_dotenv()  # This will still work locally but won't override Render's env vars
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN',"7930480821:AAGW0Y16VD2mg1RVV284IlnxyTWg0iWdGqY")
PORT = int(os.getenv('PORT', 8443))
WEBHOOK_URL = os.getenv('WEBHOOK_URL')

# Conversation states
SELECT_GRADE_COLUMN = 1

# Gender detection and mapping
def detect_gender_column(df):
    """Detect gender column with support for multiple languages"""
    for col in df.columns:
        col_lower = str(col).lower()
        
        if any(keyword in col_lower for keyword in ['sex', 'gender', 'ወታዊ', 'ጾታ', 'jinsia']):
            return col
        
        unique_values = df[col].dropna().astype(str).str.upper().unique()
        
        male_indicators = any(v in ['M', 'MALE', 'ወ', 'ወንድ', 'MUME'] for v in unique_values)
        female_indicators = any(v in ['F', 'FEMALE', 'ሴ', 'ሴት', 'KIKE'] for v in unique_values)
        
        if male_indicators and female_indicators:
            return col
    
    return None

def clean_gender_values(series):
    """Clean and standardize gender values"""
    gender_map = {
        'M': 'MALE', 'F': 'FEMALE',
        'MALE': 'MALE', 'FEMALE': 'FEMALE',
        'BOY': 'MALE', 'GIRL': 'FEMALE',
        'ወ': 'MALE', 'ሴ': 'FEMALE',
        'ወንድ': 'MALE', 'ሴት': 'FEMALE',
        'MUME': 'MALE', 'KIKE': 'FEMALE'
    }
    
    return (series
            .astype(str)
            .str.upper()
            .str.strip()
            .map(gender_map)
            .dropna()

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '📊 Grade Distribution Bot\n\n'
        'Send me a spreadsheet with student grades to analyze gender performance.'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = """
📘 *Bot Help Guide*

1. Send an Excel/CSV file containing:
   - Gender column (M/F, Male/Female, ወ/ሴ)
   - Grade column (numeric values)

2. The bot will:
   - Auto-detect gender information
   - Let you select grade column
   - Provide detailed statistics

3. Supported formats:
   - Gender: M/F, Male/Female, ወ/ሴ
   - Files: .xlsx, .csv (with headers)
"""
    await update.message.reply_text(help_text, parse_mode='Markdown')

# File processing
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        user = update.message.from_user
        logger.info(f"File from {user.first_name} (ID: {user.id})")
        
        file = await context.bot.get_file(update.message.document.file_id)
        file_buffer = io.BytesIO()
        await file.download_to_memory(out=file_buffer)
        file_buffer.seek(0)

        try:
            df = pd.read_excel(file_buffer)
        except:
            file_buffer.seek(0)
            df = pd.read_csv(file_buffer)

        df.columns = [str(col).strip() for col in df.columns]
        logger.info(f"Columns: {list(df.columns)}")

        sex_col = detect_gender_column(df)
        if not sex_col:
            await update.message.reply_text(
                "❌ Couldn't detect gender column.\nFound columns:\n" +
                "\n".join([f"- {col}" for col in df.columns])
            )
            return ConversationHandler.END

        df[sex_col] = clean_gender_values(df[sex_col])
        df = df[df[sex_col].isin(['MALE', 'FEMALE'])]
        
        if df.empty:
            await update.message.reply_text("❌ No valid gender data found.")
            return ConversationHandler.END

        context.user_data['df'] = df
        context.user_data['sex_col'] = sex_col

        keyboard = [
            [InlineKeyboardButton(col, callback_data=str(i))]
            for i, col in enumerate(df.columns)
            if col != sex_col
        ]
        
        await update.message.reply_text(
            f"✅ Gender column: {sex_col}\n"
            "📝 Select grade column:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return SELECT_GRADE_COLUMN

    except Exception as e:
        logger.error(f"File error: {str(e)}")
        await update.message.reply_text(
            "❌ Error processing file. Please check format and try again."
        )
        return ConversationHandler.END

async def select_column_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    try:
        selection = int(query.data)
        df = context.user_data['df']
        sex_col = context.user_data['sex_col']
        grade_col = df.columns[selection]

        df[grade_col] = pd.to_numeric(df[grade_col], errors='coerce')
        df = df.dropna(subset=[grade_col])
        
        if df.empty:
            await query.edit_message_text("❌ No valid grades found.")
            return ConversationHandler.END

        if df[grade_col].max() > 100:
            df[grade_col] = df[grade_col] / 10

        result = generate_report(df, sex_col, grade_col)
        await query.edit_message_text(result, parse_mode='Markdown')
        return ConversationHandler.END

    except Exception as e:
        logger.error(f"Callback error: {str(e)}")
        await query.edit_message_text("❌ Error analyzing data.")
        return ConversationHandler.END

def generate_report(df, sex_col, grade_col):
    """Generate formatted statistics report"""
    bins = [-1, 0, 4, 7, 10]
    labels = ['0', '1-4', '5-7', '8-10']
    df['grade_range'] = pd.cut(df[grade_col], bins=bins, labels=labels)

    dist = pd.crosstab(df[sex_col], df['grade_range'], margins=True)
    dist = dist.reindex(columns=labels + ['All'], fill_value=0)

    stats = {
        'total': len(df),
        'male': len(df[df[sex_col] == 'MALE']),
        'female': len(df[df[sex_col] == 'FEMALE']),
        'avg': round(df[grade_col].mean(), 2),
        'pass_rate': round((df[grade_col] >= 5).mean() * 100, 1)
    }

    report = [
        "📊 *Grade Analysis Report*",
        f"🔹 *Columns:* Gender=`{sex_col}`, Grades=`{grade_col}`",
        "",
        "🔢 *Grade Distribution*",
        "```",
        "┌──────────┬──────┬──────┬──────┬──────┬──────┐",
        "│ Gender   │  0   │ 1-4  │ 5-7  │ 8-10 │ All  │",
        "├──────────┼──────┼──────┼──────┼──────┼──────┤",
    ]

    for gender in ['MALE', 'FEMALE', 'All']:
        row = f"│ {gender.ljust(8)} │"
        for cat in labels + ['All']:
            row += f" {str(dist.loc[gender, cat]).center(4)} │"
        report.append(row)
    
    report += [
        "└──────────┴──────┴──────┴──────┴──────┴──────┘",
        "```",
        "",
        "📌 *Statistics*",
        "```",
        f"Total Students: {stats['total']}",
        f"Male: {stats['male']}, Female: {stats['female']}",
        f"Average Grade: {stats['avg']}/10",
        f"Pass Rate: {stats['pass_rate']}%",
        "```"
    ]

    return "\n".join(report)

# Application setup
def setup_application() -> Application:
    """Set up the Telegram application with webhook or polling"""
    application = Application.builder().token(TOKEN).build()

    # Conversation handler
    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.Document.ALL, handle_document)],
        states={SELECT_GRADE_COLUMN: [CallbackQueryHandler(select_column_callback)]},
        fallbacks=[CommandHandler("cancel", lambda u, c: ConversationHandler.END)]
    )

    # Command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(conv_handler)

    return application

def main() -> None:
    """Main entry point for the bot"""
    if not TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set")

    application = setup_application()

    # Deployment configuration for Render
    if WEBHOOK_URL:
        logger.info("Running in webhook mode for Render")
        application.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=TOKEN,
            webhook_url=f"{WEBHOOK_URL}/{TOKEN}"
        )
    else:
        logger.info("Running in polling mode (local development)")
        application.run_polling()

if __name__ == '__main__':
    main()
