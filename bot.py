import logging
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
import io
import os
import json
import asyncio
from dotenv import load_dotenv
from flask import Flask, request
from datetime import datetime

# Load environment variables
load_dotenv()
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not set in .env")
PORT = int(os.environ.get('PORT', 5000))
WEBHOOK_PATH = '/webhook'

# Enable logging to console and file
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app for Render
app = Flask(__name__)
application = None  # Global application instance for webhook

@app.route('/')
def home():
    return "Telegram Grade Distribution Bot is running!"

@app.route('/health')
def health():
    return {"status": "running", "timestamp": datetime.now().isoformat()}

@app.route(WEBHOOK_PATH, methods=['POST'])
def webhook():
    """Handle incoming Telegram updates synchronously."""
    global application
    try:
        update = Update.de_json(json.loads(request.get_data(as_text=True)), application.bot)
        application.process_update(update)  # Synchronous processing
        return '', 200
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        return '', 500

# Conversation states
SELECT_GRADE_COLUMN = 1

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    await update.message.reply_text(
        '📊 Grade Distribution Bot\n\n'
        'Send me an Excel (.xlsx, .xls) or CSV file with student grades to analyze.\n'
        'I will detect column headers and analyze your data.'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    await update.message.reply_text(
        'Send an Excel/CSV file with student data. '
        'First row must have headers.\n'
        'I will detect sex/gender and grade columns.\n'
        'Max file size: 10MB.'
    )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process uploaded spreadsheet."""
    user_id = update.effective_user.id
    logger.info(f"Received file from user {user_id}: {update.message.document.file_name}")
    
    try:
        # Validate file type and size
        file_name = update.message.document.file_name.lower()
        if not file_name.endswith(('.xlsx', '.xls', '.csv')):
            await update.message.reply_text(
                "❌ Please send an Excel (.xlsx, .xls) or CSV (.csv) file."
            )
            return ConversationHandler.END
        
        file = await context.bot.get_file(update.message.document.file_id)
        if file.file_size > 10_000_000:  # 10MB limit
            await update.message.reply_text("❌ File too large. Max size: 10MB.")
            return ConversationHandler.END

        # Download file
        file_buffer = io.BytesIO()
        await file.download_to_memory(out=file_buffer)
        file_buffer.seek(0)

        # Read file with first row as headers
        try:
            df = pd.read_excel(file_buffer, header=0)
        except:
            file_buffer.seek(0)
            df = pd.read_csv(file_buffer, header=0)

        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]

        # Auto-detect sex column
        sex_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if 'sex' in col_lower or 'gender' in col_lower:
                sex_col = col
                break
        
        if not sex_col:
            for col in df.columns:
                unique_vals = df[col].dropna().astype(str).str.upper().unique()
                if any(val in ['M', 'F', 'MALE', 'FEMALE'] for val in unique_vals):
                    sex_col = col
                    break

        if not sex_col:
            columns = "\n".join([f"- {col}" for col in df.columns])
            await update.message.reply_text(
                "❌ Couldn't detect sex/gender column.\n"
                "I looked for:\n"
                "- Columns named 'sex' or 'gender'\n"
                "- Columns with M/F or Male/Female values\n\n"
                f"Found columns:\n{columns}"
            )
            return ConversationHandler.END

        # Clean sex data
        df[sex_col] = df[sex_col].astype(str).str.strip().str.upper()
        df[sex_col] = df[sex_col].replace({
            'M': 'MALE', 'F': 'FEMALE',
            'MALE': 'MALE', 'FEMALE': 'FEMALE'
        })
        df = df[df[sex_col].isin(['MALE', 'FEMALE'])]

        # Store data
        context.user_data['df'] = df
        context.user_data['sex_col'] = sex_col
        context.user_data['retries'] = 0  # Initialize retry counter

        # Show columns for grade selection
        columns = "\n".join([f"{i+1}. {col}" for i, col in enumerate(df.columns)])
        await update.message.reply_text(
            f"✅ Detected sex/gender column: `{sex_col}`\n\n"
            f"📋 Available columns:\n{columns}\n\n"
            "Reply with the NUMBER of the grade column to analyze:",
            parse_mode='Markdown'
        )
        return SELECT_GRADE_COLUMN

    except Exception as e:
        logger.error(f"Error processing file for user {user_id}: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Error processing your file.\n"
            "Please ensure:\n"
            "- First row has headers\n"
            "- Contains sex/gender data\n"
            "- File is not password-protected"
        )
        return ConversationHandler.END

async def select_grade_column(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle grade column selection."""
    user_id = update.effective_user.id
    try:
        # Enforce retry limit
        context.user_data['retries'] = context.user_data.get('retries', 0)
        if context.user_data['retries'] >= 3:
            await update.message.reply_text(
                "⚠️ Too many invalid inputs. Please start over with /start."
            )
            return ConversationHandler.END

        selection = int(update.message.text) - 1
        df = context.user_data['df'].copy()
        sex_col = context.user_data['sex_col']

        if selection < 0 or selection >= len(df.columns):
            context.user_data['retries'] += 1
            await update.message.reply_text(
                f"⚠️ Invalid column number. Try again (attempt {context.user_data['retries']}/3).",
                parse_mode='Markdown'
            )
            return SELECT_GRADE_COLUMN
        
        grade_col = df.columns[selection]
        context.user_data['retries'] = 0  # Reset retries on success

        # Validate grade data
        df[grade_col] = pd.to_numeric(df[grade_col], errors='coerce')
        if df[grade_col].isna().all():
            await update.message.reply_text(
                f"❌ Column `{grade_col}` contains no valid numeric grades."
            )
            return ConversationHandler.END
        df = df.dropna(subset=[grade_col])

        # Scale grades if needed (assume 0-100 to 0-10)
        if df[grade_col].max() > 10:
            df[grade_col] = df[grade_col] / 10
            logger.info(f"Scaled grades in {grade_col} for user {user_id}")

        # Calculate and send statistics
        result = calculate_statistics(df, sex_col, grade_col)
        await update.message.reply_text(result, parse_mode='Markdown')
        return ConversationHandler.END

    except ValueError:
        context.user_data['retries'] = context.user_data.get('retries', 0) + 1
        await update.message.reply_text(
            f"⚠️ Please enter a valid number (attempt {context.user_data['retries']}/3).",
            parse_mode='Markdown'
        )
        return SELECT_GRADE_COLUMN
    except Exception as e:
        logger.error(f"Grade selection error for user {user_id}: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Error processing your selection. Try again or send a new file."
        )
        return ConversationHandler.END

def calculate_statistics(df, sex_col, grade_col):
    """Calculate grade distribution statistics."""
    try:
        # Define grade ranges
        bins = [-1, 0, 4, 7, 10]
        labels = ['0', '1-4', '5-7', '8-10']
        df['grade_range'] = pd.cut(df[grade_col], bins=bins, labels=labels, right=True)

        # Create cross-tabulation
        grade_dist = pd.crosstab(
            index=df[sex_col],
            columns=df['grade_range'],
            margins=True,
            margins_name="TOTAL"
        ).reindex(columns=labels + ['TOTAL'], fill_value=0)

        # Build result message
        result = f"📊 *Analysis Report*\n\n"
        result += f"🔹 *Using Columns:*\n"
        result += f"- Sex/Gender: `{sex_col}`\n"
        result += f"- Grades: `{grade_col}`\n\n"
        
        # Counts table
        result += "🔢 *Grade Distribution Counts*\n"
        result += "```\n"
        result += "┌──────────┬──────┬──────┬──────┬──────┬───────┐\n"
        result += "│ Gender   │  0   │ 1-4  │ 5-7  │ 8-10 │ TOTAL │\n"
        result += "├──────────┼──────┼──────┼──────┼──────┼───────┤\n"
        
        for sex in grade_dist.index:
            sex_display = "TOTAL" if sex == "TOTAL" else sex.capitalize()
            row = f"│ {sex_display.ljust(8)} │"
            for grade in labels + ['TOTAL']:
                count = grade_dist.loc[sex, grade]
                row += f" {str(count).center(4)} │"
            result += row + "\n"
            if sex != grade_dist.index[-1]:
                result += "├──────────┼──────┼──────┼──────┼──────┼───────┤\n"
        
        result += "└──────────┴──────┴──────┴──────┴──────┴───────┘\n"
        result += "```\n\n"
        
        # Summary statistics
        total_students = len(df)
        male_count = len(df[df[sex_col] == 'MALE'])
        female_count = len(df[df[sex_col] == 'FEMALE'])
        avg_grade = round(float(df[grade_col].mean()), 2)
        pass_rate = round(float(len(df[df[grade_col] >= 5]) / total_students * 100), 1)
        
        result += "📌 *Key Statistics*\n"
        result += "```\n"
        result += f"Total Students: {total_students}\n"
        result += f"Male Students:  {male_count}\n"
        result += f"Female Students: {female_count}\n"
        result += f"Average Grade:  {avg_grade}/10\n"
        result += f"Pass Rate (≥5): {pass_rate}%\n"
        result += "```"
        
        return result

    except Exception as e:
        logger.error(f"Statistics error: {e}", exc_info=True)
        return f"❌ Error analyzing `{grade_col}`. Please check for valid numeric grades."

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel the conversation."""
    await update.message.reply_text("Operation cancelled. Send /start to begin again.")
    return ConversationHandler.END

async def setup_application():
    """Initialize and configure the Telegram bot."""
    global application
    application = Application.builder().token(TOKEN).build()

    # Conversation handler
    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.Document.ALL, handle_document)],
        states={
            SELECT_GRADE_COLUMN: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_grade_column)],
        },
        fallbacks=[
            CommandHandler('cancel', cancel),
            CommandHandler('help', help_command)
        ],
    )

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(conv_handler)

    # Set webhook
    webhook_url = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}{WEBHOOK_PATH}"
    try:
        await application.bot.set_webhook(url=webhook_url)
        logger.info(f"Webhook set to {webhook_url}")
    except Exception as e:
        logger.error(f"Failed to set webhook: {e}")
        raise

def main():
    """Start the bot with Flask."""
    try:
        # Run bot setup in async context
        loop = asyncio.get_event_loop()
        loop.run_until_complete(setup_application())
        
        # Start Flask server
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=PORT, use_reloader=False)
        
    except Exception as e:
        logger.error(f"Main error: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    load_dotenv()
    main()
