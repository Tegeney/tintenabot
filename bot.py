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
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', "YOUR_TELEGRAM_BOT_TOKEN") # Replace with your actual token or ensure it's in .env
PORT = int(os.getenv('PORT', 8443))
WEBHOOK_URL = os.getenv('WEBHOOK_URL')

# Conversation states
SELECT_GRADE_COLUMN = 1

# Gender detection and mapping
def detect_gender_column(df):
    """Detect gender column with support for multiple languages"""
    for col in df.columns:
        col_lower = str(col).lower()
        
        # Check for common keywords in column name
        if any(keyword in col_lower for keyword in ['sex', 'gender', 'ወታዊ', 'ጾታ', 'jinsia']):
            return col
        
        # Check for unique values indicative of gender
        unique_values = df[col].dropna().astype(str).str.upper().unique()
        
        male_indicators = any(v in ['M', 'MALE', 'ወ', 'ወንድ', 'MUME'] for v in unique_values)
        female_indicators = any(v in ['F', 'FEMALE', 'ሴ', 'ሴት', 'KIKE'] for v in unique_values)
        
        if male_indicators and female_indicators:
            return col
    
    return None

def clean_gender_values(series):
    """Clean and standardize gender values to MALE/FEMALE"""
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
            .dropna())

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the /start command is issued."""
    await update.message.reply_text(
        '📊 **Grade Distribution Bot**\n\n'
        'Send me a spreadsheet (`.xlsx` or `.csv`) with student grades to analyze gender performance. '
        'Make sure your file has a column for gender and a column for grades.',
        parse_mode='Markdown'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a help message when the /help command is issued."""
    help_text = """
📘 **Bot Help Guide**

1.  **Prepare your file:** Send an Excel (`.xlsx`) or CSV (`.csv`) file.
    * It must contain a **gender column** (e.g., M/F, Male/Female, ወ/ሴ, MUME/KIKE).
    * It must contain a **grade column** (numeric values, preferably out of 100).

2.  **How the bot works:**
    * The bot will automatically try to detect the gender column.
    * It will then ask you to select the correct grade column from a list.
    * Finally, it will provide a detailed statistical report on grade distribution by gender.

3.  **Supported formats:**
    * **Gender values:** 'M', 'F', 'Male', 'Female', 'Boy', 'Girl', 'ወ', 'ሴ', 'ወንድ', 'ሴት', 'MUME', 'KIKE'.
    * **File types:** `.xlsx`, `.csv` (with headers).
"""
    await update.message.reply_text(help_text, parse_mode='Markdown')

---

## File Processing

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles incoming document (Excel/CSV) files."""
    user = update.message.from_user
    logger.info(f"File received from {user.first_name} (ID: {user.id})")
    
    try:
        file = await context.bot.get_file(update.message.document.file_id)
        file_buffer = io.BytesIO()
        await file.download_to_memory(out=file_buffer)
        file_buffer.seek(0)

        # Try reading as Excel, then as CSV
        try:
            df = pd.read_excel(file_buffer)
            logger.info("File read as Excel.")
        except Exception:
            file_buffer.seek(0) # Reset buffer position for CSV read
            df = pd.read_csv(file_buffer)
            logger.info("File read as CSV.")

        # Clean column names (strip whitespace)
        df.columns = [str(col).strip() for col in df.columns]
        logger.info(f"Detected columns: {list(df.columns)}")

        # Detect gender column
        sex_col = detect_gender_column(df)
        if not sex_col:
            await update.message.reply_text(
                "❌ **Error:** Couldn't automatically detect a gender column in your file.\n\n"
                "Please ensure one of your columns contains gender-related keywords "
                "or values (e.g., 'Gender', 'Sex', 'M', 'F', 'Male', 'Female').\n\n"
                f"Found columns:\n`{'`, `'.join(df.columns)}`",
                parse_mode='Markdown'
            )
            return ConversationHandler.END

        # Clean and filter gender values
        df[sex_col] = clean_gender_values(df[sex_col])
        df = df[df[sex_col].isin(['MALE', 'FEMALE'])]
        
        if df.empty:
            await update.message.reply_text(
                "❌ **Error:** No valid gender data (MALE/FEMALE) found after processing. "
                "Please check your gender column for valid entries.",
                parse_mode='Markdown'
            )
            return ConversationHandler.END

        # Store DataFrame and gender column in user_data for later use
        context.user_data['df'] = df
        context.user_data['sex_col'] = sex_col

        # Prepare keyboard for grade column selection
        # Exclude the identified gender column from grade column options
        keyboard = [
            [InlineKeyboardButton(col, callback_data=str(i))]
            for i, col in enumerate(df.columns)
            if col != sex_col
        ]
        
        await update.message.reply_text(
            f"✅ **Gender column detected:** `{sex_col}`\n\n"
            "📝 Now, please **select the column that contains the student grades**:",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
        return SELECT_GRADE_COLUMN

    except Exception as e:
        logger.error(f"Error processing file for user {user.id}: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ **An unexpected error occurred while processing your file.** "
            "Please ensure it's a valid `.xlsx` or `.csv` file with headers and try again.",
            parse_mode='Markdown'
        )
        return ConversationHandler.END

async def select_column_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the user's selection of the grade column."""
    query = update.callback_query
    await query.answer() # Acknowledge the callback query

    try:
        selection = int(query.data)
        df = context.user_data.get('df')
        sex_col = context.user_data.get('sex_col')

        if df is None or sex_col is None:
            await query.edit_message_text(
                "❌ **Error:** Data not found. Please restart by sending your file again.",
                parse_mode='Markdown'
            )
            return ConversationHandler.END

        grade_col = df.columns[selection]

        # Convert grade column to numeric, coercing errors (non-numeric become NaN)
        df[grade_col] = pd.to_numeric(df[grade_col], errors='coerce')
        df = df.dropna(subset=[grade_col]) # Remove rows where grade is NaN
        
        if df.empty:
            await query.edit_message_text(
                "❌ **Error:** No valid numeric grades found in the selected column after cleaning. "
                "Please ensure the column contains numbers.",
                parse_mode='Markdown'
            )
            return ConversationHandler.END

        # Generate and send the report
        result = generate_report(df, sex_col, grade_col)
        await query.edit_message_text(result, parse_mode='Markdown')
        
        # Clear user data after report is generated
        context.user_data.clear()
        return ConversationHandler.END

    except Exception as e:
        logger.error(f"Error during column selection callback: {e}", exc_info=True)
        await query.edit_message_text(
            "❌ **An unexpected error occurred while analyzing your data.** Please try again.",
            parse_mode='Markdown'
        )
        # Clear user data on error to allow a clean restart
        if 'df' in context.user_data:
            del context.user_data['df']
        if 'sex_col' in context.user_data:
            del context.user_data['sex_col']
        return ConversationHandler.END

---

## Report Generation

def generate_report(df, sex_col, grade_col):
    """Generates a formatted statistics report based on gender and grade distribution."""
    
    # Define new grade bins and labels for 0-49, 50-74, 75-100
    bins = [-1, 49, 74, 100] # -1 ensures 0 is included in the first bin
    labels = ['0-49', '50-74', '75-100']
    
    # Create a new column 'grade_range' based on the defined bins
    # `right=True` (default) means the bin includes the rightmost edge, e.g., (0, 49]
    df['grade_range'] = pd.cut(df[grade_col], bins=bins, labels=labels, right=True)

    # Create a cross-tabulation for gender vs. grade range distribution
    # `margins=True` adds 'All' row/column for totals
    dist = pd.crosstab(df[sex_col], df['grade_range'], margins=True)
    
    # Reindex columns to ensure all labels are present, even if no data, and 'All' is last
    dist = dist.reindex(columns=labels + ['All'], fill_value=0)

    # Calculate overall statistics
    stats = {
        'total': len(df),
        'male': len(df[df[sex_col] == 'MALE']),
        'female': len(df[df[sex_col] == 'FEMALE']),
        'avg': round(df[grade_col].mean(), 2),
        'pass_rate': round((df[grade_col] >= 50).mean() * 100, 1) # Assuming 50 is the passing grade
    }

    # Build the report string
    report = [
        "📊 **Grade Analysis Report**",
        f"🔹 **Columns:** Gender=`{sex_col}`, Grades=`{grade_col}`",
        "",
        "🔢 **Grade Distribution**",
        "```", # Start of preformatted text block
        "┌──────────┬────────┬────────┬──────────┬──────┐",
        "│ Gender   │ 0-49   │ 50-74  │ 75-100   │ All  │",
        "├──────────┼────────┼────────┼──────────┼──────┤",
    ]

    # Add data rows for MALE, FEMALE, and All (total)
    for gender in ['MALE', 'FEMALE', 'All']:
        row = f"│ {gender.ljust(8)} │" # Left-align gender name
        for cat in labels + ['All']:
            value = str(dist.loc[gender, cat])
            # Adjust padding for each column type to ensure alignment
            if cat == '0-49' or cat == '50-74':
                row += f" {value.center(6)} │"
            elif cat == '75-100':
                row += f" {value.center(8)} │"
            else: # 'All' column
                row += f" {value.center(4)} │"
        report.append(row)
    
    report += [
        "└──────────┴────────┴────────┴──────────┴──────┘",
        "```", # End of preformatted text block
        "",
        "📌 **Statistics**",
        "```", # Start of preformatted text block
        f"Total Students: {stats['total']}",
        f"Male: {stats['male']}, Female: {stats['female']}",
        f"Average Grade: {stats['avg']}", # Average grade is out of 100
        f"Pass Rate (>= 50): {stats['pass_rate']}%",
        "```" # End of preformatted text block
    ]

    return "\n".join(report)

---

## Application Setup

def setup_application() -> Application:
    """Sets up the Telegram application with handlers and defines the conversation flow."""
    application = Application.builder().token(TOKEN).build()

    # Conversation handler for file upload and column selection
    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.Document.ALL, handle_document)],
        states={
            SELECT_GRADE_COLUMN: [CallbackQueryHandler(select_column_callback)]
        },
        fallbacks=[CommandHandler("cancel", lambda u, c: ConversationHandler.END)] # Simple cancel command
    )

    # Register command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(conv_handler) # Add the conversation handler

    return application

def main() -> None:
    """Main entry point for running the bot."""
    if not TOKEN or TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set or is default. Please set it.")
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set. Get your token from BotFather.")

    application = setup_application()

    # Deployment configuration for Render (using webhooks) or local polling
    if WEBHOOK_URL:
        logger.info(f"Running in webhook mode. Listening on port {PORT}, URL: {WEBHOOK_URL}/{TOKEN}")
        application.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=TOKEN, # This path must match the end of your webhook URL
            webhook_url=f"{WEBHOOK_URL}/{TOKEN}"
        )
    else:
        logger.info("Running in polling mode (for local development/testing).")
        application.run_polling(poll_interval=3) # Poll every 3 seconds

if __name__ == '__main__':
    main()
