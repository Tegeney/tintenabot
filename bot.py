import logging
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
import io
import os
from dotenv import load_dotenv
from flask import Flask
import threading

# Load environment variables
load_dotenv()
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', "7805729196:AAHCZrSmEnf4gyl7pQuDOxv058tGPXYs-P4")
PORT = int(os.environ.get('PORT', 5000))

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Create Flask app for Render
app = Flask(__name__)

@app.route('/')
def home():
    return "Telegram Grade Distribution Bot is running!"

def run_flask_app():
    app.run(host='0.0.0.0', port=PORT)

# Conversation states
SELECT_GRADE_COLUMN = 1

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '📊 Grade Distribution Bot\n\n'
        'Send me a spreadsheet with student grades to analyze.\n'
        'I will automatically detect column headers and analyze your data.'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        'Send an Excel/CSV file with student data. '
        'The first row should contain column headers.\n'
        'I will automatically detect sex/gender and grade columns.'
    )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        # Download the file
        file = await context.bot.get_file(update.message.document.file_id)
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

        # Auto-detect sex column (with improved detection)
        sex_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if 'sex' in col_lower or 'gender' in col_lower:
                sex_col = col
                break
        
        # If no explicit column found, look for M/F values
        if not sex_col:
            for col in df.columns:
                unique_vals = df[col].dropna().astype(str).str.upper().unique()
                if any(val in ['M', 'F', 'MALE', 'FEMALE'] for val in unique_vals):
                    sex_col = col
                    break

        if not sex_col:
            await update.message.reply_text(
                "❌ Couldn't detect sex/gender column.\n"
                "I looked for:\n"
                "- Columns named 'sex' or 'gender'\n"
                "- Columns containing M/F or Male/Female values\n\n"
                "Here are the columns I found:\n" + 
                "\n".join([f"- {col}" for col in df.columns])
            )
            return ConversationHandler.END

        # Clean sex data
        df[sex_col] = df[sex_col].astype(str).str.strip().str.upper()
        df[sex_col] = df[sex_col].replace({
            'M': 'MALE',
            'F': 'FEMALE',
            'MALE': 'MALE', 
            'FEMALE': 'FEMALE'
        })
        df = df[df[sex_col].isin(['MALE', 'FEMALE'])]

        # Store data
        context.user_data['df'] = df
        context.user_data['sex_col'] = sex_col

        # Show columns for grade selection
        columns = "\n".join([f"{i+1}. {col}" for i, col in enumerate(df.columns)])
        await update.message.reply_text(
            f"✅ Detected sex/gender column: {sex_col}\n\n"
            f"📋 Available columns:\n{columns}\n\n"
            "Please reply with the NUMBER of the grade column to analyze:",
            parse_mode='Markdown'
        )

        return SELECT_GRADE_COLUMN

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        await update.message.reply_text(
            "❌ Error processing your file.\n"
            "Please ensure:\n"
            "- First row contains column headers\n"
            "- Contains sex/gender information\n"
            "- File is not password protected"
        )
        return ConversationHandler.END

async def select_grade_column(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        selection = int(update.message.text) - 1
        df = context.user_data['df'].copy()
        sex_col = context.user_data['sex_col']

        if selection < 0 or selection >= len(df.columns):
            await update.message.reply_text(
                "⚠️ Invalid column number. Please try again.",
                parse_mode='Markdown'
            )
            return SELECT_GRADE_COLUMN
        
        grade_col = df.columns[selection]

        # Clean grade data
        df[grade_col] = pd.to_numeric(df[grade_col], errors='coerce')
        df = df.dropna(subset=[grade_col])

        # Convert to 0-10 scale if needed
        if df[grade_col].max() > 10:
            df[grade_col] = df[grade_col] / 10

        # Calculate statistics
        result = calculate_statistics(df, sex_col, grade_col)
        
        # Send results
        await update.message.reply_text(result, parse_mode='Markdown')
        
        return ConversationHandler.END

    except ValueError:
        await update.message.reply_text(
            "⚠️ Please enter a valid number.",
            parse_mode='Markdown'
        )
        return SELECT_GRADE_COLUMN
    except Exception as e:
        logger.error(f"Error in grade selection: {e}")
        await update.message.reply_text(
            "❌ Error processing your selection.\n"
            "Please try again or send a new file.",
            parse_mode='Markdown'
        )
        return ConversationHandler.END

def calculate_statistics(df, sex_col, grade_col):
    """Calculate grade distribution statistics with clear column info."""
    try:
        # Define grade ranges
        bins = [-1, 0, 4, 7, 10]
        labels = ['0', '1-4', '5-7', '8-10']
        
        # Create grade ranges
        df.loc[:, 'grade_range'] = pd.cut(df[grade_col], bins=bins, labels=labels, right=True)
        
        # Create cross-tabulation with all categories
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
        logger.error(f"Error in statistics calculation: {e}")
        return (
            f"❌ Error analyzing `{grade_col}`\n"
            f"Please check this column contains valid numerical grade data\n"
            f"Technical details: {str(e)}"
        )

def main() -> None:
    """Start the bot."""
    if not TOKEN:
        raise ValueError("No Telegram token provided in environment variables")
    
    # Start Flask server in a separate thread for Render
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()

    # Create Telegram bot application
    application = Application.builder().token(TOKEN).build()

    # Conversation handler
    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.Document.ALL, handle_document)],
        states={
            SELECT_GRADE_COLUMN: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_grade_column)],
        },
        fallbacks=[
            CommandHandler('cancel', lambda update, context: ConversationHandler.END),
            CommandHandler('help', help_command)
        ],
    )

    # Command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(conv_handler)

    # Run the bot
    print("Bot is running and waiting for files...")
    try:
        application.run_polling()
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        logger.error(f"Error running bot: {e}")

if __name__ == '__main__':
    load_dotenv()
    main()
