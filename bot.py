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
from flask import Flask
import threading
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import numpy as np

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
SELECT_GRADE_COLUMN, SELECT_ACTION = 1, 2

# Color scheme
COLORS = {
    'male': '#3498db',
    'female': '#e91e63',
    'total': '#2ecc71',
    'text': '#333333',
    'background': '#f9f9f9'
}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message with interactive buttons."""
    keyboard = [
        [InlineKeyboardButton("📊 Analyze Grades", callback_data='analyze')],
        [InlineKeyboardButton("ℹ️ Help", callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        '📊 *Grade Distribution Analyzer*\n\n'
        'I can analyze student grade distributions by gender and provide '
        'detailed statistics and visualizations.\n\n'
        'Send me an Excel/CSV file or click "Analyze" to get started!',
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show help information."""
    help_text = (
        "📚 *How to use this bot:*\n\n"
        "1. Send an Excel/CSV file containing student data\n"
        "2. The file should have column headers in the first row\n"
        "3. I'll automatically detect:\n"
        "   - Gender/Sex column (looking for 'sex', 'gender', or M/F values)\n"
        "   - Grade column (you'll select this)\n\n"
        "📌 *Tips for best results:*\n"
        "- Ensure your grade column contains only numbers\n"
        "- Gender data should be M/F or Male/Female\n"
        "- Files should not be password protected\n\n"
        "📊 *Features:*\n"
        "- Grade distribution by gender\n"
        "- Pass/fail rates\n"
        "- Visual charts and graphs\n"
        "- Statistical analysis"
    )
    
    await update.message.reply_text(
        help_text,
        parse_mode='Markdown',
        disable_web_page_preview=True
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button presses."""
    query = update.callback_query
    await query.answer()
    
    if query.data == 'analyze':
        await query.edit_message_text(
            "Please send me an Excel or CSV file with student grades.\n\n"
            "I'll automatically detect the gender column and ask you to "
            "select the grade column to analyze.",
            parse_mode='Markdown'
        )
    elif query.data == 'help':
        await help_command(update, context)

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process uploaded spreadsheet file."""
    try:
        user = update.message.from_user
        logger.info(f"File received from {user.first_name} (ID: {user.id})")
        
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

        # Auto-detect sex column with improved detection
        sex_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['sex', 'gender', 'جنسیت', 'género']):
                sex_col = col
                break
        
        # If no explicit column found, look for M/F values
        if not sex_col:
            for col in df.columns:
                unique_vals = df[col].dropna().astype(str).str.upper().unique()
                if any(val in ['M', 'F', 'MALE', 'FEMALE', 'MASCULINO', 'FEMENINO'] for val in unique_vals):
                    sex_col = col
                    break

        if not sex_col:
            await update.message.reply_text(
                "❌ *Couldn't detect sex/gender column.*\n\n"
                "I looked for:\n"
                "- Columns named 'sex' or 'gender'\n"
                "- Columns containing M/F or Male/Female values\n\n"
                "Here are the columns I found:\n" + 
                "\n".join([f"- `{col}`" for col in df.columns]) +
                "\n\nPlease make sure your file includes gender information and try again.",
                parse_mode='Markdown'
            )
            return ConversationHandler.END

        # Clean sex data
        df[sex_col] = df[sex_col].astype(str).str.strip().str.upper()
        df[sex_col] = df[sex_col].replace({
            'M': 'MALE',
            'F': 'FEMALE',
            'MALE': 'MALE', 
            'FEMALE': 'FEMALE',
            'MASCULINO': 'MALE',
            'FEMENINO': 'FEMALE'
        })
        df = df[df[sex_col].isin(['MALE', 'FEMALE'])]

        # Store data
        context.user_data['df'] = df
        context.user_data['sex_col'] = sex_col

        # Create column selection keyboard
        keyboard = [
            [InlineKeyboardButton(col, callback_data=str(i))]
            for i, col in enumerate(df.columns)
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"✅ Detected gender column: `{sex_col}`\n\n"
            "📋 *Please select the grade column to analyze:*",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

        return SELECT_GRADE_COLUMN

    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ *Error processing your file.*\n\n"
            "Please ensure:\n"
            "- First row contains column headers\n"
            "- Contains gender information\n"
            "- File is not password protected\n"
            "- File format is Excel (.xlsx) or CSV\n\n"
            f"Technical details: {str(e)}",
            parse_mode='Markdown'
        )
        return ConversationHandler.END

async def select_grade_column(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle grade column selection."""
    query = update.callback_query
    await query.answer()
    
    try:
        selection = int(query.data)
        df = context.user_data['df'].copy()
        sex_col = context.user_data['sex_col']

        if selection < 0 or selection >= len(df.columns):
            await query.edit_message_text(
                "⚠️ Invalid column selection. Please try again.",
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

        # Store selected column
        context.user_data['grade_col'] = grade_col
        context.user_data['processed_df'] = df

        # Show action selection
        keyboard = [
            [InlineKeyboardButton("📈 Basic Statistics", callback_data='basic')],
            [InlineKeyboardButton("📊 Detailed Analysis", callback_data='detailed')],
            [InlineKeyboardButton("📉 View Charts", callback_data='charts')],
            [InlineKeyboardButton("📤 Export Results", callback_data='export')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            f"✅ Selected grade column: `{grade_col}`\n\n"
            "📌 *What would you like to see?*",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

        return SELECT_ACTION

    except Exception as e:
        logger.error(f"Error in grade selection: {e}", exc_info=True)
        await query.edit_message_text(
            "❌ Error processing your selection.\n"
            "Please try again or send a new file.",
            parse_mode='Markdown'
        )
        return ConversationHandler.END

async def select_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle action selection after grade column is chosen."""
    query = update.callback_query
    await query.answer()
    
    action = query.data
    df = context.user_data['processed_df']
    sex_col = context.user_data['sex_col']
    grade_col = context.user_data['grade_col']
    
    try:
        if action == 'basic':
            result = generate_basic_stats(df, sex_col, grade_col)
            await query.edit_message_text(
                result,
                parse_mode='Markdown'
            )
        elif action == 'detailed':
            result = generate_detailed_analysis(df, sex_col, grade_col)
            await query.edit_message_text(
                result,
                parse_mode='Markdown'
            )
        elif action == 'charts':
            await send_charts(update, context, df, sex_col, grade_col)
        elif action == 'export':
            await export_results(update, context, df, sex_col, grade_col)
            
        # Show menu again
        keyboard = [
            [InlineKeyboardButton("📈 Basic Statistics", callback_data='basic')],
            [InlineKeyboardButton("📊 Detailed Analysis", callback_data='detailed')],
            [InlineKeyboardButton("📉 View Charts", callback_data='charts')],
            [InlineKeyboardButton("📤 Export Results", callback_data='export')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="📌 *What would you like to see next?*",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        
        return SELECT_ACTION
        
    except Exception as e:
        logger.error(f"Error in action handling: {e}", exc_info=True)
        await query.edit_message_text(
            "❌ Error generating results.\n"
            f"Technical details: {str(e)}",
            parse_mode='Markdown'
        )
        return ConversationHandler.END

def generate_basic_stats(df, sex_col, grade_col):
    """Generate basic statistics message."""
    total_students = len(df)
    male_count = len(df[df[sex_col] == 'MALE'])
    female_count = len(df[df[sex_col] == 'FEMALE'])
    avg_grade = round(float(df[grade_col].mean()), 2)
    pass_rate = round(float(len(df[df[grade_col] >= 5]) / total_students * 100), 1)
    
    result = (
        f"📊 *Basic Statistics*\n\n"
        f"🔹 *Using Columns:*\n"
        f"- Gender: `{sex_col}`\n"
        f"- Grades: `{grade_col}`\n\n"
        f"👥 *Students*\n"
        f"```\n"
        f"Total:    {total_students}\n"
        f"Male:     {male_count}\n"
        f"Female:   {female_count}\n"
        f"```\n\n"
        f"📝 *Grades*\n"
        f"```\n"
        f"Average:  {avg_grade}/10\n"
        f"Pass Rate: {pass_rate}% (≥5)\n"
        f"```"
    )
    
    return result

def generate_detailed_analysis(df, sex_col, grade_col):
    """Generate detailed analysis with grade distribution."""
    # Define grade ranges
    bins = [-1, 0, 4, 7, 10]
    labels = ['0', '1-4', '5-7', '8-10']
    
    # Create grade ranges
    df['grade_range'] = pd.cut(df[grade_col], bins=bins, labels=labels, right=True)
    
    # Create cross-tabulation
    grade_dist = pd.crosstab(
        index=df[sex_col],
        columns=df['grade_range'],
        margins=True,
        margins_name="TOTAL"
    ).reindex(columns=labels + ['TOTAL'], fill_value=0)
    
    # Calculate percentages
    grade_pct = grade_dist.div(grade_dist['TOTAL'], axis=0) * 100
    grade_pct = grade_pct.round(1)
    
    # Build result message
    result = f"📊 *Detailed Analysis*\n\n"
    result += f"🔹 *Using Columns:*\n"
    result += f"- Gender: `{sex_col}`\n"
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
    
    # Percentages table
    result += "📈 *Grade Distribution Percentages*\n"
    result += "```\n"
    result += "┌──────────┬──────┬──────┬──────┬──────┬───────┐\n"
    result += "│ Gender   │  0   │ 1-4  │ 5-7  │ 8-10 │ TOTAL │\n"
    result += "├──────────┼──────┼──────┼──────┼──────┼───────┤\n"
    
    for sex in grade_pct.index:
        sex_display = "TOTAL" if sex == "TOTAL" else sex.capitalize()
        row = f"│ {sex_display.ljust(8)} │"
        for grade in labels + ['TOTAL']:
            pct = grade_pct.loc[sex, grade]
            row += f" {str(pct).rjust(4)}%│"
        result += row + "\n"
        
        if sex != grade_pct.index[-1]:
            result += "├──────────┼──────┼──────┼──────┼──────┼───────┤\n"
    
    result += "└──────────┴──────┴──────┴──────┴──────┴───────┘\n"
    result += "```\n\n"
    
    # Additional statistics
    result += "📌 *Key Statistics*\n"
    result += "```\n"
    
    for gender in ['MALE', 'FEMALE', 'TOTAL']:
        if gender == 'TOTAL':
            subset = df
            prefix = "All"
        else:
            subset = df[df[sex_col] == gender]
            prefix = gender.capitalize()
        
        count = len(subset)
        if count == 0:
            continue
            
        avg = subset[grade_col].mean()
        median = subset[grade_col].median()
        pass_rate = len(subset[subset[grade_col] >= 5]) / count * 100
        top_rate = len(subset[subset[grade_col] >= 8]) / count * 100
        
        result += (
            f"{prefix} Students: {count}\n"
            f"{prefix} Average: {avg:.1f}\n"
            f"{prefix} Median: {median:.1f}\n"
            f"{prefix} Pass Rate: {pass_rate:.1f}%\n"
            f"{prefix} Top Grades (8+): {top_rate:.1f}%\n"
            f"────────────────────\n"
        )
    
    result += "```"
    
    return result

async def send_charts(update: Update, context: ContextTypes.DEFAULT_TYPE, df, sex_col, grade_col):
    """Generate and send visualization charts."""
    query = update.callback_query
    chat_id = query.message.chat_id
    
    try:
        # Create figures
        fig1 = create_distribution_plot(df, sex_col, grade_col)
        fig2 = create_box_plot(df, sex_col, grade_col)
        fig3 = create_pass_rate_chart(df, sex_col, grade_col)
        
        # Send plots
        await context.bot.send_message(
            chat_id=chat_id,
            text="📊 *Visualizing Grade Data*",
            parse_mode='Markdown'
        )
        
        # Send each figure
        for fig in [fig1, fig2, fig3]:
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            await context.bot.send_photo(chat_id=chat_id, photo=buf)
            plt.close(fig)
            
    except Exception as e:
        logger.error(f"Error generating charts: {e}", exc_info=True)
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Error generating charts. Please try again.",
            parse_mode='Markdown'
        )

def create_distribution_plot(df, sex_col, grade_col):
    """Create grade distribution histogram."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Create histogram
    ax = sns.histplot(
        data=df,
        x=grade_col,
        hue=sex_col,
        bins=10,
        kde=True,
        palette={'MALE': COLORS['male'], 'FEMALE': COLORS['female']},
        alpha=0.7,
        edgecolor='white'
    )
    
    # Add vertical line at passing grade
    plt.axvline(x=5, color='red', linestyle='--', linewidth=1)
    plt.text(5.1, plt.ylim()[1]*0.9, 'Passing Grade (5)', color='red')
    
    # Customize plot
    plt.title('Grade Distribution by Gender', fontsize=14, pad=20)
    plt.xlabel('Grade (0-10 scale)', fontsize=12)
    plt.ylabel('Number of Students', fontsize=12)
    plt.legend(title='Gender', labels=['Male', 'Female'])
    
    # Set background color
    ax.set_facecolor(COLORS['background'])
    plt.gcf().set_facecolor(COLORS['background'])
    
    return plt.gcf()

def create_box_plot(df, sex_col, grade_col):
    """Create box plot of grades by gender."""
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    # Create boxplot
    ax = sns.boxplot(
        data=df,
        x=sex_col,
        y=grade_col,
        palette={'MALE': COLORS['male'], 'FEMALE': COLORS['female']},
        width=0.5
    )
    
    # Add swarm plot for individual points
    sns.swarmplot(
        data=df,
        x=sex_col,
        y=grade_col,
        color='black',
        alpha=0.4,
        size=3
    )
    
    # Customize plot
    plt.title('Grade Distribution Comparison', fontsize=14, pad=20)
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Grade (0-10 scale)', fontsize=12)
    plt.xticks([0, 1], ['Male', 'Female'])
    
    # Add horizontal line at passing grade
    plt.axhline(y=5, color='red', linestyle='--', linewidth=1)
    plt.text(1.5, 5.1, 'Passing Grade', color='red')
    
    # Set background color
    ax.set_facecolor(COLORS['background'])
    plt.gcf().set_facecolor(COLORS['background'])
    
    return plt.gcf()

def create_pass_rate_chart(df, sex_col, grade_col):
    """Create pass rate comparison chart."""
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    # Calculate pass rates
    pass_rates = []
    for gender in ['MALE', 'FEMALE']:
        subset = df[df[sex_col] == gender]
        pass_rate = len(subset[subset[grade_col] >= 5]) / len(subset) * 100
        pass_rates.append(pass_rate)
    
    # Create bar plot
    ax = sns.barplot(
        x=['Male', 'Female'],
        y=pass_rates,
        palette=[COLORS['male'], COLORS['female']],
        alpha=0.7
    )
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1f}%", 
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center', 
            xytext=(0, 10), 
            textcoords='offset points'
        )
    
    # Customize plot
    plt.title('Pass Rate Comparison (Grade ≥5)', fontsize=14, pad=20)
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Pass Rate (%)', fontsize=12)
    plt.ylim(0, 100)
    
    # Set background color
    ax.set_facecolor(COLORS['background'])
    plt.gcf().set_facecolor(COLORS['background'])
    
    return plt.gcf()

async def export_results(update: Update, context: ContextTypes.DEFAULT_TYPE, df, sex_col, grade_col):
    """Export analysis results to Excel."""
    query = update.callback_query
    chat_id = query.message.chat_id
    
    try:
        # Create Excel file in memory
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Save raw data
            df.to_excel(writer, sheet_name='Raw Data', index=False)
            
            # Create summary sheets
            create_summary_sheets(writer, df, sex_col, grade_col)
            
            # Create charts sheet
            create_charts_sheet(writer, df, sex_col, grade_col)
        
        # Prepare file for sending
        output.seek(0)
        await context.bot.send_document(
            chat_id=chat_id,
            document=output,
            filename='grade_analysis.xlsx',
            caption="📤 Here's your complete grade analysis export."
        )
        
    except Exception as e:
        logger.error(f"Error exporting results: {e}", exc_info=True)
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Error exporting results. Please try again.",
            parse_mode='Markdown'
        )

def create_summary_sheets(writer, df, sex_col, grade_col):
    """Create summary sheets in Excel export."""
    workbook = writer.book
    
    # Grade distribution sheet
    bins = [-1, 0, 4, 7, 10]
    labels = ['0', '1-4', '5-7', '8-10']
    df['grade_range'] = pd.cut(df[grade_col], bins=bins, labels=labels, right=True)
    
    grade_dist = pd.crosstab(
        index=df[sex_col],
        columns=df['grade_range'],
        margins=True,
        margins_name="TOTAL"
    ).reindex(columns=labels + ['TOTAL'], fill_value=0)
    
    grade_dist.to_excel(writer, sheet_name='Grade Distribution')
    
    # Statistics sheet
    stats_data = []
    for gender in ['MALE', 'FEMALE', 'TOTAL']:
        if gender == 'TOTAL':
            subset = df
            prefix = "All"
        else:
            subset = df[df[sex_col] == gender]
            prefix = gender.capitalize()
        
        count = len(subset)
        if count == 0:
            continue
            
        stats_data.append({
            'Group': prefix,
            'Students': count,
            'Average': subset[grade_col].mean(),
            'Median': subset[grade_col].median(),
            'Std Dev': subset[grade_col].std(),
            'Min': subset[grade_col].min(),
            'Max': subset[grade_col].max(),
            'Pass Rate (%)': len(subset[subset[grade_col] >= 5]) / count * 100,
            'Top Grades (%)': len(subset[subset[grade_col] >= 8]) / count * 100
        })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_excel(writer, sheet_name='Statistics', index=False)
    
    # Formatting
    for sheet in ['Grade Distribution', 'Statistics']:
        worksheet = writer.sheets[sheet]
        worksheet.set_column('A:Z', 15)

def create_charts_sheet(writer, df, sex_col, grade_col):
    """Create charts in Excel export."""
    workbook = writer.book
    worksheet = workbook.add_worksheet('Charts')
    
    # Create a chart object
    chart = workbook.add_chart({'type': 'column'})
    
    # Grade distribution data
    bins = [-1, 0, 4, 7, 10]
    labels = ['0', '1-4', '5-7', '8-10']
    df['grade_range'] = pd.cut(df[grade_col], bins=bins, labels=labels, right=True)
    
    grade_dist = pd.crosstab(
        index=df[sex_col],
        columns=df['grade_range'],
        margins=True,
        margins_name="TOTAL"
    ).reindex(columns=labels + ['TOTAL'], fill_value=0)
    
    # Configure the chart
    chart.set_title({'name': 'Grade Distribution by Gender'})
    chart.set_x_axis({'name': 'Grade Range'})
    chart.set_y_axis({'name': 'Number of Students'})
    
    # Add data series
    for i, gender in enumerate(['MALE', 'FEMALE']):
        chart.add_series({
            'name': gender.capitalize(),
            'categories': ['Grade Distribution', 1, 1, 1, 4],
            'values': ['Grade Distribution', i+1, 1, i+1, 4],
            'fill': {'color': COLORS['male'] if gender == 'MALE' else COLORS['female']},
        })
    
    # Insert the chart
    worksheet.insert_chart('B2', chart)

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel the current operation."""
    await update.message.reply_text(
        'Operation cancelled. You can start over by sending a new file.',
        parse_mode='Markdown'
    )
    return ConversationHandler.END

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
        entry_points=[
            MessageHandler(filters.Document.ALL, handle_document),
            CallbackQueryHandler(handle_document, pattern='^analyze$')
        ],
        states={
            SELECT_GRADE_COLUMN: [CallbackQueryHandler(select_grade_column)],
            SELECT_ACTION: [CallbackQueryHandler(select_action)]
        },
        fallbacks=[
            CommandHandler('cancel', cancel),
            CommandHandler('help', help_command)
        ],
    )

    # Command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(conv_handler)
    
    # Button handler
    application.add_handler(CallbackQueryHandler(button_handler))

    # Run the bot
    logger.info("Bot is running and waiting for files...")
    try:
        application.run_polling()
    except KeyboardInterrupt:
        logger.info("\nBot stopped by user")
    except Exception as e:
        logger.error(f"Error running bot: {e}")

if __name__ == '__main__':
    load_dotenv()
    main()
