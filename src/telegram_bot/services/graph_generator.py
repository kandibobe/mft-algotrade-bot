import matplotlib.pyplot as plt
import io
from telegram import Update
from telegram.ext import ContextTypes
import pandas as pd

async def send_pnl_chart(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Generates and sends a PnL chart."""
    # Mock data for demonstration
    dates = pd.date_range(end=pd.Timestamp.now(), periods=10)
    pnl = [0, 1.2, 0.8, 2.5, 2.1, 3.4, 4.2, 3.9, 5.1, 4.8]
    
    plt.figure(figsize=(10, 6))
    plt.plot(dates, pnl, marker='o', linestyle='-', color='cyan')
    plt.fill_between(dates, pnl, color='cyan', alpha=0.1)
    plt.title('Stoic Citadel - 24h PnL Performance', color='white')
    plt.grid(True, alpha=0.2)
    
    # Style for dark mode
    plt.gcf().set_facecolor('#121212')
    plt.gca().set_facecolor('#121212')
    plt.gca().tick_params(colors='white')
    plt.gca().xaxis.label.set_color('white')
    plt.gca().yaxis.label.set_color('white')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#121212')
    buf.seek(0)
    plt.close()
    
    await context.bot.send_photo(chat_id=chat_id, photo=buf, caption="📊 <b>PnL Performance Report</b>", parse_mode='HTML')