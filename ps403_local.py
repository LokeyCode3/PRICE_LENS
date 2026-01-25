import os
import sys
import json
import math
import datetime as dt
import numpy as np
import pandas as pd
from datasets import load_dataset
import xgboost as xgb
import shap
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import threading
from typing import Optional, cast
from db_manager import init_db, save_analysis_result, get_all_analysis_results

# Global dataset cache to avoid slow loads per request
DATA_DF: Optional[pd.DataFrame] = None
DATA_DATE_COL: Optional[str] = None
DATA_PRICE_COL: Optional[str] = None

def load_data_once():
    global DATA_DF, DATA_DATE_COL, DATA_PRICE_COL
    if DATA_DF is not None:
        return True
    try:
        ds = load_dataset(
            "ZombitX64/xauusd-gold-price-historical-data-2004-2025",
            data_files="XAU_1d_data.jsonl"
        )
        first = next(iter(ds.values()))
        df = cast(pd.DataFrame, first.to_pandas())
        dc, pc = find_columns(df)
        if not dc or not pc:
            return False
        DATA_DF = df
        DATA_DATE_COL = dc
        DATA_PRICE_COL = pc
        return True
    except Exception:
        return False

def parse_date(s):
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None

# Removed terminal prompt helper to avoid blocking in hosted environments

def find_columns(df):
    date_cols = [c for c in df.columns if c.lower() in ["date", "timestamp", "time"]]
    price_candidates = ["close", "adj close", "price", "last", "value"]
    price_cols = []
    for c in df.columns:
        if c.lower() in price_candidates:
            price_cols.append(c)
    if not date_cols or not price_cols:
        return None, None
    return date_cols[0], price_cols[0]

def window_slice(df, date_col, start, end):
    d = df.copy()
    d["__date__"] = pd.to_datetime(d[date_col]).dt.date
    return d[(d["__date__"] >= start) & (d["__date__"] <= end)].sort_values("__date__")

def compute_drivers(dfw, price_col):
    n = len(dfw)
    if n == 0:
        return None
    prices = dfw[price_col].astype(float).to_numpy()
    if n == 1:
        cost_index = 0.0
        demand_signal = 0.0
        inventory_pressure = "medium"
        inventory_code = 1
        competitor_price_gap = 0.0
        return {
            "cost_index": cost_index,
            "demand_signal": demand_signal,
            "inventory_pressure": inventory_pressure,
            "inventory_code": inventory_code,
            "competitor_price_gap": competitor_price_gap,
            "records": n
        }
    cost_index = float(((prices[-1] - prices[0]) / prices[0]) * 100.0)
    daily_change = np.diff(prices)
    consec_up = 0
    consec_down = 0
    cur_up = 0
    cur_down = 0
    for ch in daily_change:
        if ch > 0:
            cur_up += 1
            consec_up = max(consec_up, cur_up)
            cur_down = 0
        elif ch < 0:
            cur_down += 1
            consec_down = max(consec_down, cur_down)
            cur_up = 0
        else:
            cur_up = 0
            cur_down = 0
    x = np.arange(len(prices))
    slope = float(np.polyfit(x, prices, 1)[0]) if len(prices) >= 2 else 0.0
    demand_signal = float(slope * 100.0 + (consec_up - consec_down))
    returns = np.diff(prices) / prices[:-1]
    vol = float(np.std(returns)) if len(returns) > 0 else 0.0
    if vol < 0.003:
        inventory_pressure = "low"
        inventory_code = 0
    elif vol < 0.01:
        inventory_pressure = "medium"
        inventory_code = 1
    else:
        inventory_pressure = "high"
        inventory_code = 2
    roll = pd.Series(prices).rolling(7, min_periods=1).mean().to_numpy()
    competitor_price_gap = float(prices[-1] - roll[-1])
    return {
        "cost_index": cost_index,
        "demand_signal": demand_signal,
        "inventory_pressure": inventory_pressure,
        "inventory_code": inventory_code,
        "competitor_price_gap": competitor_price_gap,
        "records": n
    }

def build_training(df, date_col, price_col):
    df = df.sort_values(date_col)
    prices = df[price_col].astype(float).to_numpy()
    dates = pd.to_datetime(df[date_col]).dt.date.to_numpy()
    W = 14
    X = []
    y_dir = []
    y_mkt = []
    for i in range(W, len(df)):
        sub = df.iloc[i-W:i]
        drivers = compute_drivers(sub, price_col)
        if not drivers:
            continue
        pct = drivers["cost_index"]
        if pct > 0.5:
            direction = "Increasing"
        elif pct < -0.5:
            direction = "Decreasing"
        else:
            direction = "Stable"
        slope = drivers["demand_signal"]
        if slope > 0.5:
            market = "Bullish"
        elif slope < -0.5:
            market = "Bearish"
        else:
            market = "Neutral"
        X.append([
            drivers["cost_index"],
            drivers["demand_signal"],
            float(drivers["inventory_code"]),
            drivers["competitor_price_gap"]
        ])
        y_dir.append(direction)
        y_mkt.append(market)
    return np.array(X), np.array(y_dir), np.array(y_mkt)

def encode_labels(y):
    classes = sorted(list(set(y.tolist())))
    idx = {c:i for i,c in enumerate(classes)}
    return np.array([idx[v] for v in y]), classes

# Removed CLI runner to ensure hosted environments don't block on input

def analyze_range(start, end):
    ok = load_data_once()
    if not ok or DATA_DF is None:
        return {"error": "dataset_loading"}
    df = cast(pd.DataFrame, DATA_DF)
    dc = cast(str, DATA_DATE_COL)
    pc = cast(str, DATA_PRICE_COL)
    # Clamp requested end within available range to avoid empty windows on future dates
    df_dates = pd.to_datetime(df[dc]).dt.date
    max_date = df_dates.max()
    end_clamped = min(end, max_date)
    dfw = window_slice(df, dc, start, end_clamped)
    if len(dfw) < 1:
        return {"error": "window"}
    drivers = compute_drivers(dfw, pc)
    if drivers is None:
        return {"error": "drivers"}
    ci = drivers['cost_index']
    dsig = drivers['demand_signal']
    inv = drivers['inventory_pressure']
    gap = drivers['competitor_price_gap']
    if abs(ci) < 0.3:
        cost_status = "Stable"
    else:
        cost_status = ("Up " + f"{ci:.2f}%") if ci > 0 else ("Down " + f"{abs(ci):.2f}%")
    demand_status = "Steady"
    if dsig > 0.5:
        demand_status = "Rising"
    elif dsig < -0.5:
        demand_status = "Falling"
    if abs(gap) < 0.01:
        competitor_status = "At avg"
    else:
        competitor_status = ("Above avg by " + f"{gap:.2f}") if gap > 0 else ("Below avg by " + f"{abs(gap):.2f}")
    return {
        "range": f"{start} → {end_clamped}",
        "records": len(dfw),
        "cost_status": cost_status,
        "demand_status": demand_status,
        "inventory_status": inv,
        "competitor_status": competitor_status
    }

def render_history_page(results):
    h = []
    h.append("<html><head><title>Analysis History</title><meta name=viewport content='width=device-width, initial-scale=1'>")
    h.append("<script src='https://cdn.tailwindcss.com'></script><script>tailwind.config={corePlugins:{preflight:false}}</script>")
    h.append("""<style>
        body{background:#000;color:#fff;font-family:system-ui,Arial;margin:0;padding:20px;min-height:100vh}
        .history-card {
            background: rgba(20, 20, 20, 0.6);
            border: 1px solid rgba(255, 215, 0, 0.1);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
            transition: all 0.2s ease;
        }
        .history-card:hover {
            border-color: rgba(255, 215, 0, 0.4);
            transform: translateX(4px);
        }
        .animate-fade-in { animation: fadeIn 0.8s ease-out forwards; opacity: 0; transform: translateY(20px); }
        @keyframes fadeIn { to { opacity: 1; transform: translateY(0); } }
    </style>""")
    h.append("</head><body>")
    
    h.append("<div class='max-w-4xl mx-auto mt-12 mb-8 animate-fade-in'>")
    h.append("<div class='flex justify-between items-center mb-8'>")
    h.append("<h1 class='text-4xl font-bold text-yellow-500'>Analysis History</h1>")
    h.append("<a href='/' class='px-4 py-2 bg-yellow-900/30 text-yellow-500 rounded hover:bg-yellow-900/50 transition-colors'>Back to Home</a>")
    h.append("</div>")
    
    if not results:
        h.append("<div class='text-gray-400 text-center py-12'>No history found.</div>")
    else:
        h.append("<div class='space-y-4'>")
        for res in results:
            # res is a dict from RealDictCursor
            r_range = f"{res['from_date']} → {res['to_date']}"
            r_cost = res['cost_status']
            r_demand = res['demand_status']
            r_inv = res['inventory_status']
            r_comp = res['competitor_status']
            r_created = res['created_at'].strftime("%Y-%m-%d %H:%M")
            
            h.append(f"<div class='history-card'>")
            h.append(f"<div class='flex justify-between items-start mb-4 border-b border-gray-800 pb-2'>")
            h.append(f"<div class='text-xl font-bold text-white'>{r_range}</div>")
            h.append(f"<div class='text-xs text-gray-500'>{r_created}</div>")
            h.append(f"</div>")
            h.append(f"<div class='grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm'>")
            h.append(f"<div><span class='text-gray-400'>Cost:</span> <span class='text-yellow-400 font-mono'>{r_cost}</span></div>")
            h.append(f"<div><span class='text-gray-400'>Demand:</span> <span class='text-yellow-400 font-mono'>{r_demand}</span></div>")
            h.append(f"<div><span class='text-gray-400'>Inventory:</span> <span class='text-yellow-400 font-mono'>{r_inv}</span></div>")
            h.append(f"<div><span class='text-gray-400'>Competitor:</span> <span class='text-yellow-400 font-mono'>{r_comp}</span></div>")
            h.append(f"</div>")
            h.append(f"</div>")
        h.append("</div>")
        
    h.append("</div></body></html>")
    return "".join(h)

def render_analysis_page(start, end, result, error=None):
    h = []
    h.append("<html><head><title>Analysis - Bento Grid</title><meta name=viewport content='width=device-width, initial-scale=1'>")
    h.append("<script src='https://cdn.tailwindcss.com'></script><script>tailwind.config={corePlugins:{preflight:false}}</script>")
    h.append("""<style>
        body{background:#000;color:#fff;font-family:system-ui,Arial;margin:0;padding:20px;min-height:100vh}
        .bento-grid { display: grid; grid-template-columns: repeat(1, 1fr); gap: 1.5rem; max-width: 80rem; margin: 0 auto; padding: 2rem; }
        @media (min-width: 768px) { .bento-grid { grid-template-columns: repeat(3, 1fr); grid-auto-rows: minmax(180px, auto); } }
        
        .bento-card {
            background: rgba(20, 20, 20, 0.6);
            border: 1px solid rgba(255, 215, 0, 0.1);
            border-radius: 1.5rem;
            padding: 1.5rem;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        
        .bento-card:hover {
            transform: translateY(-4px) scale(1.01);
            border-color: rgba(255, 215, 0, 0.4);
            box-shadow: 0 20px 40px -10px rgba(255, 215, 0, 0.1);
            z-index: 10;
        }
        
        .bento-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255, 215, 0, 0.5), transparent);
            opacity: 0;
            transition: opacity 0.4s ease;
        }
        .bento-card:hover::before { opacity: 1; }
        
        .card-icon {
            width: 48px; height: 48px;
            border-radius: 12px;
            background: rgba(255, 215, 0, 0.1);
            display: flex; items-center; justify-content: center;
            color: #FFD700;
            margin-bottom: 1rem;
        }
        
        .card-title { font-size: 1.25rem; font-weight: 600; color: #e5e5e5; margin-bottom: 0.5rem; }
        .card-value { font-size: 2.5rem; font-weight: 700; color: #FFD700; letter-spacing: -0.02em; }
        .card-desc { font-size: 0.875rem; color: #a3a3a3; margin-top: 0.5rem; }
        
        .col-span-2 { grid-column: span 2; }
        .row-span-2 { grid-row: span 2; }
        
        .animate-fade-in { animation: fadeIn 0.8s ease-out forwards; opacity: 0; transform: translateY(20px); }
        @keyframes fadeIn { to { opacity: 1; transform: translateY(0); } }
        
        .delay-100 { animation-delay: 100ms; }
        .delay-200 { animation-delay: 200ms; }
        .delay-300 { animation-delay: 300ms; }
    </style>""")
    h.append("</head><body>")
    
    # Header
    h.append("<div class='text-center mt-12 mb-8 animate-fade-in'>")
    h.append("<h1 class='text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-yellow-200 via-yellow-400 to-yellow-600 mb-4'>Market Analysis</h1>")
    if start and end:
        h.append(f"<p class='text-gray-400'>Analysis Range: <span class='text-yellow-500'>{start}</span> to <span class='text-yellow-500'>{end}</span></p>")
    h.append("</div>")
    
    # Analysis List Section (Vertical, Top-to-Bottom)
    if result:
        h.append("<div class='max-w-4xl mx-auto mb-12 p-8 bg-neutral-900/90 border border-yellow-500/30 rounded-2xl shadow-2xl animate-fade-in delay-100'>")
        h.append("<h3 class='text-2xl font-bold text-yellow-400 mb-6 border-b border-yellow-500/20 pb-4'>Analysis Summary</h3>")
        h.append("<div class='space-y-4 text-lg'>")
        
        # 1. Records Used
        h.append("<div class='flex items-start sm:items-center flex-col sm:flex-row gap-1 sm:gap-4'>")
        h.append("<span class='text-gray-400 sm:w-64 font-medium'>Records used:</span>")
        h.append(f"<span class='text-white font-mono text-xl'>{result.get('records', 0)}</span>")
        h.append("</div>")
        
        # 2. Cost Status
        h.append("<div class='flex items-start sm:items-center flex-col sm:flex-row gap-1 sm:gap-4'>")
        h.append("<span class='text-gray-400 sm:w-64 font-medium'>Cost status:</span>")
        h.append(f"<span class='text-white font-mono text-xl'>{result.get('cost_status', 'N/A')}</span>")
        h.append("</div>")
        
        # 3. Demand Status
        h.append("<div class='flex items-start sm:items-center flex-col sm:flex-row gap-1 sm:gap-4'>")
        h.append("<span class='text-gray-400 sm:w-64 font-medium'>Demand status:</span>")
        h.append(f"<span class='text-white font-mono text-xl'>{result.get('demand_status', 'N/A')}</span>")
        h.append("</div>")
        
        # 4. Inventory Status
        h.append("<div class='flex items-start sm:items-center flex-col sm:flex-row gap-1 sm:gap-4'>")
        h.append("<span class='text-gray-400 sm:w-64 font-medium'>Inventory status <span class='text-xs text-yellow-600'>(simulated)</span>:</span>")
        h.append(f"<span class='text-white font-mono text-xl'>{result.get('inventory_status', 'N/A')}</span>")
        h.append("</div>")
        
        # 5. Competitor Status
        h.append("<div class='flex items-start sm:items-center flex-col sm:flex-row gap-1 sm:gap-4'>")
        h.append("<span class='text-gray-400 sm:w-64 font-medium'>Competitor status:</span>")
        h.append(f"<span class='text-white font-mono text-xl'>{result.get('competitor_status', 'N/A')}</span>")
        h.append("</div>")
        
        h.append("</div>") # End space-y-4
        h.append("</div>") # End container
    
    # Bento Grid
    h.append("<div class='bento-grid'>")
    
    if error:
        h.append(f"<div class='bento-card col-span-3 text-center items-center justify-center p-12 text-red-400 border-red-900'>{error}</div>")
    elif result:
        # Card 1: Cost Status (Large)
        h.append("<div class='bento-card col-span-2 animate-fade-in delay-100'>")
        h.append("<div class='card-icon'><svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><line x1='12' y1='1' x2='12' y2='23'></line><path d='M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6'></path></svg></div>")
        h.append("<div>")
        h.append("<div class='card-title'>Cost Momentum</div>")
        h.append(f"<div class='card-value'>{result.get('cost_status', 'N/A')}</div>")
        h.append("<div class='card-desc'>Current price trend analysis based on historical data points.</div>")
        h.append("</div>")
        h.append("</div>")
        
        # Card 2: Demand (Tall)
        h.append("<div class='bento-card row-span-2 animate-fade-in delay-200'>")
        h.append("<div class='card-icon'><svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M3 3v18h18'></path><path d='M18.7 8l-5.1 5.2-2.8-2.7L7 14.3'></path></svg></div>")
        h.append("<div>")
        h.append("<div class='card-title'>Demand Signal</div>")
        h.append(f"<div class='card-value text-4xl'>{result.get('demand_status', 'N/A')}</div>")
        h.append("<div class='card-desc mt-4'>Derived from slope and consecutive daily changes. Indicates market sentiment strength.</div>")
        h.append("</div>")
        h.append("<div class='mt-auto h-32 bg-gradient-to-t from-yellow-500/20 to-transparent rounded-lg w-full'></div>")
        h.append("</div>")
        
        # Card 3: Inventory (Standard)
        h.append("<div class='bento-card animate-fade-in delay-300'>")
        h.append("<div class='card-icon'><svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z'></path><polyline points='3.27 6.96 12 12.01 20.73 6.96'></polyline><line x1='12' y1='22.08' x2='12' y2='12'></line></svg></div>")
        h.append("<div class='card-title'>Inventory Pressure</div>")
        h.append(f"<div class='text-2xl font-bold text-white'>{result.get('inventory_status', 'N/A')}</div>")
        h.append("</div>")
        
        # Card 4: Records (Standard)
        h.append("<div class='bento-card animate-fade-in delay-300'>")
        h.append("<div class='card-icon'><svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'></circle><polyline points='12 6 12 12 16 14'></polyline></svg></div>")
        h.append("<div class='card-title'>Data Points</div>")
        h.append(f"<div class='text-2xl font-bold text-white'>{result.get('records', 0)}</div>")
        h.append("</div>")
        
        # Card 5: Competitor (Wide)
        h.append("<div class='bento-card col-span-2 animate-fade-in delay-300'>")
        h.append("<div class='card-icon'><svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2'></path><circle cx='9' cy='7' r='4'></circle><path d='M23 21v-2a4 4 0 0 0-3-3.87'></path><path d='M16 3.13a4 4 0 0 1 0 7.75'></path></svg></div>")
        h.append("<div class='flex justify-between items-end'>")
        h.append("<div><div class='card-title'>Competitor Gap</div><div class='card-value text-3xl'>" + str(result.get('competitor_status', 'N/A')) + "</div></div>")
        h.append("<div class='text-sm text-yellow-500 font-mono'>LIVE TRACKING</div>")
        h.append("</div>")
        h.append("</div>")
        
        # History Button Card
        h.append("<div class='bento-card flex items-center justify-center cursor-pointer hover:bg-yellow-900/20 group animate-fade-in delay-300' onclick=\"window.location.href='/history'\">")
        h.append("<div class='text-center transition-transform group-hover:-translate-y-1'>")
        h.append("<div class='text-4xl mb-2 text-yellow-500'>📜</div>")
        h.append("<div class='font-bold'>History</div>")
        h.append("</div>")
        h.append("</div>")
        
    # Back Button Card
    h.append("<div class='bento-card col-span-1 md:col-span-3 flex items-center justify-center cursor-pointer hover:bg-yellow-900/20 group' onclick=\"window.location.href='/'\">")
    h.append("<div class='text-center transition-transform group-hover:-translate-x-2 flex items-center justify-center gap-3'>")
    h.append("<div class='text-4xl text-yellow-500'>←</div>")
    h.append("<div class='font-bold text-lg'>Back to Home</div>")
    h.append("</div>")
    h.append("</div>")
    
    h.append("</div>") # End Grid
    h.append("</body></html>")
    return "".join(h)

def render_html(start, end, result, error=None):
    h = []
    h.append("<html><head><title>Gold Pricing UI</title><meta name=viewport content='width=device-width, initial-scale=1'>")
    h.append("<script src='https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js'></script><script src='https://cdn.tailwindcss.com'></script><script>tailwind.config={corePlugins:{preflight:false}}</script>")
    h.append("<style>body{background:#000000;color:#ffffff;font-family:system-ui,Arial;padding:0;margin:0;display:flex;justify-content:center;align-items:flex-start;min-height:100vh}input{padding:8px;margin:4px;background:#222;color:#fff;border:1px solid #444}button{padding:8px 12px;background:#333;color:#fff;border:1px solid #555;cursor:pointer}button:hover{background:#444}.card{border:none;padding:0;margin-top:12px}.loader{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;background:#000;z-index:9999;transition:transform 0.8s cubic-bezier(0.7, 0, 0.3, 1)}.spinner{width:70.4px;height:70.4px;--clr: rgb(247, 197, 159);--clr-alpha: rgb(247, 197, 159,.1);animation: spinner 1.6s infinite ease;transform-style: preserve-3d}.spinner>div{background-color: var(--clr-alpha);height:100%;position:absolute;width:100%;border:3.5px solid var(--clr)}.spinner div:nth-of-type(1){transform: translateZ(-35.2px) rotateY(180deg)}.spinner div:nth-of-type(2){transform: rotateY(-270deg) translateX(50%);transform-origin: top right}.spinner div:nth-of-type(3){transform: rotateY(270deg) translateX(-50%);transform-origin: center left}.spinner div:nth-of-type(4){transform: rotateX(90deg) translateY(-50%);transform-origin: top center}.spinner div:nth-of-type(5){transform: rotateX(-90deg) translateY(50%);transform-origin: bottom center}.spinner div:nth-of-type(6){transform: translateZ(35.2px)}@keyframes spinner{0%{transform: rotate(45deg) rotateX(-25deg) rotateY(25deg)}50%{transform: rotate(45deg) rotateX(-385deg) rotateY(25deg)}100%{transform: rotate(45deg) rotateX(-385deg) rotateY(385deg)}}#content{display:none}#shader-landing{display:none;position:fixed;inset:0;z-index:5000;background:transparent;color:#fff;display:flex;flex-direction:column;align-items:center;justify-content:center;transition:transform 0.8s cubic-bezier(0.7, 0, 0.3, 1)}.sparks{position:fixed;inset:0;z-index:2;pointer-events:none}.sparks canvas{width:100%;height:100%;display:block}</style>")
    h.append("</head><body>")
    
    h.append("<div id='loader' class='loader'><div class='spinner'><div></div><div></div><div></div><div></div><div></div><div></div></div></div>")
    
    # Global Shader Container (Moved out of landing page)
    h.append("<div id='shader-container' style='position:fixed;inset:0;z-index:0;'></div>")
    
    # Shader Landing Page (UI Only)
    h.append("<div id='shader-landing' style='display:none;'>")
    h.append("<style>")
    h.append("""
    .underline-path {
        stroke-dasharray: 400;
        stroke-dashoffset: 400;
        transition: stroke-dashoffset 1.5s ease-out, d 0.3s ease;
    }
    .animated-text-group:hover .underline-path {
        d: path("M10,15 Q200,5 390,15");
    }
    /* Animated Shiny Text CSS */
    .shiny-text {
        background: linear-gradient(90deg, #facc15, #fde047, #facc15);
        background-size: 200% auto;
        color: transparent;
        -webkit-background-clip: text;
        background-clip: text;
        animation: shine 1.5s linear infinite;
        /* Fallback color if gradient fails */
        text-fill-color: transparent;
        -webkit-text-fill-color: transparent;
    }
    .shiny-text:hover {
        /* Hover Glow Effect */
        filter: drop-shadow(0 0 8px rgba(250, 204, 21, 0.6));
    }
    /* Gold Input Style */
    .gold-input {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 215, 0, 0.3);
        color: #FFD700;
        padding: 10px 14px;
        width: 160px;
        border-radius: 8px;
        font-size: 1rem;
        outline: none;
        backdrop-filter: blur(4px);
        transition: all 0.3s ease;
        text-align: center;
        letter-spacing: 1px;
    }
    .gold-input::placeholder {
        color: rgba(255, 215, 0, 0.4);
        font-weight: 300;
    }
    .gold-input:focus {
        border-color: #FFD700;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.2);
        background: rgba(0, 0, 0, 0.6);
    }
    .gold-input:hover {
        border-color: rgba(255, 215, 0, 0.6);
    }
    @keyframes shine {
        to {
            background-position: 200% center;
        }
    }
    """)
    h.append("</style>")
    # REMOVED shader-container from here
    h.append("<div style='position:relative;z-index:1;text-align:center;pointer-events:none;'>")
    
    # Animated Underline + Shiny Text Implementation
    h.append("""
    <div class="animated-text-group" style="position:relative; display:inline-block; pointer-events:auto; margin-bottom:20px;">
        <h1 class="shiny-text" style="font-size:3rem; margin:0; position:relative; z-index:10; font-weight:bold;">Gold Analytics</h1>
        <svg class="animated-underline" viewBox="0 0 400 20" preserveAspectRatio="none" style="position:absolute; bottom:-5px; left:0; width:100%; height:20px; z-index:0; pointer-events:none; overflow:visible;">
            <path class="underline-path" d="M10,10 Q200,20 390,10" fill="transparent" stroke="#FFD700" stroke-width="4" stroke-linecap="round" />
        </svg>
    </div>
    """)
    
    h.append("<br>")
    h.append("<button id='enterBtn' style='pointer-events:auto;padding:12px 24px;font-size:1.2rem;background:rgba(255,215,0,0.2);border:1px solid #FFD700;color:#FFD700;border-radius:4px;cursor:pointer;transition:all 0.3s ease'>Enter Application</button>")
    h.append("</div>")
    h.append("</div>")

    
    # Particles Container
    h.append("<div style='position:relative;width:100%;min-height:100vh;overflow:hidden;display:flex;justify-content:center;padding-top:20px'>")
    
    # REMOVED Particles Background Layer (Shader is now global)
    # h.append("<div style='position:absolute;inset:0;z-index:0;pointer-events:none'>")
    # h.append("<canvas id='particles-canvas' style='width:100%;height:100%;display:block'></canvas>")
    # h.append("</div>")
    
    # Content Layer
    h.append("<div id='content' style='position:relative;z-index:1;display:none;flex-direction:column;align-items:center;gap:12px;text-align:center'>")
    
    
    h.append("<h2 class='shiny-text'>Gold Price Analysis</h2>")
    h.append("<form method='GET' action='/analyze' style='display:flex;flex-direction:column;gap:24px;align-items:center;width:100%'>")
    h.append("<div style='display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:center'>")
    h.append("<div style='position:relative'><input id='fromInput' type='text' name='from' required class='gold-input' placeholder='YYYY-MM-DD' autocomplete='off'></div>")
    h.append("<button id='btnFrom' type='button' style='padding:8px 12px;background:rgba(255,215,0,0.2);color:#FFD700;border:1px solid #FFD700;border-radius:4px;cursor:pointer;transition:all 0.3s ease'>From</button>")
    h.append("<div style='position:relative'><input id='toInput' type='text' name='to' class='gold-input' placeholder='YYYY-MM-DD' autocomplete='off'></div>")
    h.append("<button id='btnTo' type='button' style='padding:8px 12px;background:rgba(255,215,0,0.2);color:#FFD700;border:1px solid #FFD700;border-radius:4px;cursor:pointer;transition:all 0.3s ease'>To</button>")
    h.append("</div>")
    h.append("<button id='btnAnalyze' type='submit' style='padding:12px 32px; font-size:1.1rem; position: relative; overflow: visible;background:rgba(255,215,0,0.2);color:#FFD700;border:1px solid #FFD700;border-radius:4px;cursor:pointer;transition:all 0.3s ease'>Analyze</button>")
    h.append("</form>")
    
    # ClickSpark Implementation (Vanilla JS adaptation of @react-bits/ClickSpark-JS-CSS)
    h.append("<script>")
    h.append("""
    class ClickSpark {
        constructor(element, options = {}) {
            this.element = element;
            this.options = {
                sparkColor: options.sparkColor || '#fff',
                sparkSize: options.sparkSize || 10,
                sparkRadius: options.sparkRadius || 15,
                sparkCount: options.sparkCount || 8,
                duration: options.duration || 400,
            };
            this.init();
        }

        init() {
            this.element.addEventListener('click', (e) => this.animate(e));
        }

        animate(e) {
            const rect = this.element.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            for (let i = 0; i < this.options.sparkCount; i++) {
                this.createSpark(x, y);
            }
        }
        
        createSpark(x, y) {
            const spark = document.createElement('div');
            Object.assign(spark.style, {
                position: 'absolute',
                left: x + 'px',
                top: y + 'px',
                width: this.options.sparkSize + 'px',
                height: this.options.sparkSize + 'px',
                backgroundColor: this.options.sparkColor,
                borderRadius: '50%',
                pointerEvents: 'none',
                transform: 'translate(-50%, -50%) scale(1)',
                transition: `transform ${this.options.duration}ms ease-out, opacity ${this.options.duration}ms ease-out`,
                opacity: '1',
                zIndex: '9999'
            });
            
            this.element.appendChild(spark);
            
            const angle = Math.random() * Math.PI * 2;
            const radius = this.options.sparkRadius;
            const dist = Math.random() * radius + radius; 
            const tx = Math.cos(angle) * dist;
            const ty = Math.sin(angle) * dist;
            
            // Force reflow
            void spark.offsetWidth;
            
            requestAnimationFrame(() => {
                spark.style.transform = `translate(calc(-50% + ${tx}px), calc(-50% + ${ty}px)) scale(0)`;
                spark.style.opacity = '0';
            });
            
            setTimeout(() => {
                spark.remove();
            }, this.options.duration);
        }
    }

    // Initialize ClickSpark on the entire document body
    document.addEventListener('DOMContentLoaded', () => {
        new ClickSpark(document.body, {
            sparkColor: '#FFD700', // Gold color to match the theme
            sparkSize: 10,
            sparkRadius: 20,
            sparkCount: 8,
            duration: 400
        });
    });

    // Magnet Implementation (Vanilla JS adaptation of @react-bits/Magnet-JS-CSS)
    class Magnet {
        constructor(element, options = {}) {
            this.element = element;
            this.options = {
                padding: options.padding || 80,
                magnetStrength: options.magnetStrength || 2,
            };
            this.init();
        }

        init() {
            // Use standard styling to ensure smooth movement
            this.element.style.transition = 'transform 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
            document.addEventListener('mousemove', (e) => this.update(e));
            document.addEventListener('mouseleave', () => this.reset());
        }

        update(e) {
            const rect = this.element.getBoundingClientRect();
            // Calculate center using offset for better stability vs getBoundingClientRect which changes on transform
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;
            
            const distX = e.clientX - centerX;
            const distY = e.clientY - centerY;
            
            // Check if mouse is close enough (using padding)
            if (Math.abs(distX) < this.options.padding && Math.abs(distY) < this.options.padding) {
                const moveX = distX / this.options.magnetStrength;
                const moveY = distY / this.options.magnetStrength;
                this.element.style.transform = `translate(${moveX}px, ${moveY}px)`;
            } else {
                this.element.style.transform = 'translate(0, 0)';
            }
        }
        
        reset() {
            this.element.style.transform = 'translate(0, 0)';
        }
    }

    // Initialize Magnet on specific interactive buttons
    document.addEventListener('DOMContentLoaded', () => {
        const ids = ['btnFrom', 'btnTo', 'btnAnalyze', 'enterBtn'];
        ids.forEach(id => {
            const el = document.getElementById(id);

            if (el) {
                new Magnet(el, {
                    padding: 80,
                    magnetStrength: 2
                });
            }
        });
    });
    // Custom Gold Calendar
    class GoldCalendar {
        constructor(inputId) {
            this.input = document.getElementById(inputId);
            if(!this.input) return;
            
            this.wrapper = document.createElement('div');
            // Tailwind classes for the calendar container
            this.wrapper.className = 'absolute z-50 mt-2 p-4 bg-neutral-900 border border-yellow-600/30 rounded-lg shadow-2xl hidden w-72 text-white';
            this.input.parentNode.appendChild(this.wrapper);
            
            this.now = new Date();
            this.currentMonth = this.now.getMonth();
            this.currentYear = this.now.getFullYear();
            
            this.input.addEventListener('click', (e) => {
                e.stopPropagation();
                document.querySelectorAll('.calendar-wrapper').forEach(el => el.classList.add('hidden')); // Close others
                this.wrapper.classList.remove('hidden');
                this.render();
            });
            
            // Close on click outside
            document.addEventListener('click', (e) => {
                if (!this.wrapper.contains(e.target) && e.target !== this.input) {
                    this.wrapper.classList.add('hidden');
                }
            });
            
            this.wrapper.classList.add('calendar-wrapper');
        }

        render() {
            this.wrapper.innerHTML = '';
            
            // Header
            const header = document.createElement('div');
            header.className = 'flex justify-between items-center mb-4';
            const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
            
            const prevBtn = document.createElement('button');
            prevBtn.innerText = '<';
            prevBtn.className = 'p-1 hover:text-yellow-400 font-bold bg-transparent border-none';
            prevBtn.onclick = (e) => { e.stopPropagation(); this.changeMonth(-1); };
            
            const nextBtn = document.createElement('button');
            nextBtn.innerText = '>';
            nextBtn.className = 'p-1 hover:text-yellow-400 font-bold bg-transparent border-none';
            nextBtn.onclick = (e) => { e.stopPropagation(); this.changeMonth(1); };
            
            const title = document.createElement('div');
            title.className = 'font-bold text-yellow-500';
            title.innerText = `${monthNames[this.currentMonth]} ${this.currentYear}`;
            
            header.appendChild(prevBtn);
            header.appendChild(title);
            header.appendChild(nextBtn);
            this.wrapper.appendChild(header);
            
            // Days Header
            const daysHeader = document.createElement('div');
            daysHeader.className = 'grid grid-cols-7 gap-1 mb-2 text-center text-xs text-gray-500';
            ['Su','Mo','Tu','We','Th','Fr','Sa'].forEach(d => {
                const el = document.createElement('div');
                el.innerText = d;
                daysHeader.appendChild(el);
            });
            this.wrapper.appendChild(daysHeader);
            
            // Days Grid
            const grid = document.createElement('div');
            grid.className = 'grid grid-cols-7 gap-1';
            
            const firstDay = new Date(this.currentYear, this.currentMonth, 1).getDay();
            const daysInMonth = new Date(this.currentYear, this.currentMonth + 1, 0).getDate();
            
            // Empty slots
            for(let i=0; i<firstDay; i++) {
                grid.appendChild(document.createElement('div'));
            }
            
            // Days
            for(let d=1; d<=daysInMonth; d++) {
                const dayEl = document.createElement('div');
                dayEl.innerText = d;
                
                // --- GOLD HOVER EFFECT LOGIC ---
                let baseClasses = "h-8 w-8 flex items-center justify-center rounded-full text-sm cursor-pointer transition-colors duration-200 ease-in-out";
                
                // Hover classes: bg-gradient-to-r from-yellow-400 to-amber-500 text-black
                let hoverClasses = "hover:bg-gradient-to-r hover:from-yellow-400 hover:to-amber-500 hover:text-black hover:shadow-lg hover:font-bold";
                
                dayEl.className = `${baseClasses} ${hoverClasses} text-gray-200`;
                
                dayEl.onclick = (e) => {
                    e.stopPropagation();
                    const val = `${this.currentYear}-${String(this.currentMonth+1).padStart(2,'0')}-${String(d).padStart(2,'0')}`;
                    this.input.value = val;
                    this.wrapper.classList.add('hidden');
                };
                
                grid.appendChild(dayEl);
            }
            
            this.wrapper.appendChild(grid);
        }
        
        changeMonth(delta) {
            this.currentMonth += delta;
            if(this.currentMonth > 11) { this.currentMonth = 0; this.currentYear++; }
            if(this.currentMonth < 0) { this.currentMonth = 11; this.currentYear--; }
            this.render();
        }
    }

    document.addEventListener('DOMContentLoaded', () => {
        new GoldCalendar('fromInput');
        new GoldCalendar('toInput');
    });
    """)
    h.append("</script>")
    
    
    if error:
        if error == "dataset_loading":
            h.append("<div class='card'><b>Loading data…</b> Please try Analyze again in a moment.</div>")
        elif error == "window":
            h.append("<div class='card'><b>No data</b> for the selected range. Try a different date window.</div>")
        else:
            h.append(f"<div class='card'><b>Error:</b> {error}</div>")
    elif result:
        h.append("<div class='card'>")
        h.append(f"<div><b>Selected range:</b> {result['range']}</div>")
        h.append(f"<div><b>Records used:</b> {result['records']}</div>")
        h.append(f"<div><b>Cost status:</b> {result['cost_status']}</div>")
        h.append(f"<div><b>Demand status:</b> {result['demand_status']}</div>")
        h.append(f"<div><b>Inventory status (simulated):</b> {result['inventory_status']}</div>")
        h.append(f"<div><b>Competitor status:</b> {result['competitor_status']}</div>")
        h.append("</div>")
    h.append("</div>") # Close content div
    h.append("</div>") # Close main wrapper div
    
    # Logic to handle Loader -> Shader Landing -> Content
    h.append("<script>")
    h.append("""
    function initShaderHero() {
        const container = document.getElementById('shader-container');
        if (!container) return;
        
        // Safety check for Three.js
        if (typeof THREE === 'undefined') {
            console.error('Three.js not loaded');
            // Fallback: show content immediately if shader fails
            var sl = document.getElementById('shader-landing');
            if(sl) sl.style.display = 'none';
            var c = document.getElementById('content');
            if(c) c.style.display = 'flex';
            return;
        }

        // 1. RENDERER OPTIMIZATION
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ 
            alpha: true, 
            antialias: true, 
            powerPreference: "high-performance" 
        });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Limit pixel ratio
        container.appendChild(renderer.domElement);
        
        const geometry = new THREE.PlaneGeometry(2, 2);
        
        const uniforms = {
            u_resolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
            u_time: { value: 0.0 }
        };
        
        // New Gradient Fluid Shader Logic (Hero Style)
        const material = new THREE.ShaderMaterial({
            uniforms: uniforms,
            vertexShader: `
                void main() {
                    gl_Position = vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                precision mediump float;
                uniform vec2 u_resolution;
                uniform float u_time;

                // Simplex noise function
                vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }

                float snoise(vec2 v){
                    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                            -0.577350269189626, 0.024390243902439);
                    vec2 i  = floor(v + dot(v, C.yy) );
                    vec2 x0 = v -   i + dot(i, C.xx);
                    vec2 i1;
                    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
                    vec4 x12 = x0.xyxy + C.xxzz;
                    x12.xy -= i1;
                    i = mod(i, 289.0);
                    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
                    + i.x + vec3(0.0, i1.x, 1.0 ));
                    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
                    m = m*m ;
                    m = m*m ;
                    vec3 x = 2.0 * fract(p * C.www) - 1.0;
                    vec3 h = abs(x) - 0.5;
                    vec3 ox = floor(x + 0.5);
                    vec3 a0 = x - ox;
                    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
                    vec3 g;
                    g.x  = a0.x  * x0.x  + h.x  * x0.y;
                    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
                    return 130.0 * dot(m, g);
                }

                void main() {
                    vec2 uv = gl_FragCoord.xy / u_resolution.xy;
                    uv.x *= u_resolution.x / u_resolution.y;
                    
                    float t = u_time * 0.1; // Slow motion
                    
                    // Layered noise for fluid effect
                    float n1 = snoise(uv * 3.0 + vec2(t, t * 0.5));
                    float n2 = snoise(uv * 6.0 - vec2(t * 0.5, t));
                    float n3 = snoise(uv * 2.0 + n1 + n2);
                    
                    // Gold and Dark Blue Palette
                    vec3 color1 = vec3(0.05, 0.05, 0.1); // Dark Blue/Black
                    vec3 color2 = vec3(0.8, 0.6, 0.2);   // Gold
                    vec3 color3 = vec3(0.1, 0.1, 0.2);   // Dark Slate
                    
                    float mix1 = smoothstep(-1.0, 1.0, n1);
                    float mix2 = smoothstep(-1.0, 1.0, n3);
                    
                    vec3 finalColor = mix(color1, color3, mix1);
                    finalColor = mix(finalColor, color2, mix2 * 0.4); // Subtle gold highlights
                    
                    // Vignette
                    vec2 center = gl_FragCoord.xy / u_resolution.xy - 0.5;
                    float dist = length(center);
                    finalColor *= 1.0 - dist * 0.5;
                    
                    gl_FragColor = vec4(finalColor, 1.0);
                }
            `
        });
        
        const plane = new THREE.Mesh(geometry, material);
        scene.add(plane);
        
        // 2. ANIMATION SMOOTHNESS
        const clock = new THREE.Clock();
        let animationId;
        
        function animate() {
            animationId = requestAnimationFrame(animate);
            // Delta-time based smooth update with speed reduction
            // 0.2 multiplier slows it down to 20% of real-time speed for cinematic feel
            const delta = Math.min(clock.getDelta(), 0.1); // Clamp delta to prevent jumps
            uniforms.u_time.value += delta * 0.5; // Slightly faster than before but still slow
            renderer.render(scene, camera);
        }
        
        animate();
        
        // 4. RESIZE HANDLING (Debounced)
        let resizeTimeout;
        function onResize() {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                const width = window.innerWidth;
                const height = window.innerHeight;
                renderer.setSize(width, height);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                uniforms.u_resolution.value.set(width, height);
                camera.aspect = width / height;
                camera.updateProjectionMatrix();
            }, 200);
        }
        window.addEventListener('resize', onResize);

        // 3. MEMORY & CLEANUP
        container.cleanup = () => {
            cancelAnimationFrame(animationId);
            window.removeEventListener('resize', onResize);
            renderer.dispose();
            geometry.dispose();
            material.dispose();
            if (container.contains(renderer.domElement)) {
                container.removeChild(renderer.domElement);
            }
        };
    }

    setTimeout(function(){
        var l = document.getElementById('loader');
        var sl = document.getElementById('shader-landing');
        
        // Show landing page behind loader
        if(sl) { 
            sl.style.display = 'flex'; 
        }
        
        // Initialize Shader so it's ready
        initShaderHero();
        
        // Trigger smooth slide out of loader
        if(l) {
            setTimeout(() => {
                l.style.transform = 'translateX(100%)';
            }, 100);
            
            // Remove loader after transition
            setTimeout(() => {
                l.style.display = 'none';
            }, 1000);
        }
        
        // Trigger Animated Text Draw
        setTimeout(() => {
            const path = document.querySelector('.underline-path');
            if(path) path.style.strokeDashoffset = '0';
        }, 500); // Trigger slightly after slide starts
    }, 2000);

    document.getElementById('enterBtn').addEventListener('click', function() {
        var sl = document.getElementById('shader-landing');
        var c = document.getElementById('content');
        
        // Show content immediately (behind landing)
        if(c) { 
            c.style.display = 'flex'; 
            // Trigger resize for layout if needed
            window.dispatchEvent(new Event('resize'));
        }
        
        // Slide landing out to the right
        if(sl) {
            sl.style.transform = 'translateX(100%)';
        }
        
        // NO cleanup of shader (it persists as background)
        // var container = document.getElementById('shader-container');
        // if (container && container.cleanup) {
        //      container.cleanup();
        // }

        setTimeout(() => {
            if(sl) sl.style.display = 'none';
        }, 800); // Wait for transition
    });
    """)
    h.append("</script>")
    
    # Particles Implementation (Vanilla JS adaptation of @react-bits/Particles-JS-CSS)
    h.append("<script>")
    h.append("""
    class Particles {
        constructor(canvasId, options = {}) {
            this.canvas = document.getElementById(canvasId);
            if (!this.canvas) return;
            this.ctx = this.canvas.getContext('2d');
            this.options = {
                particleCount: options.particleCount || 200,
                particleSpread: options.particleSpread || 10,
                speed: options.speed || 0.1,
                particleColors: options.particleColors || ["#ffffff"],
                particleBaseSize: options.particleBaseSize || 100, // Adjusted logic below for 2D
            };
            this.particles = [];
            this.resize();
            this.init();
            window.addEventListener('resize', () => this.resize());
            this.animate();
        }

        resize() {
            this.width = window.innerWidth;
            this.height = window.innerHeight;
            this.canvas.width = this.width;
            this.canvas.height = this.height;
        }

        init() {
            for (let i = 0; i < this.options.particleCount; i++) {
                this.particles.push(this.createParticle());
            }
        }

        createParticle() {
            // Map 3D-like base size to 2D pixel radius. 
            // Assuming 100 -> ~3px, 40 -> ~1.2px (plus randomness)
            const base = this.options.particleBaseSize * 0.05; 
            return {
                x: Math.random() * this.width,
                y: Math.random() * this.height,
                z: Math.random() * 2 + 0.5, 
                size: (Math.random() * base + 1), 
                color: this.options.particleColors[Math.floor(Math.random() * this.options.particleColors.length)],
                vx: (Math.random() - 0.5) * this.options.speed,
                vy: (Math.random() - 0.5) * this.options.speed
            };
        }

        animate() {
            this.ctx.clearRect(0, 0, this.width, this.height);
            
            this.particles.forEach(p => {
                // Update position
                p.x += p.vx * p.z; // Parallax effect: closer particles move faster
                p.y += p.vy * p.z;
                
                // Wrap around screen
                if (p.x < 0) p.x = this.width;
                if (p.x > this.width) p.x = 0;
                if (p.y < 0) p.y = this.height;
                if (p.y > this.height) p.y = 0;
                
                // Draw
                this.ctx.beginPath();
                this.ctx.arc(p.x, p.y, p.size * p.z, 0, Math.PI * 2);
                this.ctx.fillStyle = p.color;
                this.ctx.fill();
            });
            
            requestAnimationFrame(() => this.animate());
        }
    }

    // Initialize Particles
    document.addEventListener('DOMContentLoaded', () => {
        new Particles('particles-canvas', {
            particleCount: 100,
            particleSpread: 20,
            speed: 0.71,
            particleColors: ["#ffffff", "#ffffff", "#ffffff"],
            particleBaseSize: 5
        });
    });
    """)
    h.append("</script>")
    
    h.append("<script>(function(){var f=document.getElementById('fromInput');var t=document.getElementById('toInput');var bF=document.getElementById('btnFrom');var bT=document.getElementById('btnTo');function openPicker(el){if(el&&el.showPicker){el.showPicker()}else if(el){el.focus()}}if(bF){bF.addEventListener('click',function(){openPicker(f)})}if(bT){bT.addEventListener('click',function(){openPicker(t)})}})();</script>")
    
    # Analyze Button Navigation Logic
    h.append("<script>")
    h.append("""
    var btnAnalyze = document.getElementById('btnAnalyze');
    if(btnAnalyze) {
        btnAnalyze.addEventListener('click', function(e) {
            e.preventDefault();
            var f = document.getElementById('fromInput');
            var t = document.getElementById('toInput');
            var fromVal = f ? f.value : '';
            var toVal = t ? t.value : '';
            
            if(!fromVal) {
                // Flash the input or show simple alert if empty
                if(f) { f.style.borderColor = 'red'; setTimeout(()=>f.style.borderColor='', 500); }
                return;
            }
            
            // Programmatic navigation to new page
            window.location.href = '/analysis?from=' + encodeURIComponent(fromVal) + '&to=' + encodeURIComponent(toVal);
        });
    }
    """)
    h.append("</script>")
    
    h.append("</body></html>")
    return "".join(h)

class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/history":
            results = get_all_analysis_results()
            body = render_history_page(results)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            return

        if url.path == "/analysis":
            qs = parse_qs(url.query)
            start_s = qs.get("from", [""])[0]
            end_s = qs.get("to", [""])[0]
            start = parse_date(start_s)
            end = parse_date(end_s) if end_s else dt.date.today()
            
            result = None
            error = None
            if not start:
                error = "Please select a start date"
            elif not end:
                end = dt.date.today()
                
            if not error and start and end:
                 if start > end:
                    error = "Start date must be before end date"
                 else:
                    r = analyze_range(start, end)
                    if "error" in r:
                        error = r["error"]
                    else:
                        result = r
                        # Save result to database
                        try:
                            save_analysis_result(start, end, result)
                            print(f"Analysis result saved for range {start} to {end}")
                        except Exception as e:
                            print(f"Failed to save analysis result: {e}")
            
            body = render_analysis_page(start, end, result, error)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            return
            
        if url.path == "/analyze":
            # Redirect to /analysis for consistency if accessed directly
            qs = parse_qs(url.query)
            start_s = qs.get("from", [""])[0]
            end_s = qs.get("to", [""])[0]
            
            new_url = f"/analysis?from={start_s}&to={end_s}"
            self.send_response(302)
            self.send_header("Location", new_url)
            self.end_headers()
            return

        body = render_html(None, None, None)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

def serve(host="127.0.0.1", port=8000):
    # Initialize database
    try:
        init_db()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Database initialization failed: {e}")

    # Preload dataset in background to make /analyze fast
    threading.Thread(target=load_data_once, daemon=True).start()
    httpd = HTTPServer((host, port), AppHandler)
    print(f"Preview: http://{host}:{port}/")
    httpd.serve_forever()

if __name__ == "__main__":
    serve()
