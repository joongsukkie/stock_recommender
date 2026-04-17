"""
Stock Recommender — Agentic AI System
An agentic workflow that uses OpenAI to orchestrate multi-step stock research,
analysis, and recommendation with projected price forecasting.
"""

import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

# (Tool definitions removed — using pre-fetch + single LLM call for speed)

# ---------------------------------------------------------------------------
# Industry → representative tickers mapping
# ---------------------------------------------------------------------------

INDUSTRY_TICKERS = {
    "technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSM", "AVGO", "CRM", "ORCL", "ADBE", "INTC", "AMD", "CSCO", "IBM"],
    "healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "LLY", "ABT", "BMY", "AMGN", "MDT", "GILD", "ISRG", "CVS", "CI"],
    "finance": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "SCHW", "AXP", "USB", "PNC", "TFC", "BK", "COF", "CME"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "PXD", "HAL", "DVN", "FANG", "HES", "BKR"],
    "consumer": ["PG", "KO", "PEP", "WMT", "COST", "NKE", "MCD", "SBUX", "HD", "LOW", "TGT", "CL", "EL", "GIS", "KMB"],
    "industrials": ["CAT", "HON", "UPS", "BA", "GE", "MMM", "LMT", "RTX", "DE", "UNP", "FDX", "EMR", "ITW", "ETN", "WM"],
    "real estate": ["AMT", "PLD", "CCI", "EQIX", "SPG", "PSA", "O", "WELL", "DLR", "AVB", "EQR", "VTR", "ARE", "MAA", "UDR"],
    "telecommunications": ["T", "VZ", "TMUS", "CMCSA", "CHTR", "LUMN", "FTR", "USM", "SHEN", "ATUS", "LBRDA", "WBD", "PARA", "DISH", "FOX"],
    "materials": ["LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "DOW", "DD", "PPG", "VMC", "MLM", "ALB", "CF", "MOS"],
    "utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC", "ES", "AWK", "DTE", "PPL", "FE"],
}


# ---------------------------------------------------------------------------
# Simple in-memory cache + retry for yfinance (avoids rate limits)
# ---------------------------------------------------------------------------
_cache = {}
_CACHE_TTL = 300  # 5 minutes


def _get_ticker_info(ticker: str) -> dict:
    """Get ticker info with caching and retry logic."""
    cache_key = f"info:{ticker}"
    if cache_key in _cache:
        ts, data = _cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return data
    for attempt in range(3):
        try:
            data = yf.Ticker(ticker).info
            _cache[cache_key] = (time.time(), data)
            return data
        except Exception:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    return {}


def _get_ticker_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Get ticker history with caching and retry logic."""
    cache_key = f"hist:{ticker}:{period}"
    if cache_key in _cache:
        ts, data = _cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return data
    for attempt in range(3):
        try:
            data = yf.Ticker(ticker).history(period=period)
            _cache[cache_key] = (time.time(), data)
            return data
        except Exception:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    return pd.DataFrame()


def _get_ticker_news(ticker: str) -> list:
    """Get ticker news with caching and retry logic."""
    cache_key = f"news:{ticker}"
    if cache_key in _cache:
        ts, data = _cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return data
    for attempt in range(3):
        try:
            data = yf.Ticker(ticker).news or []
            _cache[cache_key] = (time.time(), data)
            return data
        except Exception:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    return []


def _match_industry(query: str) -> str:
    """Fuzzy-match a user's industry query to our known sectors."""
    query_lower = query.lower()
    for key in INDUSTRY_TICKERS:
        if key in query_lower or query_lower in key:
            return key
    # Broader matching
    aliases = {
        "tech": "technology",
        "health": "healthcare",
        "pharma": "healthcare",
        "biotech": "healthcare",
        "bank": "finance",
        "financial": "finance",
        "oil": "energy",
        "gas": "energy",
        "renewable": "energy",
        "solar": "energy",
        "retail": "consumer",
        "food": "consumer",
        "beverage": "consumer",
        "defense": "industrials",
        "aerospace": "industrials",
        "manufacturing": "industrials",
        "telecom": "telecommunications",
        "media": "telecommunications",
        "mining": "materials",
        "chemical": "materials",
        "reit": "real estate",
        "property": "real estate",
        "electric": "utilities",
        "water": "utilities",
        "power": "utilities",
    }
    for alias, sector in aliases.items():
        if alias in query_lower:
            return sector
    return "technology"  # default


# (Individual tool functions removed — data is now pre-fetched in bulk)


# ---------------------------------------------------------------------------
# Pre-fetch + single LLM call approach (fast enough for Render free tier)
# ---------------------------------------------------------------------------

def _prefetch_stock_data(tickers: list) -> list:
    """Fetch data for multiple tickers quickly using concurrent-style batch."""
    results = []
    for t in tickers:
        info = _get_ticker_info(t)
        if not info:
            continue
        hist = _get_ticker_history(t, "1y")
        volatility = None
        if not hist.empty and len(hist) > 20:
            returns = hist["Close"].pct_change().dropna()
            volatility = round(float(returns.std() * np.sqrt(252) * 100), 2)

        news = _get_ticker_news(t)
        headlines = []
        for item in news[:3]:
            content = item.get("content", {})
            title = content.get("title") if isinstance(content, dict) else item.get("title", "")
            if title:
                headlines.append(title)

        results.append({
            "ticker": t,
            "name": info.get("shortName", t),
            "sector": info.get("sector", "N/A"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "forward_pe": info.get("forwardPE", "N/A"),
            "dividend_yield": info.get("dividendYield", "N/A"),
            "beta": info.get("beta", "N/A"),
            "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
            "target_price": info.get("targetMeanPrice", "N/A"),
            "revenue_growth": info.get("revenueGrowth", "N/A"),
            "earnings_growth": info.get("earningsGrowth", "N/A"),
            "profit_margin": info.get("profitMargins", "N/A"),
            "debt_to_equity": info.get("debtToEquity", "N/A"),
            "recommendation": info.get("recommendationKey", "N/A"),
            "annual_volatility_pct": volatility,
            "recent_headlines": headlines,
        })
    return results


def run_agent(user_preferences: dict, api_key: str) -> dict:
    """
    Run the agentic stock recommendation workflow.
    Step 1: Pre-fetch real stock data from yfinance (fast, no LLM needed)
    Step 2: Send all data to LLM in a single call for analysis + recommendation
    This avoids multi-round LLM tool calls that timeout on Render.
    """
    client = OpenAI(api_key=api_key)

    # Determine which tickers to research
    industries_raw = user_preferences['industry']
    if isinstance(industries_raw, list):
        industry_text = ", ".join(industries_raw)
    else:
        industry_text = industries_raw

    # Pick tickers based on industry selection
    if "any" in industry_text.lower():
        # Grab top 2 from each of 5 major sectors
        sectors = ["technology", "healthcare", "finance", "energy", "consumer"]
        tickers = []
        for s in sectors:
            tickers.extend(INDUSTRY_TICKERS[s][:2])
    elif "," in industry_text:
        sectors = [_match_industry(s.strip()) for s in industry_text.split(",")]
        tickers = []
        for s in set(sectors):
            tickers.extend(INDUSTRY_TICKERS.get(s, [])[:5])
    else:
        sector = _match_industry(industry_text)
        tickers = INDUSTRY_TICKERS.get(sector, INDUSTRY_TICKERS["technology"])[:8]

    # Step 1: Pre-fetch all stock data
    stock_data = _prefetch_stock_data(tickers)

    if not stock_data:
        return {"error": "Could not fetch stock data. The data provider may be temporarily unavailable.", "recommendations": []}

    # Step 2: Single LLM call with all data
    system_prompt = """You are a professional stock investment research agent. You will receive real-time stock data and user preferences. Analyze the data and provide investment recommendations.

Respond with ONLY a JSON object (no markdown fences, no extra text) in this EXACT format:
{
    "recommendations": [
        {
            "ticker": "SYMBOL",
            "name": "Company Name",
            "current_price": 123.45,
            "recommendation_score": 8.5,
            "suggested_allocation_pct": 40,
            "reasoning": "2-3 sentence explanation referencing specific financial metrics and how it fits the user's goals.",
            "bull_case": "Specific growth catalysts and upside potential for this company.",
            "bear_case": "Specific risks and what could go wrong for this company.",
            "risk_level": "Very Low/Low/Medium/High/Very High",
            "key_metrics": {
                "pe_ratio": 25.3,
                "forward_pe": 22.1,
                "dividend_yield": 0.015,
                "beta": 1.1,
                "annual_volatility_pct": 28.5,
                "revenue_growth": 0.12,
                "earnings_growth": 0.15,
                "profit_margin": 0.22,
                "debt_to_equity": 45.2,
                "target_price": 150.0,
                "52_week_high": 180.0,
                "52_week_low": 110.0
            }
        }
    ],
    "portfolio_summary": "2-3 sentence summary of the portfolio strategy and diversification approach.",
    "total_investment": 10000,
    "risk_assessment": "Assessment of overall portfolio risk and what market conditions help or hurt it."
}

RULES:
- recommendation_score is 1-10 (10 = strongest buy)
- suggested_allocation_pct values MUST sum to 100
- Match the EXACT number of recommendations the user asked for
- Use 5 risk levels: Very Low, Low, Medium, High, Very High
- For very conservative: blue-chip dividend aristocrats with low volatility
- For conservative: stable dividend payers
- For moderate: balance growth and stability
- For aggressive: growth stocks with higher upside
- For very aggressive: high-growth, high-momentum stocks
- Reasoning MUST reference specific numbers from the provided data
- Bull/bear cases MUST be specific to each company
- Use the actual data provided — do not make up numbers"""

    user_message = f"""USER PREFERENCES:
- Investment Amount: ${user_preferences['investment_amount']:,.2f}
- Risk Tolerance: {user_preferences['risk_level']}
- Preferred Industries: {industry_text}
- Number of Stocks Wanted: {user_preferences['num_stocks']}
- Notes: {user_preferences.get('notes', 'None')}

REAL-TIME STOCK DATA:
{json.dumps(stock_data, indent=2, default=str)}

Analyze this data and select the best {user_preferences['num_stocks']} stocks for this investor. Respond with ONLY the JSON object."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
    )

    final_text = response.choices[0].message.content or ""

    # Parse the response
    try:
        cleaned = final_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        if not cleaned.startswith("{"):
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start != -1 and end > start:
                cleaned = cleaned[start:end]
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"error": "The AI agent returned an unparseable response. Please try again.", "raw_response": final_text, "recommendations": []}


# ---------------------------------------------------------------------------
# Stock projection engine
# ---------------------------------------------------------------------------

def generate_projection(ticker: str, months_ahead: int = 6) -> dict:
    """
    Generate a stock price projection combining:
    - Historical price data (1 year)
    - Moving averages (20, 50, 200 day)
    - Linear trend extrapolation
    - Analyst target price weighting
    - Volatility-based confidence bands
    - Earnings/revenue growth momentum
    """
    info = _get_ticker_info(ticker)
    hist = _get_ticker_history(ticker, "1y")

    if hist.empty:
        return {"error": "No historical data available. The data provider may be temporarily rate-limited — please try again in a moment."}

    # Historical data
    dates = [d.strftime("%Y-%m-%d") for d in hist.index]
    prices = hist["Close"].tolist()

    # Moving averages
    ma_20 = hist["Close"].rolling(window=20).mean()
    ma_50 = hist["Close"].rolling(window=50).mean()
    ma_200 = hist["Close"].rolling(window=200).mean()

    # Current values
    current_price = prices[-1]
    target_price = info.get("targetMeanPrice")
    target_high = info.get("targetHighPrice")
    target_low = info.get("targetLowPrice")
    earnings_growth = info.get("earningsGrowth")  # quarterly
    revenue_growth = info.get("revenueGrowth")  # quarterly
    beta = info.get("beta", 1.0) or 1.0

    # Calculate daily returns and volatility
    returns = hist["Close"].pct_change().dropna()
    daily_vol = float(returns.std())
    annual_vol = daily_vol * np.sqrt(252)

    # ---- Build projection ----
    trading_days_ahead = int(months_ahead * 21)  # ~21 trading days per month
    last_date = hist.index[-1]
    future_dates = []
    d = last_date
    count = 0
    while count < trading_days_ahead:
        d += timedelta(days=1)
        if d.weekday() < 5:  # weekdays only
            future_dates.append(d)
            count += 1

    future_date_strs = [d.strftime("%Y-%m-%d") for d in future_dates]

    # 1. Linear trend from last 60 days
    recent = hist["Close"].values[-60:]
    x = np.arange(len(recent))
    coeffs = np.polyfit(x, recent, 1)
    daily_trend = coeffs[0]

    # 2. Analyst target weighting
    analyst_daily_rate = 0
    if target_price and target_price > 0:
        # Assume analyst target is ~12 months out
        analyst_daily_rate = (target_price - current_price) / 252

    # 3. Growth momentum factor
    growth_factor = 0
    if earnings_growth and isinstance(earnings_growth, (int, float)):
        growth_factor += earnings_growth * 0.3
    if revenue_growth and isinstance(revenue_growth, (int, float)):
        growth_factor += revenue_growth * 0.2
    # Convert quarterly growth to daily contribution
    growth_daily = current_price * growth_factor / 252

    # Weighted projection: combine signals
    # Weights: trend=0.35, analyst=0.40, growth_momentum=0.25
    w_trend, w_analyst, w_growth = 0.35, 0.40, 0.25
    if not target_price or target_price <= 0:
        w_trend, w_analyst, w_growth = 0.55, 0.0, 0.45

    daily_expected = (
        w_trend * daily_trend +
        w_analyst * analyst_daily_rate +
        w_growth * growth_daily
    )

    # Generate projected prices
    projected_prices = []
    upper_band = []
    lower_band = []
    p = current_price
    for i in range(1, trading_days_ahead + 1):
        p_proj = current_price + daily_expected * i
        # Confidence band widens over time (sqrt of time)
        band_width = current_price * annual_vol * np.sqrt(i / 252) * 1.0
        projected_prices.append(round(p_proj, 2))
        upper_band.append(round(p_proj + band_width, 2))
        lower_band.append(round(p_proj - band_width, 2))

    return {
        "ticker": ticker,
        "name": info.get("shortName", ticker),
        "current_price": round(current_price, 2),
        "historical": {
            "dates": dates,
            "prices": [round(p, 2) for p in prices],
            "ma_20": [round(v, 2) if not np.isnan(v) else None for v in ma_20.tolist()],
            "ma_50": [round(v, 2) if not np.isnan(v) else None for v in ma_50.tolist()],
            "ma_200": [round(v, 2) if not np.isnan(v) else None for v in ma_200.tolist()],
        },
        "projection": {
            "dates": future_date_strs,
            "prices": projected_prices,
            "upper_band": upper_band,
            "lower_band": lower_band,
        },
        "metadata": {
            "analyst_target": target_price,
            "analyst_high": target_high,
            "analyst_low": target_low,
            "annual_volatility": round(annual_vol * 100, 2),
            "beta": beta,
            "daily_trend": round(daily_trend, 4),
            "earnings_growth": earnings_growth,
            "revenue_growth": revenue_growth,
            "projection_months": months_ahead,
        },
    }


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.json
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return jsonify({"error": "Please enter your OpenAI API key."}), 400
    try:
        # industry can be a list or comma-separated string from the frontend
        raw_industry = data.get("industry", "technology")
        if isinstance(raw_industry, list):
            industry = ", ".join(raw_industry)
        else:
            industry = raw_industry
        preferences = {
            "investment_amount": float(data.get("investment_amount", 10000)),
            "risk_level": data.get("risk_level", "moderate"),
            "industry": industry,
            "num_stocks": int(data.get("num_stocks", 3)),
            "notes": data.get("notes", ""),
        }
        result = run_agent(preferences, api_key=api_key)
        return jsonify(result)
    except Exception as e:
        error_msg = str(e)
        # Provide friendlier messages for common errors
        if "Incorrect API key" in error_msg or "invalid_api_key" in error_msg:
            error_msg = "Invalid OpenAI API key. Please check your key and try again."
        elif "Rate limit" in error_msg or "rate_limit" in error_msg:
            error_msg = "OpenAI rate limit reached. Please wait a moment and try again."
        elif "insufficient_quota" in error_msg:
            error_msg = "Your OpenAI account has insufficient credits. Please add credits at platform.openai.com."
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            error_msg = "The request timed out. Please try again — the AI agent needs 30-60 seconds to research stocks."
        return jsonify({"error": error_msg}), 500


@app.route("/api/projection/<ticker>")
def projection(ticker):
    months = request.args.get("months", 6, type=int)
    try:
        result = generate_projection(ticker.upper(), months_ahead=months)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_ENV") != "production"
    app.run(debug=debug, host="0.0.0.0", port=port)
