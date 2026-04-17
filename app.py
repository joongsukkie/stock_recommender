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

# ---------------------------------------------------------------------------
# Tool definitions that the LLM agent can call
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_stocks_by_industry",
            "description": "Search for top stocks in a given industry/sector. Returns ticker symbols and basic info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "industry": {"type": "string", "description": "The industry or sector to search (e.g. 'Technology', 'Healthcare', 'Energy')"},
                    "count": {"type": "integer", "description": "Number of stocks to return"},
                },
                "required": ["industry", "count"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_data",
            "description": "Get detailed financial data for a specific stock ticker including price, fundamentals, and recent performance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. 'AAPL')"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_risk_profile",
            "description": "Analyze the risk profile of a stock based on volatility, beta, and other risk metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_sentiment",
            "description": "Get recent news headlines and overall sentiment for a stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                },
                "required": ["ticker"],
            },
        },
    },
]

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


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def search_stocks_by_industry(industry: str, count: int) -> dict:
    sector = _match_industry(industry)
    tickers = INDUSTRY_TICKERS.get(sector, INDUSTRY_TICKERS["technology"])[:count]
    results = []
    for t in tickers:
        info = _get_ticker_info(t)
        results.append({
            "ticker": t,
            "name": info.get("shortName", t),
            "sector": info.get("sector", sector),
            "market_cap": info.get("marketCap", "N/A"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice", "N/A"),
        })
    return {"industry": sector, "stocks": results}


def get_stock_data(ticker: str) -> dict:
    info = _get_ticker_info(ticker)
    hist = _get_ticker_history(ticker, "6mo")
    recent_prices = []
    if not hist.empty:
        last_5 = hist.tail(5)
        for date, row in last_5.iterrows():
            recent_prices.append({"date": str(date.date()), "close": round(row["Close"], 2)})

    return {
        "ticker": ticker,
        "name": info.get("shortName", ticker),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice", "N/A"),
        "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
        "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
        "pe_ratio": info.get("trailingPE", "N/A"),
        "forward_pe": info.get("forwardPE", "N/A"),
        "dividend_yield": info.get("dividendYield", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "revenue": info.get("totalRevenue", "N/A"),
        "profit_margin": info.get("profitMargins", "N/A"),
        "debt_to_equity": info.get("debtToEquity", "N/A"),
        "earnings_growth": info.get("earningsGrowth", "N/A"),
        "revenue_growth": info.get("revenueGrowth", "N/A"),
        "recommendation": info.get("recommendationKey", "N/A"),
        "target_price": info.get("targetMeanPrice", "N/A"),
        "recent_prices": recent_prices,
    }


def analyze_risk_profile(ticker: str) -> dict:
    info = _get_ticker_info(ticker)
    hist = _get_ticker_history(ticker, "1y")
    volatility = None
    if not hist.empty and len(hist) > 20:
        returns = hist["Close"].pct_change().dropna()
        volatility = round(float(returns.std() * np.sqrt(252) * 100), 2)

    return {
        "ticker": ticker,
        "beta": info.get("beta", "N/A"),
        "annual_volatility_pct": volatility or "N/A",
        "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
        "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice", "N/A"),
        "debt_to_equity": info.get("debtToEquity", "N/A"),
        "current_ratio": info.get("currentRatio", "N/A"),
        "risk_assessment": (
            "High" if (volatility and volatility > 40) else
            "Medium" if (volatility and volatility > 25) else
            "Low" if volatility else "Unknown"
        ),
    }


def get_news_sentiment(ticker: str) -> dict:
    news = _get_ticker_news(ticker)
    headlines = []
    for item in news[:5]:
        content = item.get("content", {})
        title = content.get("title") if isinstance(content, dict) else item.get("title", "")
        headlines.append(title or "N/A")
    return {
        "ticker": ticker,
        "recent_headlines": headlines,
        "headline_count": len(headlines),
    }


# Map tool names to functions
TOOL_FUNCTIONS = {
    "search_stocks_by_industry": search_stocks_by_industry,
    "get_stock_data": get_stock_data,
    "analyze_risk_profile": analyze_risk_profile,
    "get_news_sentiment": get_news_sentiment,
}


# ---------------------------------------------------------------------------
# Agentic loop — the LLM decides which tools to call and in what order
# ---------------------------------------------------------------------------

def run_agent(user_preferences: dict, api_key: str) -> dict:
    """
    Run the agentic stock recommendation workflow.
    The LLM orchestrates the multi-step research process by deciding
    which tools to call, analyzing results, and producing recommendations.
    """
    client = OpenAI(api_key=api_key)
    system_prompt = """You are a professional stock investment research agent. Your job is to help users find the best stocks to invest in based on their preferences.

You have access to tools that let you:
1. Search for stocks in a specific industry
2. Get detailed financial data for stocks
3. Analyze risk profiles of stocks
4. Get news sentiment for stocks

WORKFLOW — follow these steps:
1. First, search for stocks in the user's preferred industry using search_stocks_by_industry.
2. Then, for the most promising candidates (at least 3-5 stocks), gather detailed data using get_stock_data.
3. Analyze risk profiles for those candidates using analyze_risk_profile.
4. Check news sentiment for the top candidates using get_news_sentiment.
5. Based on ALL gathered data, provide your final recommendations.

When you have finished all research, respond with a JSON object in this EXACT format (no markdown, no extra text):
{
    "recommendations": [
        {
            "ticker": "SYMBOL",
            "name": "Company Name",
            "current_price": 123.45,
            "recommendation_score": 8.5,
            "suggested_allocation_pct": 40,
            "reasoning": "2-3 sentence explanation of why this stock is recommended, referencing specific financial metrics, recent performance, and how it fits the user's risk tolerance and goals.",
            "bull_case": "What could go right — growth catalysts, upcoming products, market tailwinds.",
            "bear_case": "What could go wrong — risks, headwinds, competition.",
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
    "portfolio_summary": "Detailed 2-3 sentence summary of the overall portfolio strategy, explaining the diversification approach and how it matches the user's risk tolerance, investment amount, and goals.",
    "total_investment": 10000,
    "risk_assessment": "Detailed assessment of overall portfolio risk including volatility expectations and what market conditions could help or hurt this portfolio."
}

IMPORTANT:
- recommendation_score is 1-10 (10 = strongest buy)
- suggested_allocation_pct values must sum to 100
- Match the number of recommendations to what the user asked for
- Tailor risk levels to the user's risk tolerance (5 levels: Very Low, Low, Medium, High, Very High)
- For very conservative investors, favor blue-chip dividend aristocrats
- For conservative investors, favor stable dividend-paying stocks with low volatility
- For moderate investors, balance growth and stability
- For aggressive investors, favor growth stocks with higher upside
- For very aggressive investors, favor high-growth, high-momentum stocks
- The reasoning field MUST be detailed — reference specific numbers from the data
- The bull_case and bear_case MUST be specific to each company, not generic
"""

    industries = user_preferences['industry']
    if isinstance(industries, list):
        industry_text = ", ".join(industries)
    else:
        industry_text = industries

    if "any" in industry_text.lower():
        industry_instruction = "Search across multiple industries (technology, healthcare, finance, energy, consumer) to find the best opportunities regardless of sector."
    elif "," in industry_text:
        sectors = [s.strip() for s in industry_text.split(",")]
        industry_instruction = f"Search for stocks across these industries: {', '.join(sectors)}. Use search_stocks_by_industry for each sector."
    else:
        industry_instruction = f"Start by searching for stocks in the {industry_text} industry, then analyze the most promising ones in detail."

    user_message = f"""Please research and recommend stocks based on these preferences:
- Investment Amount: ${user_preferences['investment_amount']:,.2f}
- Risk Tolerance: {user_preferences['risk_level']}
- Preferred Industries: {industry_text}
- Number of Stocks Wanted: {user_preferences['num_stocks']}
- Additional Notes: {user_preferences.get('notes', 'None')}

{industry_instruction}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # Agentic loop — keep going until the LLM produces a final answer
    max_iterations = 15
    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
        )

        msg = response.choices[0].message
        messages.append(msg)

        # If no tool calls, the agent is done
        if not msg.tool_calls:
            break

        # Execute each tool call the agent requested
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            fn = TOOL_FUNCTIONS.get(fn_name)
            if fn:
                result = fn(**fn_args)
            else:
                result = {"error": f"Unknown tool: {fn_name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str),
            })

    # Parse the final response
    final_text = msg.content or ""
    try:
        # Strip markdown code fences if present
        cleaned = final_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw_response": final_text, "recommendations": []}


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
        return jsonify({"error": str(e)}), 500


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
