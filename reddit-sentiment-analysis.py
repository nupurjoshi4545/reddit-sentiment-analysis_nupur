'''*****************************************************************************
Purpose: To analyze the sentiments of the reddit
This program uses Vader SentimentIntensityAnalyzer to calculate the ticker compound value. 
You can change multiple parameters to suit your needs. See below under "set program parameters."
Implementation:
I am using sets for 'x in s' comparison, sets time complexity for "x in s" is O(1) compare to list: O(n).
Limitations:
It depends mainly on the defined parameters for current implementation:
It completely ignores the heavily downvoted comments, and there can be a time when
the most mentioned ticker is heavily downvoted, but you can change that in upvotes variable.
Author: github:asad70
-------------------------------------------------------------------
****************************************************************************'''

import requests
from data import *
from config import NEWSAPI_KEY
import time
import pandas as pd
import matplotlib.pyplot as plt
import squarify
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import emoji    # removes emojis
import re   # removes links
import en_core_web_sm
import string
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import yfinance as yf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

from datetime import datetime

report_filename = f"report_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"
report_file = open(report_filename, 'w', encoding='utf-8')
console = Console(record=True)
file_console = Console(file=report_file, highlight=False, markup=False, width=100)


def fetch_newsapi(tickers_list):
    '''fetch financial news headlines from NewsAPI for given tickers'''
    print("\nFetching NewsAPI data...")
    a_comments = {}
    tickers_found = {}

    for ticker in tickers_list:
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={ticker}&language=en&sortBy=publishedAt&pageSize=20"
            f"&apiKey={NEWSAPI_KEY}"
        )
        try:
            res = requests.get(url)
            articles = res.json().get('articles', [])
            texts = []
            for article in articles:
                title = article.get('title') or ''
                desc = article.get('description') or ''
                combined = f"{title}. {desc}".strip()
                if combined and ticker.upper() in combined.upper():
                    texts.append(combined)
            if texts:
                tickers_found[ticker] = len(texts)
                a_comments[ticker] = texts
        except Exception as e:
            print(f"NewsAPI error for {ticker}: {e}")

    return tickers_found, a_comments



def fetch_posts(sub, headers):
    url = f"https://www.reddit.com/r/{sub}/hot.json?limit=25"
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    return res.json()['data']['children']


def fetch_comments(sub, post_id, headers):
    url = f"https://www.reddit.com/r/{sub}/comments/{post_id}.json?limit=100&sort=new"
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    data = res.json()
    comments = []
    for item in data[1]['data']['children']:
        d = item['data']
        if item['kind'] == 't1':
            comments.append({
                'body': d.get('body', ''),
                'score': d.get('score', 0),
                'author': d.get('author', '')
            })
    return comments


def data_extractor():
    '''extracts all the data from reddit using the public JSON API (no credentials needed)'''

    # set the program parameters
    subs = ['wallstreetbets', 'stocks', 'investing']
    post_flairs = {'Daily Discussion', 'Weekend Discussion', 'Discussion'}
    goodAuth = {'AutoModerator'}
    uniqueCmt = True
    ignoreAuthP = {'example'}
    ignoreAuthC = {'example'}
    upvoteRatio = 0.70
    ups = 20
    upvotes = 2
    picks = 10
    picks_ayz = 10

    headers = {'User-Agent': 'reddit-sentiment-analysis/1.0'}
    posts, count, c_analyzed, tickers, titles, a_comments = 0, 0, 0, {}, [], {}
    cmt_auth = {}

    for sub in subs:
        try:
            submissions = fetch_posts(sub, headers)
        except Exception as e:
            print(f"Failed to fetch posts: {e}")
            continue

        for submission in submissions:
            d = submission['data']
            flair = d.get('link_flair_text')
            author = d.get('author', '')
            upvote_ratio = d.get('upvote_ratio', 0)
            post_ups = d.get('ups', 0)
            post_id = d.get('id')

            if upvote_ratio >= upvoteRatio and post_ups > ups and (flair in post_flairs or flair is None) and author not in ignoreAuthP:
                titles.append(d.get('title', ''))
                posts += 1
                time.sleep(1)   # respect rate limit: 1 req/sec
                try:
                    comments = fetch_comments(sub, post_id, headers)
                    for comment in comments:
                        auth = comment['author']
                        c_analyzed += 1

                        if comment['score'] > upvotes and auth not in ignoreAuthC:
                            split = comment['body'].split(" ")
                            for word in split:
                                word = word.replace("$", "")
                                if word.isupper() and len(word) <= 5 and word not in blacklist and word in us:
                                    if uniqueCmt and auth not in goodAuth:
                                        try:
                                            if auth in cmt_auth[word]: break
                                        except: pass

                                    if word in tickers:
                                        tickers[word] += 1
                                        a_comments[word].append(comment['body'])
                                        cmt_auth[word].append(auth)
                                        count += 1
                                    else:
                                        tickers[word] = 1
                                        cmt_auth[word] = [auth]
                                        a_comments[word] = [comment['body']]
                                        count += 1
                except Exception as e:
                    print(e)

    return posts, c_analyzed, tickers, titles, a_comments, picks, subs, picks_ayz


def print_helper(tickers, picks, c_analyzed, posts, subs, titles, time, start_time):
    '''prints out top tickers, and most mentioned tickers
    
    Parameter:   tickers: dict: all the tickers found
                 picks: int: top picks to analyze
                 c_analyzed: int: # of comments analyzed
                 posts: int: # of posts analyzed
                 subs: int: # of subreddits analyzed
                titles: list: list of the title of posts analyzed 
                 time: time obj: top picks to analyze
                start_time: time obj: prog start time

    Return: symbols: dict: dict of sorted tickers based on mentions
            times: list: include # of time top tickers is mentioned
            top: list: list of top tickers
    '''    

    # sorts the dictionary
    symbols = dict(sorted(tickers.items(), key=lambda item: item[1], reverse = True))
    top_picks = list(symbols.keys())[0:picks]
    time = (time.time() - start_time)
    
    # print top picks
    print("It took {t:.2f} seconds to analyze {c} comments in {p} posts in {s} subreddits.\n".format(t=time, c=c_analyzed, p=posts, s=len(subs)))
    print("Posts analyzed saved in titles")
    #for i in titles: print(i)  # prints the title of the posts analyzed
    
    
    print(f"\n{picks} most mentioned tickers: ")
    times = []
    top = []
    for i in top_picks:
        print(f"{i}: {symbols[i]}")
        times.append(symbols[i])
        top.append(f"{i}: {symbols[i]}")
   
    return symbols, times, top
    
    
def sentiment_analysis(picks_ayz, a_comments, symbols):
    '''analyzes sentiment of top tickers using FinBERT

    Parameter:   picks_ayz: int: top picks to analyze
                 a_comments: dict: all the comments to analyze
                 symbols: dict: dict of sorted tickers based on mentions
    Return:      scores: dictionary: dictionary of all the sentiment analysis
    '''
    scores = {}

    print("\nLoading FinBERT model (first run downloads ~400MB)...")
    finbert = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)

    picks_sentiment = list(symbols.keys())[0:picks_ayz]

    for symbol in picks_sentiment:
        stock_comments = a_comments[symbol]
        score_total = {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
        count = 0

        for cmnt in stock_comments:
            emojiless = emoji.replace_emoji(cmnt, replace='')
            text_clean = re.sub(r'http\S+', '', emojiless)
            text_clean = text_clean.strip()
            if not text_clean:
                continue

            # FinBERT max token length is 512 — truncate long comments
            text_clean = text_clean[:512]

            try:
                result = finbert(text_clean)[0]
                label_scores = {r['label']: r['score'] for r in result}
                pos = label_scores.get('positive', 0.0)
                neg = label_scores.get('negative', 0.0)
                neu = label_scores.get('neutral', 0.0)
                compound = pos - neg  # net sentiment score from -1 to 1

                score_total['pos'] += pos
                score_total['neg'] += neg
                score_total['neu'] += neu
                score_total['compound'] += compound
                count += 1
            except Exception:
                continue

        if count > 0:
            scores[symbol] = {
                'neg': "{:.3f}".format(score_total['neg'] / count),
                'neu': "{:.3f}".format(score_total['neu'] / count),
                'pos': "{:.3f}".format(score_total['pos'] / count),
                'compound': "{:.3f}".format(score_total['compound'] / count),
            }
        else:
            scores[symbol] = {'neg': '0.000', 'neu': '0.000', 'pos': '0.000', 'compound': '0.000'}

    return scores


def visualization(picks_ayz, scores, picks, times, top):
    '''prints sentiment analysis
       makes a most mentioned picks chart
       makes a chart of sentiment analysis of top picks
       
    Parameter:   picks_ayz: int: top picks to analyze
                 scores: dictionary: dictionary of all the sentiment analysis
                 picks: int: most mentioned picks
                times: list: include # of time top tickers is mentioned
                top: list: list of top tickers
    Return:       None
    '''
    
    # printing sentiment analysis 
    print(f"\nSentiment analysis of top {picks_ayz} picks:")
    df = pd.DataFrame(scores)
    df.index = ['Bearish', 'Neutral', 'Bullish', 'Total/Compound']
    df = df.T
    print(df)
    
    # Date Visualization
    # most mentioned picks    
    squarify.plot(sizes=times, label=top, alpha=.7 )
    plt.axis('off')
    plt.title(f"{picks} most mentioned picks")
    #plt.show()
    
    # Sentiment analysis
    df = df.astype(float)
    colors = ['red', 'springgreen', 'forestgreen', 'coral']
    df.plot(kind = 'bar', color=colors, title=f"Sentiment analysis of top {picks_ayz} picks:")
    
    
    #plt.show()

def calculate_intrinsic_value(free_cash_flow, growth_rate, discount_rate, terminal_growth, shares_outstanding):
    '''DCF: projects FCF for 10 years then calculates present value'''
    if not all([free_cash_flow, shares_outstanding]) or shares_outstanding == 0:
        return None
    if growth_rate is None: growth_rate = 0.05
    if discount_rate is None: discount_rate = 0.10
    if terminal_growth is None: terminal_growth = 0.03

    fcf = free_cash_flow
    pv_sum = 0
    for yr in range(1, 11):
        fcf *= (1 + growth_rate)
        pv_sum += fcf / ((1 + discount_rate) ** yr)

    # terminal value
    terminal_value = (fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** 10)

    intrinsic_value_total = pv_sum + pv_terminal
    return intrinsic_value_total / shares_outstanding


def analyze_stock(ticker, sentiment_score):
    '''comprehensive fundamental analysis with buy/hold/sell recommendation'''
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or info.get('quoteType') is None:
            print(f"\n  [{ticker}] No data available — may be delisted or invalid ticker.")
            return

        # ── BASIC INFO ──────────────────────────────────────────────
        name            = info.get('longName', ticker)
        sector          = info.get('sector', 'N/A')
        industry        = info.get('industry', 'N/A')
        business_summary= info.get('longBusinessSummary', '')
        employees       = info.get('fullTimeEmployees')
        country         = info.get('country', 'N/A')
        website         = info.get('website', 'N/A')

        # ── PRICE ───────────────────────────────────────────────────
        current_price   = info.get('currentPrice') or info.get('regularMarketPrice')
        prev_close      = info.get('previousClose')
        week52_high     = info.get('fiftyTwoWeekHigh')
        week52_low      = info.get('fiftyTwoWeekLow')
        ma50            = info.get('fiftyDayAverage')
        ma200           = info.get('twoHundredDayAverage')
        volume          = info.get('volume')
        avg_volume      = info.get('averageVolume')
        beta            = info.get('beta')

        price_change_pct = ((current_price - prev_close) / prev_close * 100) if current_price and prev_close else None
        above_ma50      = current_price > ma50  if current_price and ma50  else False
        above_ma200     = current_price > ma200 if current_price and ma200 else False
        volume_spike    = volume > avg_volume * 1.5 if volume and avg_volume else False

        # position in 52-week range (0% = at low, 100% = at high)
        if week52_high and week52_low and week52_high != week52_low and current_price:
            week52_position = (current_price - week52_low) / (week52_high - week52_low) * 100
        else:
            week52_position = None

        # ── VALUATION ───────────────────────────────────────────────
        market_cap      = info.get('marketCap')
        pe_ratio        = info.get('trailingPE')
        forward_pe      = info.get('forwardPE')
        pb_ratio        = info.get('priceToBook')
        ps_ratio        = info.get('priceToSalesTrailing12Months')
        peg_ratio       = info.get('pegRatio')
        eps             = info.get('trailingEps')
        enterprise_value= info.get('enterpriseValue')
        ev_ebitda       = info.get('enterpriseToEbitda')
        ev_revenue      = info.get('enterpriseToRevenue')

        # ── PROFITABILITY ────────────────────────────────────────────
        profit_margin   = info.get('profitMargins')
        gross_margin    = info.get('grossMargins')
        operating_margin= info.get('operatingMargins')
        roe             = info.get('returnOnEquity')
        roa             = info.get('returnOnAssets')
        revenue_growth  = info.get('revenueGrowth')
        earnings_growth = info.get('earningsGrowth')
        revenue         = info.get('totalRevenue')
        ebitda          = info.get('ebitda')

        # ── BALANCE SHEET ────────────────────────────────────────────
        total_cash      = info.get('totalCash')
        total_debt      = info.get('totalDebt')
        debt_to_equity  = info.get('debtToEquity')
        current_ratio   = info.get('currentRatio')
        quick_ratio     = info.get('quickRatio')
        total_assets    = info.get('totalAssets') if info.get('totalAssets') else None
        book_value      = info.get('bookValue')
        shares_outstanding = info.get('sharesOutstanding')

        # ── FREE CASH FLOW & INTRINSIC VALUE ─────────────────────────
        free_cash_flow  = info.get('freeCashflow')
        operating_cf    = info.get('operatingCashflow')
        growth_rate     = revenue_growth if revenue_growth and revenue_growth > 0 else 0.05
        intrinsic_value = calculate_intrinsic_value(
            free_cash_flow, min(growth_rate, 0.25), 0.10, 0.03, shares_outstanding
        )

        # ── DIVIDENDS ────────────────────────────────────────────────
        dividend_yield  = info.get('dividendYield')
        dividend_rate   = info.get('dividendRate')
        payout_ratio    = info.get('payoutRatio')

        # ── INSTITUTIONAL & INSIDER ──────────────────────────────────
        institutional_hold  = info.get('heldPercentInstitutions')
        insider_hold        = info.get('heldPercentInsiders')
        short_ratio         = info.get('shortRatio')
        short_pct_float     = info.get('shortPercentOfFloat')

        # insider transactions (buying or selling)
        try:
            insider_df = stock.insider_transactions
            if insider_df is not None and not insider_df.empty:
                recent_insider = insider_df.head(5)
                insider_buys  = recent_insider[recent_insider['Transaction'].str.contains('Buy|Purchase', case=False, na=False)]
                insider_sells = recent_insider[recent_insider['Transaction'].str.contains('Sell|Sale', case=False, na=False)]
                net_insider = 'BUYING' if len(insider_buys) > len(insider_sells) else ('SELLING' if len(insider_sells) > len(insider_buys) else 'NEUTRAL')
            else:
                net_insider = 'N/A'
        except Exception:
            net_insider = 'N/A'

        # share buyback — check if shares outstanding decreased YoY
        try:
            shares_hist = stock.get_shares_full(start="2023-01-01")
            if shares_hist is not None and len(shares_hist) >= 2:
                shares_change = (shares_hist.iloc[-1] - shares_hist.iloc[0]) / shares_hist.iloc[0] * 100
                buyback_signal = 'BUYING BACK SHARES' if shares_change < -1 else ('ISSUING SHARES' if shares_change > 1 else 'STABLE')
            else:
                shares_change = None
                buyback_signal = 'N/A'
        except Exception:
            shares_change = None
            buyback_signal = 'N/A'

        # ── COMPETITORS (same sector, top by market cap) ─────────────
        try:
            recommendations = stock.recommendations
            if recommendations is not None and not recommendations.empty:
                analyst_buy    = int(recommendations.iloc[-1].get('strongBuy', 0)) + int(recommendations.iloc[-1].get('buy', 0))
                analyst_hold   = int(recommendations.iloc[-1].get('hold', 0))
                analyst_sell   = int(recommendations.iloc[-1].get('sell', 0)) + int(recommendations.iloc[-1].get('strongSell', 0))
                analyst_summary = f"{analyst_buy} Buy / {analyst_hold} Hold / {analyst_sell} Sell"
            else:
                analyst_summary = 'N/A'
        except Exception:
            analyst_summary = 'N/A'

        # ── SCORING ──────────────────────────────────────────────────
        score = 0
        max_score = 16
        reasons_buy     = []
        reasons_caution = []

        # sentiment (max +2)
        if sentiment_score > 0.1:
            score += 2
            reasons_buy.append(f"Positive Reddit+News sentiment ({sentiment_score:+.3f})")
        elif sentiment_score < -0.1:
            score -= 2
            reasons_caution.append(f"Negative Reddit+News sentiment ({sentiment_score:+.3f})")

        # valuation
        if pe_ratio and 0 < pe_ratio < 20:
            score += 2
            reasons_buy.append(f"Very attractive P/E ratio ({pe_ratio:.1f})")
        elif pe_ratio and pe_ratio < 30:
            score += 1
            reasons_buy.append(f"Reasonable P/E ratio ({pe_ratio:.1f})")
        elif pe_ratio and pe_ratio > 60:
            score -= 1
            reasons_caution.append(f"High P/E ratio ({pe_ratio:.1f}) — possibly overvalued")

        if pb_ratio and pb_ratio < 1:
            score += 2
            reasons_buy.append(f"Trading below book value (P/B: {pb_ratio:.2f}) — undervalued")
        elif pb_ratio and pb_ratio < 3:
            score += 1
            reasons_buy.append(f"Reasonable P/B ratio ({pb_ratio:.2f})")
        elif pb_ratio and pb_ratio > 10:
            score -= 1
            reasons_caution.append(f"Very high P/B ratio ({pb_ratio:.2f}) — expensive relative to assets")

        # intrinsic value vs current price
        if intrinsic_value and current_price:
            margin_of_safety = (intrinsic_value - current_price) / intrinsic_value * 100
            if margin_of_safety > 30:
                score += 2
                reasons_buy.append(f"Trading {margin_of_safety:.1f}% below intrinsic value (DCF: ${intrinsic_value:.2f}) — strong margin of safety")
            elif margin_of_safety > 10:
                score += 1
                reasons_buy.append(f"Trading below intrinsic value (DCF: ${intrinsic_value:.2f}, margin of safety: {margin_of_safety:.1f}%)")
            elif margin_of_safety < -30:
                score -= 1
                reasons_caution.append(f"Trading {abs(margin_of_safety):.1f}% above intrinsic value (DCF: ${intrinsic_value:.2f}) — overvalued")
        else:
            margin_of_safety = None

        # profitability
        if profit_margin and profit_margin > 0.20:
            score += 2
            reasons_buy.append(f"Excellent profit margin ({profit_margin*100:.1f}%)")
        elif profit_margin and profit_margin > 0.10:
            score += 1
            reasons_buy.append(f"Healthy profit margin ({profit_margin*100:.1f}%)")
        elif profit_margin and profit_margin < 0:
            score -= 2
            reasons_caution.append(f"Company is unprofitable (margin: {profit_margin*100:.1f}%)")

        # growth
        if revenue_growth and revenue_growth > 0.20:
            score += 2
            reasons_buy.append(f"Strong revenue growth ({revenue_growth*100:.1f}%)")
        elif revenue_growth and revenue_growth > 0.05:
            score += 1
            reasons_buy.append(f"Moderate revenue growth ({revenue_growth*100:.1f}%)")
        elif revenue_growth and revenue_growth < 0:
            score -= 1
            reasons_caution.append(f"Declining revenue ({revenue_growth*100:.1f}%)")

        # balance sheet
        if current_ratio and current_ratio > 2:
            score += 1
            reasons_buy.append(f"Strong liquidity — current ratio {current_ratio:.2f}")
        elif current_ratio and current_ratio < 1:
            score -= 1
            reasons_caution.append(f"Weak liquidity — current ratio {current_ratio:.2f} (may struggle to pay short-term debt)")

        if debt_to_equity and debt_to_equity < 30:
            score += 1
            reasons_buy.append(f"Very low debt ({debt_to_equity:.1f} D/E)")
        elif debt_to_equity and debt_to_equity > 150:
            score -= 1
            reasons_caution.append(f"High debt load ({debt_to_equity:.1f} D/E) — risky")

        if free_cash_flow and free_cash_flow > 0:
            score += 1
            reasons_buy.append(f"Positive free cash flow (${free_cash_flow/1e9:.2f}B)")
        elif free_cash_flow and free_cash_flow < 0:
            score -= 1
            reasons_caution.append(f"Negative free cash flow (${free_cash_flow/1e9:.2f}B) — burning cash")

        # trend
        if above_ma50:
            score += 1
            reasons_buy.append("Price above 50-day MA (short-term uptrend)")
        else:
            reasons_caution.append("Price below 50-day MA (short-term downtrend)")

        if above_ma200:
            score += 1
            reasons_buy.append("Price above 200-day MA (long-term uptrend)")
        else:
            reasons_caution.append("Price below 200-day MA (long-term downtrend)")

        if volume_spike:
            score += 1
            reasons_buy.append("Volume spike — strong trading momentum")

        # institutional confidence
        if institutional_hold and institutional_hold > 0.70:
            score += 1
            reasons_buy.append(f"High institutional ownership ({institutional_hold*100:.1f}%) — professionals are holding")
        elif institutional_hold and institutional_hold < 0.20:
            reasons_caution.append(f"Low institutional ownership ({institutional_hold*100:.1f}%)")

        # insider activity
        if net_insider == 'BUYING':
            score += 1
            reasons_buy.append("Insiders are buying — management confident in company")
        elif net_insider == 'SELLING':
            score -= 1
            reasons_caution.append("Insiders are selling — management reducing exposure")

        # share buybacks
        if buyback_signal == 'BUYING BACK SHARES':
            score += 1
            reasons_buy.append(f"Company is buying back its own shares — shareholder friendly")
        elif buyback_signal == 'ISSUING SHARES':
            score -= 1
            reasons_caution.append("Company is issuing new shares — diluting existing shareholders")

        # short interest (high short = bearish signal)
        if short_pct_float and short_pct_float > 0.20:
            score -= 1
            reasons_caution.append(f"High short interest ({short_pct_float*100:.1f}%) — many betting against this stock")

        # ── VERDICT ──────────────────────────────────────────────────
        pct = score / max_score * 100
        if pct >= 60:
            verdict = "STRONG BUY [**]"
        elif pct >= 40:
            verdict = "BUY [+]"
        elif pct >= 20:
            verdict = "HOLD [~]"
        elif pct >= 0:
            verdict = "CAUTION [-]"
        else:
            verdict = "AVOID [X]"

        # ── VERDICT COLOR ────────────────────────────────────────────
        verdict_color = {
            "STRONG BUY [**]": "bold green",
            "BUY [+]":         "green",
            "HOLD [~]":        "yellow",
            "CAUTION [-]":     "orange3",
            "AVOID [X]":       "bold red",
        }.get(verdict, "white")

        # ── HEADER PANEL ─────────────────────────────────────────────
        header_text = Text()
        header_text.append(f"{name} ({ticker})\n", style="bold cyan")
        header_text.append(f"{sector} | {industry} | {country}\n", style="dim")
        if business_summary:
            short = business_summary[:200].rsplit(' ', 1)[0] + '...'
            header_text.append(f"\n{short}\n", style="italic")
        if employees:
            header_text.append(f"\nEmployees: {employees:,}", style="dim")
        console.print(Panel(header_text, border_style="cyan", padding=(0,1)))

        def row(label, value, color="white"):
            return (f"[dim]{label}[/dim]", f"[{color}]{value}[/{color}]")

        def pct_color(val):
            return "green" if val > 0 else "red" if val < 0 else "yellow"

        # ── PRICE & TREND TABLE ───────────────────────────────────────
        t1 = Table(title="[bold]Price & Trend[/bold]", box=box.SIMPLE_HEAVY, show_header=False, padding=(0,2))
        t1.add_column("Metric", style="dim", min_width=22)
        t1.add_column("Value", min_width=30)
        if current_price:
            chg_str = f"  ({price_change_pct:+.2f}% today)" if price_change_pct else ""
            chg_color = pct_color(price_change_pct) if price_change_pct else "white"
            t1.add_row("Current Price", f"[bold]${current_price:.2f}[/bold] [{chg_color}]{chg_str}[/{chg_color}]")
        if week52_low and week52_high:
            pos_str = f"  ({week52_position:.0f}% of range)" if week52_position is not None else ""
            t1.add_row("52-Week Range", f"${week52_low:.2f} - ${week52_high:.2f}[dim]{pos_str}[/dim]")
        if beta:
            b_color = "red" if beta > 1.5 else "yellow" if beta > 1 else "green"
            b_label = "high risk" if beta > 1.5 else "moderate" if beta > 1 else "low risk"
            t1.add_row("Beta", f"[{b_color}]{beta:.2f}  ({b_label})[/{b_color}]")
        if ma50:
            ma50_color = "green" if above_ma50 else "red"
            t1.add_row("50-Day MA", f"[{ma50_color}]${ma50:.2f}  ({'ABOVE - uptrend' if above_ma50 else 'BELOW - downtrend'})[/{ma50_color}]")
        if ma200:
            ma200_color = "green" if above_ma200 else "red"
            t1.add_row("200-Day MA", f"[{ma200_color}]${ma200:.2f}  ({'ABOVE - long uptrend' if above_ma200 else 'BELOW - long downtrend'})[/{ma200_color}]")
        if volume and avg_volume:
            v_color = "yellow" if volume_spike else "white"
            spike_str = "  [SPIKE]" if volume_spike else ""
            t1.add_row("Volume", f"[{v_color}]{volume:,}  (avg {avg_volume:,}){spike_str}[/{v_color}]")
        console.print(t1)

        # ── VALUATION TABLE ───────────────────────────────────────────
        t2 = Table(title="[bold]Valuation[/bold]", box=box.SIMPLE_HEAVY, show_header=False, padding=(0,2))
        t2.add_column("Metric", style="dim", min_width=22)
        t2.add_column("Value", min_width=30)
        if market_cap:    t2.add_row("Market Cap",       f"${market_cap/1e9:.2f}B")
        if enterprise_value: t2.add_row("Enterprise Value", f"${enterprise_value/1e9:.2f}B")
        if pe_ratio:      t2.add_row("P/E Ratio",        f"{'[green]' if pe_ratio < 25 else '[red]'}{pe_ratio:.2f}{'[/green]' if pe_ratio < 25 else '[/red]'}")
        if forward_pe:    t2.add_row("Forward P/E",      f"{forward_pe:.2f}")
        if pb_ratio:
            pb_color = "green" if pb_ratio < 1 else "yellow" if pb_ratio < 3 else "red"
            pb_note  = "  (undervalued)" if pb_ratio < 1 else "  (expensive)" if pb_ratio > 10 else ""
            t2.add_row("P/B Ratio", f"[{pb_color}]{pb_ratio:.2f}{pb_note}[/{pb_color}]")
        if ps_ratio:      t2.add_row("P/S Ratio",        f"{ps_ratio:.2f}")
        if peg_ratio:
            peg_color = "green" if peg_ratio < 1 else "yellow" if peg_ratio < 2 else "red"
            peg_note  = "undervalued" if peg_ratio < 1 else "fairly valued" if peg_ratio < 2 else "overvalued"
            t2.add_row("PEG Ratio", f"[{peg_color}]{peg_ratio:.2f}  ({peg_note})[/{peg_color}]")
        if ev_ebitda:     t2.add_row("EV/EBITDA",        f"{ev_ebitda:.2f}")
        if eps:           t2.add_row("EPS",               f"{'[green]' if eps > 0 else '[red]'}${eps:.2f}{'[/green]' if eps > 0 else '[/red]'}")
        console.print(t2)

        # ── INTRINSIC VALUE TABLE ─────────────────────────────────────
        t3 = Table(title="[bold]Intrinsic Value (DCF)[/bold]", box=box.SIMPLE_HEAVY, show_header=False, padding=(0,2))
        t3.add_column("Metric", style="dim", min_width=22)
        t3.add_column("Value", min_width=30)
        if intrinsic_value and current_price:
            iv_color = "green" if margin_of_safety > 0 else "red"
            mos_note = "UNDERVALUED - good entry" if margin_of_safety > 0 else "OVERVALUED - above fair value"
            t3.add_row("Intrinsic Value (DCF)", f"${intrinsic_value:.2f}  (trading: ${current_price:.2f})")
            t3.add_row("Margin of Safety", f"[{iv_color}]{margin_of_safety:.1f}%  {mos_note}[/{iv_color}]")
        else:
            t3.add_row("Intrinsic Value", "[dim]N/A - insufficient data[/dim]")
        console.print(t3)

        # ── PROFITABILITY TABLE ───────────────────────────────────────
        t4 = Table(title="[bold]Profitability[/bold]", box=box.SIMPLE_HEAVY, show_header=False, padding=(0,2))
        t4.add_column("Metric", style="dim", min_width=22)
        t4.add_column("Value", min_width=30)
        if revenue:         t4.add_row("Total Revenue",      f"${revenue/1e9:.2f}B")
        if gross_margin:    t4.add_row("Gross Margin",       f"[green]{gross_margin*100:.1f}%[/green]")
        if operating_margin:
            om_color = "green" if operating_margin > 0 else "red"
            t4.add_row("Operating Margin", f"[{om_color}]{operating_margin*100:.1f}%[/{om_color}]")
        if profit_margin:
            pm_color = "green" if profit_margin > 0.1 else "yellow" if profit_margin > 0 else "red"
            t4.add_row("Net Profit Margin", f"[{pm_color}]{profit_margin*100:.1f}%[/{pm_color}]")
        if ebitda:          t4.add_row("EBITDA",             f"${ebitda/1e9:.2f}B")
        if roe:             t4.add_row("Return on Equity",   f"[{'green' if roe > 0 else 'red'}]{roe*100:.1f}%[/{'green' if roe > 0 else 'red'}]")
        if roa:             t4.add_row("Return on Assets",   f"[{'green' if roa > 0 else 'red'}]{roa*100:.1f}%[/{'green' if roa > 0 else 'red'}]")
        if revenue_growth:  t4.add_row("Revenue Growth",     f"[{pct_color(revenue_growth)}]{revenue_growth*100:.1f}%[/{pct_color(revenue_growth)}]")
        if earnings_growth: t4.add_row("Earnings Growth",    f"[{pct_color(earnings_growth)}]{earnings_growth*100:.1f}%[/{pct_color(earnings_growth)}]")
        console.print(t4)

        # ── BALANCE SHEET TABLE ───────────────────────────────────────
        t5 = Table(title="[bold]Balance Sheet[/bold]", box=box.SIMPLE_HEAVY, show_header=False, padding=(0,2))
        t5.add_column("Metric", style="dim", min_width=22)
        t5.add_column("Value", min_width=30)
        if total_cash:      t5.add_row("Total Cash",         f"[green]${total_cash/1e9:.2f}B[/green]")
        if total_debt:      t5.add_row("Total Debt",         f"[red]${total_debt/1e9:.2f}B[/red]")
        if debt_to_equity:
            de_color = "green" if debt_to_equity < 50 else "yellow" if debt_to_equity < 150 else "red"
            t5.add_row("Debt / Equity", f"[{de_color}]{debt_to_equity:.1f}[/{de_color}]")
        if current_ratio:
            cr_color = "green" if current_ratio > 1.5 else "yellow" if current_ratio > 1 else "red"
            cr_note  = "healthy" if current_ratio > 1.5 else "tight" if current_ratio > 1 else "danger"
            t5.add_row("Current Ratio", f"[{cr_color}]{current_ratio:.2f}  ({cr_note})[/{cr_color}]")
        if quick_ratio:     t5.add_row("Quick Ratio",        f"{quick_ratio:.2f}")
        if book_value:      t5.add_row("Book Value/Share",   f"${book_value:.2f}")
        if free_cash_flow:
            fcf_color = "green" if free_cash_flow > 0 else "red"
            fcf_note  = "positive - healthy" if free_cash_flow > 0 else "negative - burning cash"
            t5.add_row("Free Cash Flow", f"[{fcf_color}]${free_cash_flow/1e9:.2f}B  ({fcf_note})[/{fcf_color}]")
        if operating_cf:    t5.add_row("Operating Cash Flow",f"${operating_cf/1e9:.2f}B")
        console.print(t5)

        # ── DIVIDENDS TABLE ───────────────────────────────────────────
        t6 = Table(title="[bold]Dividends[/bold]", box=box.SIMPLE_HEAVY, show_header=False, padding=(0,2))
        t6.add_column("Metric", style="dim", min_width=22)
        t6.add_column("Value", min_width=30)
        if dividend_yield and dividend_yield > 0:
            t6.add_row("Dividend Yield",  f"[green]{dividend_yield*100:.2f}%[/green]")
            if dividend_rate:  t6.add_row("Dividend Rate",  f"${dividend_rate:.2f}/year")
            if payout_ratio:
                pr_color = "green" if payout_ratio < 0.6 else "red"
                pr_note  = "sustainable" if payout_ratio < 0.6 else "high - watch out"
                t6.add_row("Payout Ratio", f"[{pr_color}]{payout_ratio*100:.1f}%  ({pr_note})[/{pr_color}]")
        else:
            t6.add_row("Dividends", "[dim]None[/dim]")
        console.print(t6)

        # ── OWNERSHIP TABLE ───────────────────────────────────────────
        t7 = Table(title="[bold]Ownership & Activity[/bold]", box=box.SIMPLE_HEAVY, show_header=False, padding=(0,2))
        t7.add_column("Metric", style="dim", min_width=22)
        t7.add_column("Value", min_width=30)
        if institutional_hold:
            ih_color = "green" if institutional_hold > 0.7 else "yellow" if institutional_hold > 0.3 else "red"
            ih_note  = "  (high confidence)" if institutional_hold > 0.7 else "  (low interest)" if institutional_hold < 0.2 else ""
            t7.add_row("Institutional Hold", f"[{ih_color}]{institutional_hold*100:.1f}%{ih_note}[/{ih_color}]")
        if insider_hold:    t7.add_row("Insider Hold",       f"{insider_hold*100:.1f}%")
        ni_color = "green" if net_insider == "BUYING" else "red" if net_insider == "SELLING" else "yellow"
        t7.add_row("Insider Activity",  f"[{ni_color}]{net_insider}[/{ni_color}]")
        bb_color = "green" if buyback_signal == "BUYING BACK SHARES" else "red" if buyback_signal == "ISSUING SHARES" else "yellow"
        t7.add_row("Share Buybacks",    f"[{bb_color}]{buyback_signal}[/{bb_color}]")
        if shares_change is not None:
            sc_color = "green" if shares_change < 0 else "red" if shares_change > 2 else "yellow"
            t7.add_row("Shares Change (YoY)", f"[{sc_color}]{shares_change:+.1f}%[/{sc_color}]")
        if short_pct_float:
            sp_color = "red" if short_pct_float > 0.15 else "yellow" if short_pct_float > 0.05 else "green"
            sp_note  = "  (HIGH - bearish signal)" if short_pct_float > 0.15 else ""
            t7.add_row("Short Interest", f"[{sp_color}]{short_pct_float*100:.1f}%{sp_note}[/{sp_color}]")
        if short_ratio:     t7.add_row("Short Ratio",        f"{short_ratio:.1f} days to cover")
        t7.add_row("Analyst Ratings",   f"[bold]{analyst_summary}[/bold]")
        t7.add_row("Foreign/Promoter",  "[dim]N/A (US stocks)[/dim]")
        console.print(t7)

        # ── MARKET POSITION TABLE ─────────────────────────────────────
        t8 = Table(title="[bold]Market Position[/bold]", box=box.SIMPLE_HEAVY, show_header=False, padding=(0,2))
        t8.add_column("Metric", style="dim", min_width=22)
        t8.add_column("Value", min_width=30)
        t8.add_row("Sector",    sector)
        t8.add_row("Industry",  industry)
        if market_cap:
            if market_cap > 200e9:   mktcap_label = "[bold green]Mega Cap - dominant market leader[/bold green]"
            elif market_cap > 10e9:  mktcap_label = "[green]Large Cap - established company[/green]"
            elif market_cap > 2e9:   mktcap_label = "[yellow]Mid Cap - growth potential[/yellow]"
            else:                    mktcap_label = "[red]Small Cap - higher risk/reward[/red]"
            t8.add_row("Company Size", mktcap_label)
        if beta:
            t8.add_row("Market Sensitivity", f"Beta {beta:.2f} - moves ~{beta:.1f}x the market")
        console.print(t8)

        # ── VERDICT PANEL ─────────────────────────────────────────────
        verdict_text = Text()
        verdict_text.append(f"  Sentiment Score:  {sentiment_score:+.3f}\n", style="white")
        verdict_text.append(f"  Analysis Score:   {score}/{max_score} ({pct:.0f}%)\n\n", style="white")
        verdict_text.append(f"  VERDICT:  {verdict}\n", style=f"bold {verdict_color}")

        if reasons_buy:
            verdict_text.append("\n  Reasons to BUY:\n", style="bold green")
            for r in reasons_buy:
                verdict_text.append(f"    [+] {r}\n", style="green")
        if reasons_caution:
            verdict_text.append("\n  Reasons for CAUTION:\n", style="bold red")
            for r in reasons_caution:
                verdict_text.append(f"    [-] {r}\n", style="red")

        verdict_text.append("\n  [!] This is not financial advice. Always do your own research.", style="dim italic")
        console.print(Panel(verdict_text, title="[bold]Final Verdict[/bold]", border_style=verdict_color, padding=(0,1)))

    except Exception as e:
        console.print(f"\n  [{ticker}] Analysis error: {e}")


def merge_sources(base_tickers, base_comments, extra_tickers, extra_comments):
    '''merge tickers and comments from multiple sources into one'''
    for ticker, count in extra_tickers.items():
        if ticker in base_tickers:
            base_tickers[ticker] += count
            base_comments[ticker].extend(extra_comments[ticker])
        else:
            base_tickers[ticker] = count
            base_comments[ticker] = extra_comments[ticker]
    return base_tickers, base_comments


def main():
    '''main function'''
    start_time = time.time()

    # Reddit
    posts, c_analyzed, tickers, titles, a_comments, picks, subs, picks_ayz = data_extractor()
    symbols, times, top = print_helper(tickers, picks, c_analyzed, posts, subs, titles, time, start_time)

    # get top tickers from Reddit to query other platforms
    top_tickers = list(symbols.keys())[:picks]

    # NewsAPI
    news_tickers, news_comments = fetch_newsapi(top_tickers)
    tickers, a_comments = merge_sources(tickers, a_comments, news_tickers, news_comments)

    # re-sort after merging all sources
    symbols = dict(sorted(tickers.items(), key=lambda item: item[1], reverse=True))

    print("\n--- Combined sources (Reddit + NewsAPI) ---")
    print(f"Top {picks} tickers after merge:")
    for t in list(symbols.keys())[:picks]:
        print(f"  {t}: {symbols[t]}")

    scores = sentiment_analysis(picks_ayz, a_comments, symbols)
    visualization(picks_ayz, scores, picks, times, top)

    # detailed financial analysis for each top ticker
    console.print(Panel("[bold cyan]DETAILED FINANCIAL ANALYSIS[/bold cyan]  (Reddit + NewsAPI + FinBERT + yfinance)", border_style="cyan"))
    for ticker, score_data in scores.items():
        sentiment_score = float(score_data['compound'])
        analyze_stock(ticker, sentiment_score)
    
if __name__ == '__main__':
    main()
    # save full report to text file
    report_file.write(console.export_text())
    report_file.close()
    console.print(f"\n[bold cyan]Full report saved to: {report_filename}[/bold cyan]")
    