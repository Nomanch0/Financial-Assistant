import os
import json
import re
import math
import secrets
import base64
from io import BytesIO
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from dateutil.relativedelta import relativedelta
from dateutil import parser as date_parser

# ✅ Matplotlib for charts in chat (server-side images)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Flask + Gradio
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import gradio as gr

# ✅ OpenRouter client
from openai import OpenAI

from flask_cors import CORS


# =========================================
# 0) CONFIG
# =========================================

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY environment variable is not set. "
        "Please set it before running (do NOT hardcode tokens in code)."
    )

OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL", "qwen/qwen3-vl-235b-a22b-thinking"
)

or_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

DATA_DIR = os.environ.get("DATA_DIR", os.getcwd())

EMBED_PATH = os.path.join(DATA_DIR, "rag_embeddings (2).npy")
META_PATH = os.path.join(DATA_DIR, "rag_meta (3).jsonl")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index (2).bin")

CATALOG_PATH = os.path.join(DATA_DIR, "saudi_market_data (3) (1).json")


# =========================================
# 1) LOAD RAG RESOURCES
# =========================================

if not (os.path.exists(EMBED_PATH) and os.path.exists(META_PATH) and os.path.exists(FAISS_INDEX_PATH)):
    raise FileNotFoundError(
        "Missing one or more RAG files in DATA_DIR.\n"
        f"- {EMBED_PATH}\n- {META_PATH}\n- {FAISS_INDEX_PATH}"
    )

embeddings = np.load(EMBED_PATH)

meta = []
with open(META_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            meta.append(json.loads(line))

index = faiss.read_index(FAISS_INDEX_PATH)

embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

print("✅ Loaded RAG resources")
print("   - meta rows:", len(meta))
print("   - faiss index size:", index.ntotal)


# =========================================
# 2) LOAD ITEMS CATALOG + BUILD CATALOG INDEX
# =========================================

if not os.path.exists(CATALOG_PATH):
    raise FileNotFoundError(f"Catalog file not found: {CATALOG_PATH}")

with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    items_catalog = json.load(f)

if isinstance(items_catalog, dict):
    for k in ["items", "data", "records", "catalog"]:
        if k in items_catalog and isinstance(items_catalog[k], list):
            items_catalog = items_catalog[k]
            break

print("🧾 Loaded catalog items:", len(items_catalog))


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


MERCHANT_ALIAS = {
    "netflix": ["netflix", "netflix.com", "netflix*"],
    "spotify": ["spotify", "spotify*"],
    "amazon": ["amazon", "amzn", "amazon.com"],
}


def normalize_merchant(name: str) -> str:
    s = normalize_text(name)
    s = re.sub(r"[\*\#\-\_]", " ", s)
    s = re.sub(r"\b(ref|txn|id|no)\b\s*\d+\b", " ", s)
    s = re.sub(r"\b\d{2,}\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for canon, variants in MERCHANT_ALIAS.items():
        if any(v in s for v in variants):
            return canon
    return s if s else "unknown"


def normalize_category(cat: str) -> str:
    c = normalize_text(cat)
    if not c:
        return "unknown"
    if "groc" in c or "supermarket" in c or "hypermarket" in c:
        return "groceries"
    if "restaurant" in c or "dining" in c or "food" in c:
        return "dining"
    if "fuel" in c or "petrol" in c or "diesel" in c:
        return "fuel"
    if "travel" in c or "flight" in c or "hotel" in c:
        return "travel"
    return c


def get_field(t, key, default=None):
    if isinstance(t, dict) and key in t:
        return t[key]

    parts = key.split(".")
    cur = t
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def first_existing_field(obj, key_paths, default=None):
    for kp in key_paths:
        v = get_field(obj, kp, None)
        if v not in (None, "", []):
            return v
    return default


CATALOG_SCHEMA = {
    "name_keys": ["name", "title", "product_name", "item_name", "model"],
    "type_keys": ["type", "category", "item_type", "product_type", "Category", "CategoryName"],
    "brand_keys": ["brand", "make", "manufacturer"],
    "tags_keys": ["tags", "keywords", "features"],
    "price_keys": [
        "cleaned_price",
        "price", "amount", "cost", "value",
        "Amount.Amount", "Price.Amount",
    ],
    "currency_keys": ["currency", "Currency", "Amount.Currency", "Price.Currency"],
}


def catalog_item_to_text(it):
    parts = [
        str(first_existing_field(it, CATALOG_SCHEMA["name_keys"], "")),
        str(first_existing_field(it, CATALOG_SCHEMA["brand_keys"], "")),
        str(first_existing_field(it, CATALOG_SCHEMA["type_keys"], "")),
    ]
    tags = first_existing_field(it, CATALOG_SCHEMA["tags_keys"], [])
    if isinstance(tags, list):
        parts.extend(map(str, tags))
    elif isinstance(tags, str):
        parts.append(tags)
    return " ".join([p for p in parts if p]).strip()


def build_catalog_index(items):
    texts = [catalog_item_to_text(it) or json.dumps(it)[:200] for it in items]
    emb = embed_model.encode(texts, convert_to_numpy=True)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb.astype("float32"))
    return idx


catalog_index = build_catalog_index(items_catalog)
print("🔎 Catalog FAISS index ready:", catalog_index.ntotal)


def clean_buy_query(q: str) -> str:
    q = (q or "").lower()
    q = re.sub(r"\b(can i|could i|may i)\b", " ", q)
    q = re.sub(r"\b(buy|afford|purchase|get|order|take)\b", " ", q)
    q = re.sub(r"\b(a|an|the|for|please|kindly)\b", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


class CatalogLookupTool:
    @staticmethod
    def get_price(it):
        v = first_existing_field(it, CATALOG_SCHEMA["price_keys"], None)
        try:
            p = float(v)
            return p if p > 0 else None
        except Exception:
            return None

    @staticmethod
    def get_currency(it):
        return first_existing_field(it, CATALOG_SCHEMA["currency_keys"], "SAR") or "SAR"

    @staticmethod
    def get_name(it):
        return str(first_existing_field(it, CATALOG_SCHEMA["name_keys"], "Unknown item"))

    @staticmethod
    def get_type(it):
        v = first_existing_field(it, CATALOG_SCHEMA["type_keys"], "") or ""
        return str(v).lower().strip()

    @staticmethod
    def execute(query, top_k=25):
        raw_q = (query or "").strip()
        q = clean_buy_query(raw_q)

        q_emb = embed_model.encode([q], convert_to_numpy=True)
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)

        _, I = catalog_index.search(q_emb.astype("float32"), max(top_k, 25))

        semantic_priced = []
        for idx_ in I[0]:
            if idx_ < 0 or idx_ >= len(items_catalog):
                continue
            it = items_catalog[idx_]
            price = CatalogLookupTool.get_price(it)
            if price is None:
                continue
            semantic_priced.append({
                "name": CatalogLookupTool.get_name(it),
                "price": float(price),
                "currency": CatalogLookupTool.get_currency(it),
                "type": CatalogLookupTool.get_type(it),
            })

        if semantic_priced:
            return {"best": semantic_priced[0], "note": "Picked best semantic match from catalog."}

        return {"best": None, "note": "No usable priced match found in catalog."}


# =========================================
# 3) TRANSACTION HELPERS
# =========================================

def get_amount(t):
    amt = get_field(t, "amount", get_field(t, "Amount.Amount", 0))
    try:
        return float(amt or 0)
    except Exception:
        return 0.0


def get_currency(t):
    return get_field(t, "currency", get_field(t, "Amount.Currency", "SAR")) or "SAR"


def get_date_str(t):
    d = get_field(t, "date", get_field(t, "TransactionDateTime", ""))
    return str(d)[:10] if d else ""


def parse_txn_date(d_str: str):
    if not d_str:
        return None
    d_str = str(d_str).strip()
    try:
        return datetime.strptime(d_str[:10], "%Y-%m-%d")
    except Exception:
        try:
            return date_parser.parse(d_str).replace(tzinfo=None)
        except Exception:
            return None


def extract_price_from_text(text: str):
    cleaned = (text or "").replace(",", " ")
    full_nums = re.findall(r"\b\d+(?:\.\d+)?\b", cleaned)
    if not full_nums:
        return None
    try:
        values = [float(x) for x in full_nums]
        return max(values) if values else None
    except Exception:
        return None


def parse_date_range_from_query(query: str):
    q = (query or "").lower().strip()
    today = datetime.today()
    start = end = None

    m = re.search(r"between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})", q)
    if m:
        try:
            start = datetime.strptime(m.group(1), "%Y-%m-%d")
            end = datetime.strptime(m.group(2), "%Y-%m-%d") + timedelta(days=1)
            return start, end
        except Exception:
            return None, None

    if "this month" in q:
        start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = start + relativedelta(months=1)
        return start, end

    if "this year" in q:
        start = today.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end = start + relativedelta(years=1)
        return start, end

    if "last month" in q:
        first_this_month = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        start = first_this_month - relativedelta(months=1)
        end = first_this_month
        return start, end

    m = re.search(r"last\s+(\d+)\s+months?", q)
    if m:
        n = int(m.group(1))
        end = today + timedelta(days=1)
        start = today - relativedelta(months=n)
        return start, end

    m = re.search(r"last\s+(\d+)\s+days?", q)
    if m:
        n = int(m.group(1))
        end = today + timedelta(days=1)
        start = today - timedelta(days=n)
        return start, end

    if "last week" in q:
        end = today + timedelta(days=1)
        start = today - timedelta(days=7)
        return start, end

    return None, None


def filter_transactions_by_date(transactions, query: str):
    start, end = parse_date_range_from_query(query)
    if not start and not end:
        return transactions, None

    filtered = []
    for t in transactions:
        dt = parse_txn_date(get_date_str(t))
        if not dt:
            continue
        if start and dt < start:
            continue
        if end and dt >= end:
            continue
        filtered.append(t)

    label = None
    if start and end:
        label = f"{start.strftime('%Y-%m-%d')} to {(end - timedelta(days=1)).strftime('%Y-%m-%d')}"
    else:
        label = "filtered"

    return filtered, label


# =========================================
# 4) TOOLS
# =========================================

class TransactionAnalyzer:
    @staticmethod
    def execute(transactions):
        if not transactions:
            return "No transactions available."

        total_spent = 0.0
        categories = defaultdict(float)
        merchants = defaultdict(float)
        cities = defaultdict(float)

        for t in transactions:
            amt = get_amount(t)
            is_debit = "debit" in str(get_field(t, "credit_debit_indicator", "")).lower()
            spend_amt = abs(amt) if is_debit else 0.0
            total_spent += spend_amt

            cat_raw = get_field(t, "category_name", "") or get_field(t, "CategoryName", "") or "Unknown"
            cat = normalize_category(cat_raw)

            merch_raw = get_field(t, "merchant", "") or get_field(t, "MerchantDetails.MerchantName", "") or "Unknown"
            merch = normalize_merchant(merch_raw)

            city_raw = get_field(t, "city", "") or get_field(t, "City", "") or get_field(t, "location", "") or "Unknown"
            city = (normalize_text(city_raw).title() or "Unknown")

            categories[cat] += spend_amt
            merchants[merch] += spend_amt
            cities[city] += spend_amt

        top_category = max(categories.items(), key=lambda x: x[1]) if categories else None
        top_merchant = max(merchants.items(), key=lambda x: x[1]) if merchants else None
        top_city = max(cities.items(), key=lambda x: x[1]) if cities else None

        return {
            "total_spent": total_spent,
            "transaction_count": len(transactions),
            "top_spending_category": top_category,
            "top_spending_merchant": top_merchant,
            "top_spending_city": top_city,
            "all_categories": dict(categories),
            "all_merchants": dict(merchants),
            "all_cities": dict(cities),
        }


class SalaryDetector:
    @staticmethod
    def execute(transactions):
        credits = [t for t in transactions if "credit" in str(get_field(t, "credit_debit_indicator", "")).lower()]
        if not credits:
            return "No credit transactions (income) found."

        amounts = [get_amount(t) for t in credits]
        dates_str = [get_date_str(t) for t in credits]

        avg_credit = float(np.mean(amounts)) if amounts else 0.0
        salary_candidates = []
        for t, amt, d in zip(credits, amounts, dates_str):
            cat = (get_field(t, "category_name", "") or "").lower()
            if (
                amt > avg_credit * 1.2
                or "income" in cat
                or "salary" in cat
                or "freelance" in cat
                or "gig" in cat
            ) and d:
                salary_candidates.append((t, amt, d))

        if not salary_candidates:
            return "No clear repeating income / salary pattern detected."

        salary_amounts = [x[1] for x in salary_candidates]
        salary_dates = []
        for _, _, d in salary_candidates:
            dt = parse_txn_date(d)
            if dt:
                salary_dates.append(dt)

        avg_salary = float(np.mean(salary_amounts)) if salary_amounts else 0.0

        avg_gap_days = None
        predicted_next_date = None
        if len(salary_dates) >= 2:
            salary_dates_sorted = sorted(salary_dates)
            gaps = [
                (salary_dates_sorted[i] - salary_dates_sorted[i - 1]).days
                for i in range(1, len(salary_dates_sorted))
                if (salary_dates_sorted[i] - salary_dates_sorted[i - 1]).days > 0
            ]
            if gaps:
                avg_gap_days = float(np.mean(gaps))
                predicted_next_date = salary_dates_sorted[-1] + timedelta(days=round(avg_gap_days))

        result = {
            "detected_income_avg": avg_salary,
            "income_transaction_count": len(salary_candidates),
            "total_credit_transactions": len(credits),
            "income_dates": [d.strftime("%Y-%m-%d") for d in sorted(salary_dates)],
            "note": "Income pattern inferred from large credits and income-related categories.",
        }
        if avg_gap_days is not None and predicted_next_date is not None:
            result["avg_gap_days"] = avg_gap_days
            result["predicted_next_income_date"] = predicted_next_date.strftime("%Y-%m-%d")
        return result


class BehaviorAnalyzer:
    @staticmethod
    def execute(transactions):
        if not transactions:
            return "No transactions available."

        amounts = [get_amount(t) for t in transactions]
        abs_amounts = [abs(a) for a in amounts] if amounts else []

        weekday_spending = defaultdict(float)
        for t in transactions:
            dt = parse_txn_date(get_date_str(t))
            if dt:
                weekday = dt.strftime("%A")
                is_debit = "debit" in str(get_field(t, "credit_debit_indicator", "")).lower()
                if is_debit:
                    weekday_spending[weekday] += abs(get_amount(t))

        avg_transaction = float(np.mean(abs_amounts)) if abs_amounts else 0.0
        volatility = float(np.std(abs_amounts)) if abs_amounts else 0.0

        return {
            "avg_transaction": avg_transaction,
            "total_transactions": len(transactions),
            "weekday_spending": dict(weekday_spending),
            "spending_volatility": volatility,
        }


class SemanticSearch:
    @staticmethod
    def execute(query, k=5):
        q_emb = embed_model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)

        D, I = index.search(q_emb.astype("float32"), k)
        results = []
        for dist, idx_ in zip(D[0], I[0]):
            if idx_ < 0 or idx_ >= len(meta):
                continue
            results.append(meta[idx_]["metadata"])

        if not results:
            return {"matching_transactions": [], "note": "No reliable semantic matches found."}

        return {"matching_transactions": results, "note": f"Top {len(results)} similar transactions found."}


class DonationAnalyzer:
    @staticmethod
    def execute(transactions):
        donation_txns = []
        for t in transactions:
            donation_amt = get_field(t, "donation_amount", get_field(t, "DonationAmount", 0))
            try:
                donation_amt = float(donation_amt or 0)
            except Exception:
                donation_amt = 0.0

            category = (get_field(t, "category_name", "") or "").lower()
            info = (get_field(t, "transaction_information", "") or "").lower()

            if donation_amt > 0 or "donation" in category or "charity" in category or "donation" in info or "charity" in info:
                donation_txns.append(t)

        if not donation_txns:
            return "No donation transactions found."

        total_donation = 0.0
        donation_dates = []
        for t in donation_txns:
            amt = get_field(t, "donation_amount", get_field(t, "DonationAmount", None))
            if amt is None:
                amt = get_amount(t)
            try:
                amt = float(amt or 0)
            except Exception:
                amt = 0.0

            total_donation += amt
            dt = parse_txn_date(get_date_str(t))
            if dt:
                donation_dates.append(dt)

        donation_dates = sorted(donation_dates)
        avg_donation = total_donation / len(donation_txns) if donation_txns else 0.0

        avg_gap_days = None
        predicted_next_donation = None
        if len(donation_dates) >= 2:
            gaps = [(donation_dates[i] - donation_dates[i - 1]).days for i in range(1, len(donation_dates))]
            gaps = [g for g in gaps if g > 0]
            if gaps:
                avg_gap_days = float(np.mean(gaps))
                predicted_next_donation = donation_dates[-1] + timedelta(days=round(avg_gap_days))

        result = {
            "total_donation": total_donation,
            "donation_count": len(donation_txns),
            "avg_donation": avg_donation,
            "donation_dates": [d.strftime("%Y-%m-%d") for d in donation_dates],
            "note": "Donation transactions inferred from donation_amount or donation/charity keywords.",
        }
        if donation_dates:
            result["last_donation_date"] = donation_dates[-1].strftime("%Y-%m-%d")
        if avg_gap_days is not None and predicted_next_donation is not None:
            result["avg_gap_days"] = avg_gap_days
            result["predicted_next_donation_date"] = predicted_next_donation.strftime("%Y-%m-%d")
        return result


class LastTransactionTool:
    @staticmethod
    def execute(transactions, n_recent=5):
        if not transactions:
            return "No transactions available."

        def parse_dt(t):
            return parse_txn_date(get_date_str(t)) or datetime.min

        tx_sorted = sorted(transactions, key=parse_dt)
        first_tx = tx_sorted[0]
        last_tx = tx_sorted[-1]
        recent_tx = tx_sorted[-n_recent:]

        def tx_summary(t):
            return {
                "transaction_id": get_field(t, "transaction_id"),
                "date": get_date_str(t),
                "amount": get_amount(t),
                "currency": get_currency(t),
                "category_name": get_field(t, "category_name", ""),
                "merchant": get_field(t, "merchant", ""),
                "city": get_field(t, "city", get_field(t, "location", "")),
            }

        return {
            "first_transaction": tx_summary(first_tx),
            "last_transaction": tx_summary(last_tx),
            "recent_transactions": [tx_summary(t) for t in recent_tx],
        }


class CategoryPatternPredictor:
    @staticmethod
    def execute(transactions, query: str):
        if not transactions:
            return "No transactions available."

        q = query.lower()
        label_keywords = {
            "dining_out": ["dining", "dinning", "restaurant", "food", "eat out", "eating out"],
            "groceries": ["grocery", "groceries", "supermarket", "hypermarket"],
            "fuel": ["fuel", "petrol", "gas", "gasoline", "diesel"],
            "entertainment": ["entertainment", "cinema", "movie", "netflix", "spotify", "game"],
            "travel": ["travel", "flight", "hotel", "transport - travel"],
        }

        chosen_label = None
        chosen_keywords = []
        for label, kws in label_keywords.items():
            if any(kw in q for kw in kws):
                chosen_label = label
                chosen_keywords = kws
                break

        if not chosen_label:
            return {"matched_category_label": None, "matched_transaction_count": 0,
                    "note": "No known category keyword detected in the query."}

        matched_txns = []
        for t in transactions:
            cat = (get_field(t, "category_name", "") or "").lower()
            merch = (get_field(t, "merchant", "") or "").lower()
            info = (get_field(t, "transaction_information", "") or "").lower()
            combined = " ".join([cat, merch, info])
            if any(kw in combined for kw in chosen_keywords):
                matched_txns.append(t)

        if not matched_txns:
            return {"matched_category_label": chosen_label, "matched_transaction_count": 0,
                    "note": f"No transactions found for category '{chosen_label}'."}

        dates = []
        amounts = []
        for t in matched_txns:
            dt = parse_txn_date(get_date_str(t))
            if dt:
                dates.append(dt)
            amounts.append(abs(get_amount(t)))

        if not dates:
            return {"matched_category_label": chosen_label, "matched_transaction_count": len(matched_txns),
                    "note": "Transactions found but dates are missing/invalid; cannot analyze pattern."}

        dates_sorted = sorted(dates)
        avg_amount = float(np.mean(amounts)) if amounts else 0.0

        avg_gap_days = None
        predicted_next_date = None
        if len(dates_sorted) >= 2:
            gaps = [(dates_sorted[i] - dates_sorted[i - 1]).days for i in range(1, len(dates_sorted))]
            gaps = [g for g in gaps if g > 0]
            if gaps:
                avg_gap_days = float(np.mean(gaps))
                predicted_next_date = dates_sorted[-1] + timedelta(days=round(avg_gap_days))

        result = {
            "matched_category_label": chosen_label,
            "matched_transaction_count": len(matched_txns),
            "avg_amount": avg_amount,
            "dates": [d.strftime("%Y-%m-%d") for d in dates_sorted],
            "note": f"Pattern for category '{chosen_label}' inferred from category/merchant/info.",
        }
        result["last_category_transaction_date"] = dates_sorted[-1].strftime("%Y-%m-%d")
        if avg_gap_days is not None and predicted_next_date is not None:
            result["avg_gap_days"] = avg_gap_days
            result["predicted_next_category_transaction_date"] = predicted_next_date.strftime("%Y-%m-%d")
        return result


class MonthlySavingsAnalyzer:
    @staticmethod
    def execute(transactions):
        if not transactions:
            return "No transactions available."

        rows = []
        for t in transactions:
            dt = parse_txn_date(get_date_str(t))
            if not dt:
                continue
            amt = get_amount(t)
            is_debit = "debit" in str(get_field(t, "credit_debit_indicator", "")).lower()
            rows.append({"date": dt, "amount": amt, "is_debit": is_debit})

        if not rows:
            return "No valid dated transactions for savings analysis."

        by_month = defaultdict(lambda: {"income": 0.0, "expense": 0.0})
        for r in rows:
            m = r["date"].strftime("%Y-%m")
            if r["is_debit"]:
                by_month[m]["expense"] += abs(r["amount"])
            else:
                by_month[m]["income"] += abs(r["amount"])

        monthly_data = []
        total_income = total_expense = 0.0
        for m, v in sorted(by_month.items()):
            income = v["income"]
            expense = v["expense"]
            saving = income - expense
            monthly_data.append({"month": m, "income": income, "expense": expense, "saving": saving})
            total_income += income
            total_expense += expense

        n_months = len(monthly_data)
        avg_income = total_income / n_months if n_months else 0.0
        avg_expense = total_expense / n_months if n_months else 0.0
        avg_saving = avg_income - avg_expense

        return {
            "months": monthly_data,
            "avg_monthly_income": avg_income,
            "avg_monthly_expense": avg_expense,
            "avg_monthly_saving": avg_saving,
        }


class ShoppingTravelAnalyzer:
    @staticmethod
    def execute(transactions):
        if not transactions:
            return "No transactions available."

        shopping_kws = ["shopping", "retail", "mall", "store", "shop"]
        grocery_kws = ["grocery", "groceries", "supermarket", "hypermarket"]
        travel_kws = ["travel", "flight", "ticket", "hotel", "transport - travel"]
        invest_kws = ["investment", "invest", "portfolio", "stock", "mutual fund"]

        totals = {"shopping_total": 0.0, "grocery_total": 0.0, "travel_total": 0.0, "investment_total": 0.0}

        for t in transactions:
            is_debit = "debit" in str(get_field(t, "credit_debit_indicator", "")).lower()
            if not is_debit:
                continue

            amt = abs(get_amount(t))
            cat = (get_field(t, "category_name", "") or "").lower()
            merch = (get_field(t, "merchant", "") or "").lower()
            info = (get_field(t, "transaction_information", "") or "").lower()
            combined = " ".join([cat, merch, info])

            if any(k in combined for k in shopping_kws):
                totals["shopping_total"] += amt
            if any(k in combined for k in grocery_kws):
                totals["grocery_total"] += amt
            if any(k in combined for k in travel_kws):
                totals["travel_total"] += amt
            if any(k in combined for k in invest_kws):
                totals["investment_total"] += amt

        return totals


class KPIEngine:
    @staticmethod
    def execute(transactions):
        if not transactions:
            return "No transactions available."

        debit = []
        credit = []
        amounts_abs = []

        for t in transactions:
            amt = get_amount(t)
            amounts_abs.append(abs(amt))
            indicator = str(get_field(t, "credit_debit_indicator", "")).lower()
            if "debit" in indicator:
                debit.append(abs(amt))
            elif "credit" in indicator:
                credit.append(abs(amt))

        total_spend = float(np.sum(debit)) if debit else 0.0
        total_income = float(np.sum(credit)) if credit else 0.0
        net_cashflow = total_income - total_spend
        savings_rate = (net_cashflow / total_income) if total_income > 0 else None

        avg_txn = float(np.mean(amounts_abs)) if amounts_abs else 0.0
        volatility = float(np.std(amounts_abs)) if amounts_abs else 0.0

        largest_debit = float(max(debit)) if debit else 0.0
        largest_credit = float(max(credit)) if credit else 0.0

        return {
            "kpi_total_spend": total_spend,
            "kpi_total_income": total_income,
            "kpi_net_cashflow": net_cashflow,
            "kpi_savings_rate": savings_rate,
            "kpi_avg_transaction_abs": avg_txn,
            "kpi_volatility_abs": volatility,
            "kpi_largest_debit": largest_debit,
            "kpi_largest_credit": largest_credit,
        }


class RecurringPaymentDetector:
    @staticmethod
    def execute(transactions):
        rows = []
        for t in transactions:
            indicator = str(get_field(t, "credit_debit_indicator", "")).lower()
            if "debit" not in indicator:
                continue
            merch_raw = get_field(t, "merchant", "") or get_field(t, "MerchantDetails.MerchantName", "")
            merch = normalize_merchant(merch_raw)
            dt = parse_txn_date(get_date_str(t))
            if not dt or merch in ("unknown", ""):
                continue
            rows.append((merch, dt, abs(get_amount(t))))

        if len(rows) < 6:
            return {"recurring_merchants": [], "note": "Not enough usable debit transactions for recurring detection."}

        by_merch = defaultdict(list)
        for merch, dt, amt in rows:
            by_merch[merch].append((dt, amt))

        recurring = []
        for merch, items in by_merch.items():
            if len(items) < 3:
                continue
            items.sort(key=lambda x: x[0])
            dates = [d for d, _ in items]
            amts = [a for _, a in items]

            gaps = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
            gaps = [g for g in gaps if g > 0]
            if not gaps:
                continue

            avg_gap = float(np.mean(gaps))
            if 5 <= avg_gap <= 9:
                interval = "weekly-ish"
                expected_gap = 7
            elif 12 <= avg_gap <= 18:
                interval = "biweekly-ish"
                expected_gap = 14
            elif 25 <= avg_gap <= 35:
                interval = "monthly-ish"
                expected_gap = 30
            else:
                continue

            mu = float(np.mean(amts))
            sigma = float(np.std(amts))
            stable = (mu > 0 and (sigma / mu) <= 0.2)
            if not stable:
                continue

            next_date = dates[-1] + timedelta(days=expected_gap)

            recurring.append({
                "merchant": merch,
                "count": len(items),
                "interval": interval,
                "avg_gap_days": avg_gap,
                "avg_amount": mu,
                "amount_cv": float(sigma / mu) if mu > 0 else None,
                "last_date": dates[-1].strftime("%Y-%m-%d"),
                "predicted_next_date": next_date.strftime("%Y-%m-%d"),
            })

        recurring = sorted(recurring, key=lambda x: (x["count"], -x["avg_amount"]), reverse=True)[:10]
        return {"recurring_merchants": recurring, "note": "Recurring = repeats + interval + stable amounts (CV<=0.2)."}


class AnomalyDetector:
    @staticmethod
    def execute(transactions):
        debits = []
        rows = []
        for t in transactions:
            indicator = str(get_field(t, "credit_debit_indicator", "")).lower()
            if "debit" not in indicator:
                continue
            amt = abs(get_amount(t))
            debits.append(amt)
            rows.append(t)

        if len(debits) < 10:
            return {"anomalies": [], "note": "Not enough debit transactions to detect anomalies reliably."}

        mu = float(np.mean(debits))
        sigma = float(np.std(debits)) or 1.0

        anomalies = []
        for t in rows:
            amt = abs(get_amount(t))
            z = (amt - mu) / sigma
            if z >= 2.5:
                anomalies.append({
                    "date": get_date_str(t),
                    "amount": amt,
                    "currency": get_currency(t),
                    "merchant": normalize_merchant(get_field(t, "merchant", "") or "Unknown"),
                    "category": normalize_category(get_field(t, "category_name", "") or "Unknown"),
                    "z_score": float(z),
                })

        anomalies = sorted(anomalies, key=lambda x: x["z_score"], reverse=True)[:10]
        return {"anomalies": anomalies, "note": "Anomaly = debit much larger than usual (z>=2.5)."}


class BudgetMonitor:
    @staticmethod
    def execute(transactions, lookback_months=6, spike_pct=0.5):
        rows = []
        for t in transactions:
            dt = parse_txn_date(get_date_str(t))
            if not dt:
                continue
            indicator = str(get_field(t, "credit_debit_indicator", "")).lower()
            if "debit" not in indicator:
                continue
            amt = abs(get_amount(t))
            cat = normalize_category(get_field(t, "category_name", "") or get_field(t, "CategoryName", ""))
            rows.append((dt, cat, amt))

        if not rows:
            return {"overspends": [], "note": "No usable debit transactions for budget monitoring."}

        by_month_cat = defaultdict(float)
        months = set()
        for dt, cat, amt in rows:
            m = dt.strftime("%Y-%m")
            months.add(m)
            by_month_cat[(m, cat)] += amt

        months_sorted = sorted(months)
        if len(months_sorted) < 2:
            return {"overspends": [], "note": "Not enough months to compare spending baseline."}

        current_month = months_sorted[-1]
        prev_months = months_sorted[-(lookback_months + 1):-1]

        baseline = defaultdict(list)
        for m in prev_months:
            for (mm, cat), total in by_month_cat.items():
                if mm == m:
                    baseline[cat].append(total)

        baseline_avg = {cat: (sum(v) / len(v)) for cat, v in baseline.items() if v}

        overspends = []
        for (m, cat), total in by_month_cat.items():
            if m != current_month:
                continue
            base = baseline_avg.get(cat)
            if base and base > 0:
                change = (total - base) / base
                if change >= spike_pct:
                    overspends.append({
                        "month": current_month,
                        "category": cat,
                        "current_spend": float(total),
                        "baseline_avg": float(base),
                        "pct_increase": float(change),
                    })

        overspends = sorted(overspends, key=lambda x: x["pct_increase"], reverse=True)[:8]
        return {"current_month": current_month, "overspends": overspends, "note": "Overspend = current month much higher than baseline."}


class TrendAnalyzer:
    @staticmethod
    def execute(transactions, window_days=30):
        debits = []
        for t in transactions:
            dt = parse_txn_date(get_date_str(t))
            if not dt:
                continue
            indicator = str(get_field(t, "credit_debit_indicator", "")).lower()
            if "debit" not in indicator:
                continue
            debits.append((dt, abs(get_amount(t))))

        if len(debits) < 5:
            return {"trend": None, "note": "Not enough debit transactions for trend analysis."}

        debits.sort(key=lambda x: x[0])
        end = debits[-1][0]
        start_last = end - timedelta(days=window_days)
        start_prev = end - timedelta(days=2 * window_days)

        last = sum(a for d, a in debits if start_last <= d <= end)
        prev = sum(a for d, a in debits if start_prev <= d < start_last)

        if prev <= 0:
            return {
                "window_days": window_days,
                "last_window_spend": float(last),
                "prev_window_spend": float(prev),
                "trend": None,
                "note": "Previous window spend is 0; cannot compute % change."
            }

        pct = (last - prev) / prev
        return {
            "window_days": window_days,
            "last_window_spend": float(last),
            "prev_window_spend": float(prev),
            "pct_change": float(pct),
            "trend": "increasing" if pct > 0.05 else ("decreasing" if pct < -0.05 else "stable"),
        }


# =========================================
# 4B) ✅ KPIEngineV2 (unchanged - your original)
# =========================================

def safe_div(a, b, default=None):
    try:
        a = float(a)
        b = float(b)
        if b == 0:
            return default
        return a / b
    except Exception:
        return default

def percentile(values, p):
    try:
        if not values:
            return None
        arr = np.array(values, dtype=float)
        return float(np.percentile(arr, p))
    except Exception:
        return None

def hhi_from_shares(shares):
    try:
        s = [float(x) for x in shares if x is not None]
        if not s:
            return None
        return float(sum(x * x for x in s))
    except Exception:
        return None

def month_key(dt):
    return dt.strftime("%Y-%m")

def category_bucket(cat_name: str):
    c = (cat_name or "").lower()

    fixed_kws = [
        "rent", "utilities", "mortgage", "insurance", "bill", "electric", "water", "gas",
        "internet", "phone", "school", "tuition"
    ]
    discretionary_kws = [
        "shopping", "online shopping", "recreation", "subscriptions", "entertainment",
        "restaurant", "dining", "food", "travel"
    ]

    if any(k in c for k in fixed_kws):
        return "fixed"
    if any(k in c for k in discretionary_kws):
        return "discretionary"
    return "other"

def is_debit_txn(t):
    return "debit" in str(get_field(t, "credit_debit_indicator", "")).lower()

def is_credit_txn(t):
    return "credit" in str(get_field(t, "credit_debit_indicator", "")).lower()

def txn_category(t):
    return get_field(t, "category_name", "") or get_field(t, "CategoryName", "") or "Unknown"

def txn_merchant(t):
    return normalize_merchant(get_field(t, "merchant", "") or get_field(t, "MerchantDetails.MerchantName", "") or "Unknown")

def txn_city(t):
    raw = get_field(t, "city", "") or get_field(t, "City", "") or get_field(t, "location", "") or "Unknown"
    return (normalize_text(raw).title() or "Unknown")

def txn_location_type(t):
    return get_field(t, "location_type", "") or "Unknown"

def txn_time_of_day(t):
    return get_field(t, "time_of_day", "") or "Unknown"

def txn_is_rush_hour(t):
    return bool(get_field(t, "is_rush_hour", False))

def txn_is_travel_related(t):
    return bool(get_field(t, "is_travel_related", False))

def txn_instrument(t):
    return get_field(t, "instrument_type", "") or "Unknown"

def txn_card_scheme(t):
    return get_field(t, "card_scheme", "") or "Unknown"


class KPIEngineV2:
    @staticmethod
    def execute(transactions):
        if not transactions:
            return "No transactions available."

        dated = []
        for t in transactions:
            dt = parse_txn_date(get_date_str(t))
            if dt:
                dated.append((dt, t))
        if not dated:
            return "No valid dated transactions found."

        dated.sort(key=lambda x: x[0])
        start_dt = dated[0][0]
        end_dt = dated[-1][0]
        days_span = max(1, (end_dt - start_dt).days + 1)

        debit_amounts = []
        credit_amounts = []
        all_abs = []

        spend_by_category = defaultdict(float)
        spend_by_merchant = defaultdict(float)
        spend_by_city = defaultdict(float)
        spend_by_location_type = defaultdict(float)
        spend_by_time_of_day = defaultdict(float)
        spend_by_instrument = defaultdict(float)
        spend_by_card = defaultdict(float)

        online_spend = 0.0
        offline_spend = 0.0
        rush_spend = 0.0
        night_spend = 0.0
        travel_spend = 0.0

        fixed_spend = 0.0
        discretionary_spend = 0.0
        other_spend = 0.0

        tx_count_by_month = defaultdict(int)
        debit_count_by_month = defaultdict(int)
        credit_count_by_month = defaultdict(int)
        debit_spend_by_month = defaultdict(float)
        credit_income_by_month = defaultdict(float)
        salary_income_by_month = defaultdict(float)

        salary_amounts = []
        salary_dates = []

        merchants_seen = set()
        repeat_merchant_txn_count = 0
        unique_merchants = set()

        debit_count_by_instrument = defaultdict(int)

        for dt, t in dated:
            amt = get_amount(t)
            amt_abs = abs(amt)
            all_abs.append(amt_abs)

            mkey = month_key(dt)
            tx_count_by_month[mkey] += 1

            cat = txn_category(t)
            merch = txn_merchant(t)
            city = txn_city(t)
            loc_type = txn_location_type(t)
            tod = txn_time_of_day(t)
            instr = txn_instrument(t)
            card = txn_card_scheme(t)

            unique_merchants.add(merch)
            if merch in merchants_seen:
                repeat_merchant_txn_count += 1
            else:
                merchants_seen.add(merch)

            if is_debit_txn(t):
                debit_amounts.append(amt_abs)
                debit_count_by_month[mkey] += 1
                debit_spend_by_month[mkey] += amt_abs

                spend_by_category[cat] += amt_abs
                spend_by_merchant[merch] += amt_abs
                spend_by_city[city] += amt_abs
                spend_by_location_type[loc_type] += amt_abs
                spend_by_time_of_day[tod] += amt_abs
                spend_by_instrument[instr] += amt_abs
                spend_by_card[card] += amt_abs

                debit_count_by_instrument[instr] += 1

                if str(loc_type).lower() == "online":
                    online_spend += amt_abs
                else:
                    offline_spend += amt_abs

                if txn_is_rush_hour(t):
                    rush_spend += amt_abs
                if str(tod).lower() == "night":
                    night_spend += amt_abs
                if txn_is_travel_related(t):
                    travel_spend += amt_abs

                bucket = category_bucket(cat)
                if bucket == "fixed":
                    fixed_spend += amt_abs
                elif bucket == "discretionary":
                    discretionary_spend += amt_abs
                else:
                    other_spend += amt_abs

            elif is_credit_txn(t):
                credit_amounts.append(amt_abs)
                credit_count_by_month[mkey] += 1
                credit_income_by_month[mkey] += amt_abs

                if "primary salary" in str(cat).lower():
                    salary_amounts.append(amt_abs)
                    salary_dates.append(dt)
                    salary_income_by_month[mkey] += amt_abs

        total_spend = float(np.sum(debit_amounts)) if debit_amounts else 0.0
        total_income = float(np.sum(credit_amounts)) if credit_amounts else 0.0
        net_cashflow = total_income - total_spend
        savings_rate = safe_div(net_cashflow, total_income, default=None)

        avg_txn_abs = float(np.mean(all_abs)) if all_abs else 0.0
        vol_abs = float(np.std(all_abs)) if all_abs else 0.0
        largest_debit = float(max(debit_amounts)) if debit_amounts else 0.0
        largest_credit = float(max(credit_amounts)) if credit_amounts else 0.0

        tx_per_month = float(np.mean(list(tx_count_by_month.values()))) if tx_count_by_month else 0.0
        debits_per_month = float(np.mean(list(debit_count_by_month.values()))) if debit_count_by_month else 0.0
        credits_per_month = float(np.mean(list(credit_count_by_month.values()))) if credit_count_by_month else 0.0
        avg_daily_spend = safe_div(total_spend, days_span, default=0.0)

        median_debit = float(np.median(debit_amounts)) if debit_amounts else 0.0
        p90_debit = percentile(debit_amounts, 90) or 0.0
        p95_debit = percentile(debit_amounts, 95) or 0.0

        small_spend_ratio = safe_div(sum(1 for x in debit_amounts if x < 50), len(debit_amounts), default=0.0) if debit_amounts else 0.0
        large_spend_ratio = safe_div(sum(1 for x in debit_amounts if x > 500), len(debit_amounts), default=0.0) if debit_amounts else 0.0

        top_category = None
        top_category_share = None
        cat_hhi = None
        if spend_by_category and total_spend > 0:
            top_category = max(spend_by_category.items(), key=lambda x: x[1])
            top_category_share = float(top_category[1] / total_spend)
            shares = [v / total_spend for v in spend_by_category.values()]
            cat_hhi = hhi_from_shares(shares)

        top_merchant = None
        top_merchant_share = None
        merch_hhi = None
        if spend_by_merchant and total_spend > 0:
            top_merchant = max(spend_by_merchant.items(), key=lambda x: x[1])
            top_merchant_share = float(top_merchant[1] / total_spend)
            shares = [v / total_spend for v in spend_by_merchant.values()]
            merch_hhi = hhi_from_shares(shares)

        unique_merchants_count = len(unique_merchants)
        repeat_merchant_rate = safe_div(repeat_merchant_txn_count, len(dated), default=0.0)

        online_share = safe_div(online_spend, total_spend, default=0.0) if total_spend > 0 else 0.0
        travel_share = safe_div(travel_spend, total_spend, default=0.0) if total_spend > 0 else 0.0
        rush_share = safe_div(rush_spend, total_spend, default=0.0) if total_spend > 0 else 0.0
        night_share = safe_div(night_spend, total_spend, default=0.0) if total_spend > 0 else 0.0

        fixed_share = safe_div(fixed_spend, total_spend, default=0.0) if total_spend > 0 else 0.0
        discretionary_share = safe_div(discretionary_spend, total_spend, default=0.0) if total_spend > 0 else 0.0

        subs_spend = 0.0
        online_shop_spend = 0.0
        for cat, v in spend_by_category.items():
            cl = str(cat).lower()
            if "subscription" in cl:
                subs_spend += v
            if "online shopping" in cl:
                online_shop_spend += v

        subscription_burden_income = safe_div(subs_spend, total_income, default=None) if total_income > 0 else None
        subscription_share_spend = safe_div(subs_spend, total_spend, default=0.0) if total_spend > 0 else 0.0
        online_shopping_share = safe_div(online_shop_spend, total_spend, default=0.0) if total_spend > 0 else 0.0

        digital_wallet_spend = 0.0
        for instr, v in spend_by_instrument.items():
            if "googlepay" in str(instr).lower() or "applepay" in str(instr).lower():
                digital_wallet_spend += v
        digital_wallet_share = safe_div(digital_wallet_spend, total_spend, default=0.0) if total_spend > 0 else 0.0

        avg_debit_by_instrument = {
            k: safe_div(spend_by_instrument.get(k, 0.0), debit_count_by_instrument.get(k, 0), default=0.0)
            for k in spend_by_instrument.keys()
        }

        salary_cv = None
        income_gap_avg_days = None
        income_gap_std_days = None
        if salary_amounts:
            mu = float(np.mean(salary_amounts))
            sigma = float(np.std(salary_amounts))
            salary_cv = safe_div(sigma, mu, default=None) if mu > 0 else None

        if len(salary_dates) >= 2:
            sd = sorted(salary_dates)
            gaps = [(sd[i] - sd[i - 1]).days for i in range(1, len(sd)) if (sd[i] - sd[i - 1]).days > 0]
            if gaps:
                income_gap_avg_days = float(np.mean(gaps))
                income_gap_std_days = float(np.std(gaps))

        months_sorted = sorted(tx_count_by_month.keys())
        monthly_series = []
        for m in months_sorted:
            monthly_series.append({
                "month": m,
                "tx_count": int(tx_count_by_month[m]),
                "debit_count": int(debit_count_by_month.get(m, 0)),
                "credit_count": int(credit_count_by_month.get(m, 0)),
                "spend": float(debit_spend_by_month.get(m, 0.0)),
                "income": float(credit_income_by_month.get(m, 0.0)),
                "salary_income": float(salary_income_by_month.get(m, 0.0)),
                "net": float(credit_income_by_month.get(m, 0.0) - debit_spend_by_month.get(m, 0.0)),
            })

        return {
            "kpi_total_spend": total_spend,
            "kpi_total_income": total_income,
            "kpi_net_cashflow": net_cashflow,
            "kpi_savings_rate": savings_rate,
            "kpi_avg_transaction_abs": avg_txn_abs,
            "kpi_volatility_abs": vol_abs,
            "kpi_largest_debit": largest_debit,
            "kpi_largest_credit": largest_credit,

            "kpi_days_span": int(days_span),
            "kpi_tx_per_month_avg": tx_per_month,
            "kpi_debits_per_month_avg": debits_per_month,
            "kpi_credits_per_month_avg": credits_per_month,
            "kpi_avg_daily_spend": float(avg_daily_spend),
            "kpi_median_debit": float(median_debit),
            "kpi_debit_p90": float(p90_debit),
            "kpi_debit_p95": float(p95_debit),
            "kpi_small_spend_ratio_lt_50": float(small_spend_ratio),
            "kpi_large_spend_ratio_gt_500": float(large_spend_ratio),

            "kpi_top_category": top_category[0] if top_category else None,
            "kpi_top_category_spend": float(top_category[1]) if top_category else None,
            "kpi_top_category_share": float(top_category_share) if top_category_share is not None else None,
            "kpi_category_hhi": float(cat_hhi) if cat_hhi is not None else None,
            "kpi_fixed_spend_share": float(fixed_share),
            "kpi_discretionary_spend_share": float(discretionary_share),
            "kpi_subscription_burden_income": float(subscription_burden_income) if subscription_burden_income is not None else None,
            "kpi_subscription_share_spend": float(subscription_share_spend),
            "kpi_online_shopping_share": float(online_shopping_share),

            "kpi_top_merchant": top_merchant[0] if top_merchant else None,
            "kpi_top_merchant_spend": float(top_merchant[1]) if top_merchant else None,
            "kpi_top_merchant_share": float(top_merchant_share) if top_merchant_share is not None else None,
            "kpi_merchant_hhi": float(merch_hhi) if merch_hhi is not None else None,
            "kpi_unique_merchants_count": int(unique_merchants_count),
            "kpi_repeat_merchant_rate": float(repeat_merchant_rate),

            "kpi_online_spend_share": float(online_share),
            "kpi_travel_spend_share": float(travel_share),
            "kpi_rush_hour_spend_share": float(rush_share),
            "kpi_night_spend_share": float(night_share),

            "kpi_digital_wallet_spend_share": float(digital_wallet_share),
            "kpi_avg_debit_by_instrument": avg_debit_by_instrument,

            "kpi_salary_amount_cv": float(salary_cv) if salary_cv is not None else None,
            "kpi_salary_gap_avg_days": float(income_gap_avg_days) if income_gap_avg_days is not None else None,
            "kpi_salary_gap_std_days": float(income_gap_std_days) if income_gap_std_days is not None else None,

            "breakdown_spend_by_category": dict(spend_by_category),
            "breakdown_spend_by_merchant": dict(spend_by_merchant),
            "breakdown_spend_by_city": dict(spend_by_city),
            "breakdown_spend_by_location_type": dict(spend_by_location_type),
            "breakdown_spend_by_time_of_day": dict(spend_by_time_of_day),
            "breakdown_spend_by_instrument": dict(spend_by_instrument),
            "breakdown_spend_by_card_scheme": dict(spend_by_card),
            "monthly_series": monthly_series,
        }


# =========================================
# ✅ FIX COOKIE OVERFLOW + CHARTS IN CHAT
# =========================================

CHAT_STORE = {}  # key: (sid, account_id) -> list of dict turns


def get_or_create_sid():
    sid = session.get("sid")
    if not sid:
        sid = secrets.token_hex(16)
        session["sid"] = sid
    return sid


def get_history_for_account_mem(account_id: str, sid: str, limit: int = 30):
    return CHAT_STORE.get((sid, account_id), [])[-limit:]


def append_history_for_account_mem(account_id: str, sid: str, user_msg: str, bot_msg: str, charts=None):
    charts = charts or []
    CHAT_STORE.setdefault((sid, account_id), [])
    CHAT_STORE[(sid, account_id)].append({"user": user_msg, "bot": bot_msg, "charts": charts})
    CHAT_STORE[(sid, account_id)] = CHAT_STORE[(sid, account_id)][-60:]


def clear_history_for_account_mem(account_id: str, sid: str):
    CHAT_STORE[(sid, account_id)] = []


def fig_to_b64(fig):
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def wants_charts(q: str) -> bool:
    q = (q or "").lower()
    keys = [
        "chart", "charts", "graph", "plot",
        "bar", "pie", "doughnut", "line",
        "hist", "histogram", "stack", "radar", "spider",
        "dashboard"
    ]
    return any(k in q for k in keys)


def build_chat_charts(transactions, query: str):
    if not transactions:
        return []

    q = (query or "").lower()
    currency = get_currency(transactions[0]) if transactions else "SAR"

    ta = TransactionAnalyzer.execute(transactions)
    ms = MonthlySavingsAnalyzer.execute(transactions)
    kpi = KPIEngineV2.execute(transactions)

    charts = []

    # Pie/Doughnut: Income vs Spend
    if ("income" in q and ("spend" in q or "expense" in q)) or ("pie" in q) or ("doughnut" in q):
        if isinstance(kpi, dict):
            fig = plt.figure()
            values = [kpi.get("kpi_total_income", 0.0), kpi.get("kpi_total_spend", 0.0)]
            plt.pie(values, labels=["Income", "Spending"], autopct="%1.1f%%")
            plt.title(f"Income vs Spending ({currency})")
            charts.append({"title": "Income vs Spending (Pie)", "image_b64": fig_to_b64(fig)})

    # Bar: Top Categories
    if ("category" in q and ("bar" in q or "top" in q or "chart" in q)) or ("top categories" in q):
        if isinstance(ta, dict):
            cats = ta.get("all_categories", {}) or {}
            ranked = sorted(cats.items(), key=lambda x: x[1], reverse=True)[:10]
            if ranked:
                labels = [x[0] for x in ranked][::-1]
                vals = [x[1] for x in ranked][::-1]
                fig = plt.figure()
                plt.barh(labels, vals)
                plt.title(f"Top Categories by Spend ({currency})")
                plt.xlabel(currency)
                charts.append({"title": "Top Categories (Bar)", "image_b64": fig_to_b64(fig)})

    # Bar: Top Merchants
    if ("merchant" in q and ("bar" in q or "top" in q or "chart" in q)) or ("top merchants" in q):
        if isinstance(ta, dict):
            merch = ta.get("all_merchants", {}) or {}
            ranked = sorted(merch.items(), key=lambda x: x[1], reverse=True)[:10]
            if ranked:
                labels = [x[0] for x in ranked][::-1]
                vals = [x[1] for x in ranked][::-1]
                fig = plt.figure()
                plt.barh(labels, vals)
                plt.title(f"Top Merchants by Spend ({currency})")
                plt.xlabel(currency)
                charts.append({"title": "Top Merchants (Bar)", "image_b64": fig_to_b64(fig)})

    # Line: Monthly Income vs Expense
    if ("monthly" in q and ("line" in q or "trend" in q or "chart" in q)) or ("monthly income" in q):
        if isinstance(ms, dict):
            months = ms.get("months", []) or []
            if months:
                x = [m["month"] for m in months]
                inc = [m["income"] for m in months]
                exp = [m["expense"] for m in months]
                fig = plt.figure()
                plt.plot(x, inc, label="Income")
                plt.plot(x, exp, label="Expense")
                plt.xticks(rotation=45, ha="right")
                plt.legend()
                plt.title(f"Monthly Income vs Expense ({currency})")
                plt.ylabel(currency)
                charts.append({"title": "Monthly Income vs Expense (Line)", "image_b64": fig_to_b64(fig)})

    # Histogram: Amount distribution
    if ("hist" in q) or ("distribution" in q):
        amts = [abs(get_amount(t)) for t in transactions if get_amount(t)]
        if len(amts) >= 5:
            fig = plt.figure()
            plt.hist(amts, bins=10)
            plt.title(f"Transaction Amount Distribution ({currency})")
            plt.xlabel(currency)
            plt.ylabel("Count")
            charts.append({"title": "Amount Distribution (Histogram)", "image_b64": fig_to_b64(fig)})

    # "show charts" default pack
    if ("show charts" in q) or ("dashboard" in q) or ("all charts" in q):
        if not charts:
            try:
                if isinstance(kpi, dict):
                    fig = plt.figure()
                    values = [kpi.get("kpi_total_income", 0.0), kpi.get("kpi_total_spend", 0.0)]
                    plt.pie(values, labels=["Income", "Spending"], autopct="%1.1f%%")
                    plt.title(f"Income vs Spending ({currency})")
                    charts.append({"title": "Income vs Spending (Pie)", "image_b64": fig_to_b64(fig)})
            except Exception:
                pass

            try:
                if isinstance(ta, dict):
                    cats = ta.get("all_categories", {}) or {}
                    ranked = sorted(cats.items(), key=lambda x: x[1], reverse=True)[:8]
                    if ranked:
                        labels = [x[0] for x in ranked][::-1]
                        vals = [x[1] for x in ranked][::-1]
                        fig = plt.figure()
                        plt.barh(labels, vals)
                        plt.title(f"Top Categories ({currency})")
                        plt.xlabel(currency)
                        charts.append({"title": "Top Categories (Bar)", "image_b64": fig_to_b64(fig)})
            except Exception:
                pass

    return charts


# =========================================
# 5) AGENT
# =========================================

class AgenticRAG:
    def __init__(self):
        self.tools = {
            "transaction_analyzer": TransactionAnalyzer(),
            "salary_detector": SalaryDetector(),
            "behavior_analyzer": BehaviorAnalyzer(),
            "semantic_search": SemanticSearch(),
            "donation_analyzer": DonationAnalyzer(),
            "last_transaction_tool": LastTransactionTool(),
            "category_pattern_predictor": CategoryPatternPredictor(),
            "monthly_savings_analyzer": MonthlySavingsAnalyzer(),
            "shopping_travel_analyzer": ShoppingTravelAnalyzer(),
            "kpi_engine": KPIEngine(),
            "recurring_payment_detector": RecurringPaymentDetector(),
            "anomaly_detector": AnomalyDetector(),
            "budget_monitor": BudgetMonitor(),
            "trend_analyzer": TrendAnalyzer(),
        }

    def classify_intent(self, query: str):
        q = query.lower().strip()

        if "can i buy" in q or "can i afford" in q:
            return "afford_purchase"
        if "next salary" in q or "next income" in q or "when will i get salary" in q or "when is my next salary" in q:
            return "next_salary"

        if "kpi" in q or "metrics" in q or "dashboard" in q:
            return "kpi"
        if "top merchants" in q or "top 5 merchants" in q:
            return "top_merchants"
        if "spending by category" in q or "top categories" in q:
            return "top_categories"
        if "recurring" in q or "subscription" in q or "monthly bill" in q:
            return "recurring"
        if "unusual" in q or "anomaly" in q or "fraud" in q or "suspicious" in q:
            return "anomalies"
        if "trend" in q or "increasing" in q or "decreasing" in q:
            return "trend"
        if "overspend" in q or "budget" in q:
            return "budget"
        if "last 5 transactions" in q or "recent transactions" in q:
            return "recent"

        if q in ("all", "summary", "overview") or "full summary" in q or "account summary" in q:
            return "full_summary"

        return "generic"

    def _fmt_money(self, x, currency):
        try:
            return f"{float(x):,.2f} {currency}"
        except Exception:
            return f"{x} {currency}"

    def answer_kpi(self, transactions):
        k = KPIEngineV2.execute(transactions)
        if isinstance(k, str):
            return k
        currency = get_currency(transactions[0]) if transactions else "SAR"
        sr = k.get("kpi_savings_rate")
        sr_txt = f"{sr*100:.1f}%" if isinstance(sr, (int, float)) else "N/A"
        return (
            f" KPI Summary \n"
            f"- Total spend:  {self._fmt_money(k['kpi_total_spend'], currency)} \n"
            f"- Total income:  {self._fmt_money(k['kpi_total_income'], currency)} \n"
            f"- Net cashflow:  {self._fmt_money(k['kpi_net_cashflow'], currency)} \n"
            f"- Savings rate:  {sr_txt} \n"
            f"- Avg daily spend:  {self._fmt_money(k.get('kpi_avg_daily_spend',0), currency)} \n"
            f"- Online spend share:  {k.get('kpi_online_spend_share',0)*100:.1f}% \n"
            f"- Largest debit:  {self._fmt_money(k['kpi_largest_debit'], currency)} \n"
            f"- Largest credit:  {self._fmt_money(k['kpi_largest_credit'], currency)} "
        )

    def answer_top_merchants(self, transactions, n=5):
        ta = self.tools["transaction_analyzer"].execute(transactions)
        if isinstance(ta, str):
            return ta
        currency = get_currency(transactions[0]) if transactions else "SAR"
        merch = ta.get("all_merchants", {})
        ranked = sorted(merch.items(), key=lambda x: x[1], reverse=True)[:n]
        if not ranked:
            return "No merchant spending found."
        lines = [f"{i+1}.  {m}  — {self._fmt_money(v, currency)}" for i, (m, v) in enumerate(ranked)]
        return " Top merchants by spending: \n" + "\n".join(lines)

    def answer_top_categories(self, transactions, n=5):
        ta = self.tools["transaction_analyzer"].execute(transactions)
        if isinstance(ta, str):
            return ta
        currency = get_currency(transactions[0]) if transactions else "SAR"
        cats = ta.get("all_categories", {})
        ranked = sorted(cats.items(), key=lambda x: x[1], reverse=True)[:n]
        if not ranked:
            return "No category spending found."
        lines = [f"{i+1}.  {c}  — {self._fmt_money(v, currency)}" for i, (c, v) in enumerate(ranked)]
        return " Top categories by spending: \n" + "\n".join(lines)

    def answer_recurring(self, transactions):
        r = self.tools["recurring_payment_detector"].execute(transactions)
        if isinstance(r, str):
            return r
        items = r.get("recurring_merchants", [])
        if not items:
            return "No strong recurring/subscription patterns detected from the available data."
        currency = get_currency(transactions[0]) if transactions else "SAR"
        lines = []
        for x in items[:8]:
            lines.append(
                f"-  {x['merchant']}  ({x['interval']}, {x['count']} times) "
                f"avg  {self._fmt_money(x['avg_amount'], currency)} , last  {x['last_date']} , "
                f"next likely  {x['predicted_next_date']} "
            )
        return " Recurring payments detected: \n" + "\n".join(lines)

    def answer_anomalies(self, transactions):
        a = self.tools["anomaly_detector"].execute(transactions)
        if isinstance(a, str):
            return a
        items = a.get("anomalies", [])
        if not items:
            return a.get("note", "No anomalies detected.")
        currency = get_currency(transactions[0]) if transactions else "SAR"
        lines = []
        for x in items[:10]:
            lines.append(
                f"-  {x['date']}  — {self._fmt_money(x['amount'], currency)} — {x['merchant']} — {x['category']} (z={x['z_score']:.2f})"
            )
        return " Unusual (large) debit transactions: \n" + "\n".join(lines)

    def answer_trend(self, transactions):
        t = self.tools["trend_analyzer"].execute(transactions)
        if isinstance(t, str):
            return t
        currency = get_currency(transactions[0]) if transactions else "SAR"
        if t.get("trend") is None:
            return (
                f"Trend not available.\n"
                f"- Last window spend:  {self._fmt_money(t.get('last_window_spend',0), currency)} "
            )
        pct = t["pct_change"] * 100
        return (
            f" Spending trend ({t['window_days']} days):   {t['trend']} \n"
            f"- Last {t['window_days']}d:  {self._fmt_money(t['last_window_spend'], currency)} \n"
            f"- Prev {t['window_days']}d:  {self._fmt_money(t['prev_window_spend'], currency)} \n"
            f"- Change:  {pct:+.1f}% "
        )

    def answer_budget(self, transactions):
        b = self.tools["budget_monitor"].execute(transactions)
        if isinstance(b, str):
            return b
        overs = b.get("overspends", [])
        currency = get_currency(transactions[0]) if transactions else "SAR"
        if not overs:
            return f"No category overspends flagged for  {b.get('current_month','current month')} ."
        lines = []
        for x in overs:
            lines.append(
                f"-  {x['category']} : {self._fmt_money(x['current_spend'], currency)} "
                f"vs baseline {self._fmt_money(x['baseline_avg'], currency)} "
                f"( +{x['pct_increase']*100:.1f}% )"
            )
        return f" Overspend alerts for {b.get('current_month')} :\n" + "\n".join(lines)

    def answer_recent(self, transactions):
        lt = self.tools["last_transaction_tool"].execute(transactions, n_recent=5)
        if isinstance(lt, str):
            return lt
        lines = []
        for t in lt["recent_transactions"]:
            lines.append(
                f"- {t['date']} — {t['amount']:,.2f} {t['currency']} — {t.get('merchant','')} — {t.get('category_name','')}"
            )
        return " Last 5 transactions: \n" + "\n".join(lines)

    def answer_next_salary(self, transactions):
        salary_data = self.tools["salary_detector"].execute(transactions)
        if isinstance(salary_data, str):
            return f"I tried to detect your salary pattern, but: {salary_data}"

        avg_income = salary_data.get("detected_income_avg")
        next_date = salary_data.get("predicted_next_income_date", None)
        income_dates = salary_data.get("income_dates", [])
        currency = get_currency(transactions[0]) if transactions else "SAR"

        if not avg_income or not income_dates:
            return "I couldn't detect a clear repeating salary/income pattern from your past credits."
        if next_date:
            return (
                f"Based on your past income credits, your  average income per payment  is about "
                f" {avg_income:,.2f} {currency} .\n\n"
                f"Your  next income is likely around {next_date} .\n"
                f"(Past income dates: {', '.join(income_dates)}.)"
            )

        return (
            f"I found repeated income credits with an  average amount of {avg_income:,.2f} {currency} , "
            f"but the gaps between dates are irregular, so I’m not confident predicting the next date.\n"
            f"Past income dates: {', '.join(income_dates)}."
        )

    def answer_afford_purchase(self, transactions, query: str):
        currency = get_currency(transactions[0]) if transactions else "SAR"

        price = extract_price_from_text(query)
        assumed_item = None
        assumed_item_currency = None
        assumed_note = None

        if price is None:
            pick = CatalogLookupTool.execute(query)
            best = pick.get("best")
            if best and best.get("price"):
                assumed_item = best.get("name")
                assumed_item_currency = best.get("currency") or currency
                assumed_note = pick.get("note")
                price = float(best["price"])

        savings_data = self.tools["monthly_savings_analyzer"].execute(transactions)
        if isinstance(savings_data, str):
            return f"I tried to calculate your monthly savings, but: {savings_data}"

        avg_income = savings_data.get("avg_monthly_income", 0.0)
        avg_expense = savings_data.get("avg_monthly_expense", 0.0)
        avg_saving = savings_data.get("avg_monthly_saving", 0.0)

        if price is None:
            return (
                "I couldn’t find a price in your question and I couldn’t find a priced match in your catalog.\n\n"
                f"- Avg monthly income:  {avg_income:,.2f} {currency} \n"
                f"- Avg monthly expense:  {avg_expense:,.2f} {currency} \n"
                f"- Avg monthly saving:  {avg_saving:,.2f} {currency} \n\n"
                "Try asking: `Can I buy a laptop for 4000?` OR ensure the catalog has cleaned_price values."
            )

        intro = ""
        if assumed_item:
            intro = (
                f"Using your item catalog ({assumed_note}):\n"
                f"- Selected item:  {assumed_item} \n"
                f"- Catalog price:  {price:,.2f} {assumed_item_currency} \n\n"
            )

        if avg_saving <= 0:
            return (
                intro +
                f"Based on your data, your  average monthly saving is {avg_saving:,.2f} {currency} , "
                f"so you are  not saving positive money on average .\n\n"
                f"Buying for about  {price:,.2f} {assumed_item_currency or currency}  would likely require "
                "reducing expenses, using existing savings, or financing."
            )

        months_needed = price / avg_saving
        if months_needed <= 1.0:
            return intro + "✅ You can likely afford it within  ~1 month  of saving."
        if months_needed <= 3.0:
            return intro + f"⚠️ You need roughly  {months_needed:.1f} months  of saving."
        return intro + f"❌ You need around  {months_needed:.1f} months  of saving."

    def call_llm(self, prompt, max_tokens=600):
        try:
            completion = or_client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a factual financial analysis AI. "
                            "Use ONLY the exact data provided in the prompt. "
                            "Never invent amounts, dates, or details not present."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.85,
                extra_headers={
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "Agentic RAG Finance Assistant (Unified)",
                },
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return f"ERROR_FROM_LLM: {str(e)}"

    def _append_suggested_questions(self, response: str):
        suggested_questions = []
        if not str(response).startswith("ERROR_FROM_LLM"):
            response += " "
            for qx in suggested_questions:
                response += f"- {qx}\n"
        return response

    def execute(self, account_id: str, query: str, history=None):
        q = (query or "").strip()
        if not q:
            return "Please type a question."

        user_txns = [
            m["metadata"]
            for m in meta
            if m.get("metadata", {}).get("account_id", m.get("metadata", {}).get("AccountId")) == account_id
        ]
        if not user_txns:
            return "No transaction data available for this account."

        user_txns, date_label = filter_transactions_by_date(user_txns, q)
        if not user_txns:
            return "No transaction data available for this account in the requested date range."

        intent = self.classify_intent(q)

        if intent == "kpi":
            return self._append_suggested_questions(self.answer_kpi(user_txns))
        if intent == "top_merchants":
            return self._append_suggested_questions(self.answer_top_merchants(user_txns, n=5))
        if intent == "top_categories":
            return self._append_suggested_questions(self.answer_top_categories(user_txns, n=5))
        if intent == "recurring":
            return self._append_suggested_questions(self.answer_recurring(user_txns))
        if intent == "anomalies":
            return self._append_suggested_questions(self.answer_anomalies(user_txns))
        if intent == "trend":
            return self._append_suggested_questions(self.answer_trend(user_txns))
        if intent == "budget":
            return self._append_suggested_questions(self.answer_budget(user_txns))
        if intent == "recent":
            return self._append_suggested_questions(self.answer_recent(user_txns))
        if intent == "next_salary":
            return self._append_suggested_questions(self.answer_next_salary(user_txns))
        if intent == "afford_purchase":
            return self._append_suggested_questions(self.answer_afford_purchase(user_txns, q))

        tools_to_use = [
            "kpi_engine",
            "transaction_analyzer",
            "behavior_analyzer",
            "salary_detector",
            "donation_analyzer",
            "recurring_payment_detector",
            "anomaly_detector",
            "budget_monitor",
            "trend_analyzer",
            "last_transaction_tool",
            "shopping_travel_analyzer",
            "category_pattern_predictor",
        ]

        tool_results = {}
        for tool_name in tools_to_use:
            if tool_name == "semantic_search":
                tool_results[tool_name] = self.tools[tool_name].execute(q, k=5)
            elif tool_name == "last_transaction_tool":
                tool_results[tool_name] = self.tools[tool_name].execute(user_txns)
            elif tool_name == "category_pattern_predictor":
                tool_results[tool_name] = self.tools[tool_name].execute(user_txns, q)
            else:
                tool_results[tool_name] = self.tools[tool_name].execute(user_txns)

        context_parts = []
        context_parts.append("Sample Transactions (first 10):")
        for t in user_txns[:10]:
            context_parts.append(
                f"- Date: {get_date_str(t)} | "
                f"Amount: {get_amount(t):,.2f} {get_currency(t)} | "
                f"Category: {get_field(t, 'category_name', '')} | "
                f"Merchant: {get_field(t, 'merchant', '')} | "
                f"City: {get_field(t, 'city', get_field(t, 'location', ''))}"
            )

        context_parts.append("\nAnalysis Results:")
        for tool_name, result in tool_results.items():
            context_parts.append(f"\n{tool_name.upper()}:")
            context_parts.append(json.dumps(result, indent=2))

        prompt = f"""
You are an expert financial advisor AI.

BEHAVIOR RULES:
1. If the user's input is a simple greeting (e.g., "hello", "hi", "how are you"):
   - Respond politely and briefly.
   - Do NOT reference account data or the DATA section.
   - Example: "Hello! I'm here to help with your financial questions."

2. If the user's input is a financial question:
   - Use ONLY the information provided in the DATA section.
   - Do NOT invent, assume, estimate, or infer any amounts, dates, or facts not explicitly stated.
   - If the data is insufficient, clearly state that.

USER QUESTION:
{q}

ACCOUNT INFORMATION:
- Account ID : {account_id}
- Date Range : {date_label or "Full history"}

DATA:
{chr(10).join(context_parts)}

RESPONSE GUIDELINES (for financial questions only):
- Use clear bullet points.
- Organize content under meaningful headings.
- Align numbers, dates, and categories neatly.
- Keep the tone professional, neutral, and concise.
- Avoid long paragraphs; prefer short, scannable points.

FINAL ANSWER:
"""

        response = self.call_llm(prompt)
        return self._append_suggested_questions(response)

    def get_dashboard_data(self, account_id: str):
        txns = [
            m["metadata"]
            for m in meta
            if m.get("metadata", {}).get("account_id", m.get("metadata", {}).get("AccountId")) == account_id
        ]
        if not txns:
            return {"error": "No transactions available for this account."}

        currency = get_currency(txns[0]) if txns else "SAR"
        kpis = KPIEngineV2.execute(txns)
        if isinstance(kpis, str):
            return {"error": kpis}

        def _top_n_dict(d, n=8):
            items = sorted((d or {}).items(), key=lambda x: x[1], reverse=True)[:n]
            return [{"name": k, "value": float(v)} for k, v in items]

        return {
            "account_id": account_id,
            "currency": currency,
            "kpis": kpis,
            "top_categories": _top_n_dict(kpis.get("breakdown_spend_by_category", {}), n=8),
            "top_merchants": _top_n_dict(kpis.get("breakdown_spend_by_merchant", {}), n=8),
            "monthly_series": kpis.get("monthly_series", []),
        }


# =========================================
# 6) INIT AGENT + ACCOUNT IDS
# =========================================

agent = AgenticRAG()

account_ids = sorted({
    m.get("metadata", {}).get("account_id", m.get("metadata", {}).get("AccountId"))
    for m in meta
    if m.get("metadata", {}).get("account_id", m.get("metadata", {}).get("AccountId"))
})

print("✅ Found account_ids:", len(account_ids))
if account_ids:
    print("   - example:", account_ids[0])


# =========================================
# 7) FLASK APP
# =========================================

app = Flask(__name__)
CORS(app)

app.secret_key = os.environ.get("FLASK_SECRET_KEY")
if not app.secret_key:
    app.secret_key = secrets.token_hex(32)
    print("⚠️ FLASK_SECRET_KEY not set, using temporary key (sessions reset on restart).")

app.permanent_session_lifetime = timedelta(hours=4)

@app.before_request
def make_session_permanent():
    session.permanent = True


# ------------------------------
# Insights API for charts (dashboard)
# ------------------------------

def get_account_transactions(account_id: str):
    return [
        m["metadata"]
        for m in meta
        if m.get("metadata", {}).get("account_id", m.get("metadata", {}).get("AccountId")) == account_id
    ]

def _top_n_dict(d, n=8):
    items = sorted((d or {}).items(), key=lambda x: x[1], reverse=True)[:n]
    return [{"name": k, "value": float(v)} for k, v in items]

def compute_account_insights(account_id: str):
    txns = get_account_transactions(account_id)
    if not txns:
        return {"ok": False, "message": "No transactions available for this account."}

    currency = get_currency(txns[0]) if txns else "SAR"

    kpis = KPIEngineV2.execute(txns)
    if isinstance(kpis, str):
        return {"ok": False, "message": kpis}

    total_income = float(kpis.get("kpi_total_income", 0.0) or 0.0)
    total_spend = float(kpis.get("kpi_total_spend", 0.0) or 0.0)
    net_cashflow = float(kpis.get("kpi_net_cashflow", 0.0) or 0.0)

    months = kpis.get("monthly_series", [])
    if months:
        avg_monthly_income = float(np.mean([m.get("income", 0.0) for m in months]))
        avg_monthly_expense = float(np.mean([m.get("spend", 0.0) for m in months]))
    else:
        avg_monthly_income = 0.0
        avg_monthly_expense = 0.0

    top_merchants = _top_n_dict(kpis.get("breakdown_spend_by_merchant", {}), n=8)
    top_categories = _top_n_dict(kpis.get("breakdown_spend_by_category", {}), n=8)

    return {
        "ok": True,
        "account_id": account_id,
        "currency": currency,

        "total_income": total_income,
        "total_spend": total_spend,
        "net_cashflow": net_cashflow,
        "avg_monthly_income": float(avg_monthly_income),
        "avg_monthly_expense": float(avg_monthly_expense),
        "months": months,

        "top_merchants": top_merchants,
        "top_categories": top_categories,

        "kpis": kpis,
    }

@app.route("/api/insights/<account_id>")
def api_insights(account_id):
    if account_id not in account_ids:
        return jsonify({"ok": False, "message": "Unknown account_id"}), 404
    return jsonify(compute_account_insights(account_id))


@app.route("/", methods=["GET", "POST"])
def index():
    sid = get_or_create_sid()

    if request.method == "GET":
        acc_from_qs = request.args.get("account_id")
        if acc_from_qs and acc_from_qs in account_ids:
            session["account_id"] = acc_from_qs

    selected_account = session.get("account_id", account_ids[0] if account_ids else None)
    history = get_history_for_account_mem(selected_account, sid) if selected_account else []

    if request.method == "POST":
        message = request.form.get("message", "").strip()
        selected_account = request.form.get("account_id") or selected_account
        session["account_id"] = selected_account

        charts = []
        if not selected_account:
            reply = "Please select an Account ID first."
        elif not message:
            reply = "Please type a question."
        else:
            txns = get_account_transactions(selected_account)
            txns, date_label = filter_transactions_by_date(txns, message)

            if wants_charts(message):
                charts = build_chat_charts(txns, message)
                reply = (
                    f"Here are your requested chart(s).\n"
                    f"- Account: {selected_account}\n"
                    f"- Date range: {date_label or 'Full history'}\n"
                    f"- Charts generated: {len(charts)}"
                )
            else:
                reply = agent.execute(selected_account, message)

        append_history_for_account_mem(selected_account, sid, message, reply, charts=charts)
        return redirect(url_for("index", account_id=selected_account))

    return render_template(
        "index.html",
        history=history,
        account_ids=account_ids,
        selected_account=selected_account,
    )

@app.route("/reset")
def reset():
    sid = get_or_create_sid()
    selected_account = session.get("account_id", account_ids[0] if account_ids else None)
    if selected_account:
        clear_history_for_account_mem(selected_account, sid)
    return redirect(url_for("index", account_id=selected_account))


# =========================================
# 8) GRADIO UI (unchanged)
# =========================================

def gr_chat(message, selected_account, history):
    message = (message or "").strip()
    history = history or []

    if not selected_account:
        reply = "Please select an Account ID first."
    elif not message:
        reply = "Please type a question."
    else:
        reply = agent.execute(selected_account, message)

    history.append([message, reply])
    return history, history

def gr_reset():
    return [], []

def launch_gradio():
    default_acc = account_ids[0] if account_ids else None

    with gr.Blocks(title="Agentic RAG Finance Assistant (Unified)") as demo:
        gr.Markdown(
            "# 💳 Agentic RAG Finance Assistant (Unified)\n"
            "- RAG + Tools + Item Catalog (auto-price using cleaned_price)\n"
            "- Works with both Flask and Gradio\n\n"
        )

        account_dd = gr.Dropdown(
            choices=account_ids,
            value=default_acc,
            label="Select Account ID",
            interactive=True
        )

        chatbot = gr.Chatbot(label="Conversation", height=520)
        state_history = gr.State([])

        with gr.Row():
            msg = gr.Textbox(label="Your question", placeholder="Ask a question...", scale=8)
            send = gr.Button("Send", scale=2)

        reset_btn = gr.Button("Reset chat")

        send.click(fn=gr_chat, inputs=[msg, account_dd, state_history], outputs=[chatbot, state_history]).then(lambda: "", None, msg)
        msg.submit(fn=gr_chat, inputs=[msg, account_dd, state_history], outputs=[chatbot, state_history]).then(lambda: "", None, msg)
        reset_btn.click(fn=gr_reset, inputs=None, outputs=[chatbot, state_history])

    demo.launch(share=True)


# =========================================
# 9) MOBILE API ENDPOINTS (unchanged)
# =========================================

@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "success": True,
        "status": "running",
        "version": "1.0"
    })

@app.route("/api/accounts", methods=["GET"])
def api_get_accounts():
    return jsonify({
        "success": True,
        "accounts": account_ids
    })

@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        account_id = data.get("account_id")
        message = (data.get("message") or "").strip()
        session_id = data.get("session_id")

        if not account_id:
            return jsonify({"success": False, "error": "account_id is required"}), 400
        if not message:
            return jsonify({"success": False, "error": "message is required"}), 400

        history = []
        if session_id:
            history_key = f"history_{session_id}"
            history = session.get(history_key, [])

        response = agent.execute(account_id, message, history=history)

        if session_id:
            history.append([message, response])
            session[history_key] = history[-10:]

        return jsonify({
            "success": True,
            "response": response,
            "account_id": account_id
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/dashboard", methods=["POST"])
def api_dashboard():
    try:
        data = request.get_json() or {}
        account_id = data.get("account_id")

        if not account_id:
            return jsonify({"success": False, "error": "account_id is required"}), 400

        dashboard_data = agent.get_dashboard_data(account_id)

        if "error" in dashboard_data:
            return jsonify({"success": False, "error": dashboard_data["error"]}), 404

        return jsonify({"success": True, "data": dashboard_data})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =========================================
# 10) MAIN
# =========================================
if __name__ == "__main__":
    try:
        import google.colab  # type: ignore
        launch_gradio()
    except Exception:
        app.run(host="0.0.0.0", port=5000, debug=True)
