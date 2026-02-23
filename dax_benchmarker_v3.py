import streamlit as st
import json
import time
import zipfile
import io
import re
import struct
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DAX Benchmarker – LLM Comparison",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background: #0f1117; }
    
    .hero-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 50%, #1a2040 100%);
        border: 1px solid #2d3561;
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute; inset: 0;
        background: radial-gradient(ellipse at 50% 0%, rgba(99,102,241,0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-header h1 { 
        font-size: 2.4rem; font-weight: 700; 
        background: linear-gradient(135deg, #818cf8, #a78bfa, #60a5fa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
    }
    .hero-header p { color: #8892a4; font-size: 1.05rem; margin: 0; }
    
    .card {
        background: #1a1f2e;
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em;
        color: #6366f1; text-transform: uppercase; margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e2235, #1a1f2e);
        border: 1px solid #2d3561;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #818cf8; }
    .metric-label { font-size: 0.75rem; color: #8892a4; margin-top: 0.25rem; }
    
    .llm-badge {
        display: inline-block; padding: 0.2rem 0.7rem;
        border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    }
    .badge-openai { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
    .badge-gemini { background: rgba(59,130,246,0.15); color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); }
    .badge-claude { background: rgba(249,115,22,0.15); color: #f97316; border: 1px solid rgba(249,115,22,0.3); }
    
    .rank-badge {
        font-size: 1.5rem; font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    
    .dax-code {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.82rem;
        color: #e6edf3;
        white-space: pre-wrap;
        overflow-x: auto;
        max-height: 300px;
        overflow-y: auto;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 24px rgba(99,102,241,0.35) !important;
    }
    
    .status-ok { color: #10b981; }
    .status-warn { color: #f59e0b; }
    .status-err { color: #ef4444; }
    
    div[data-testid="stExpander"] {
        border: 1px solid #2d3561 !important;
        border-radius: 10px !important;
        background: #1a1f2e !important;
    }
    
    .sidebar-section {
        background: #1a1f2e;
        border: 1px solid #2d3561;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PBIX Parsing
# ─────────────────────────────────────────────

def parse_pbix(file_bytes: bytes) -> dict:
    """
    Parse a .pbix file (which is a ZIP) and extract semantic model metadata.
    Returns a dict with tables, columns, measures, relationships.
    """
    metadata = {
        "tables": [],
        "columns": [],
        "measures": [],
        "relationships": [],
        "raw_model": None,
        "parse_method": "unknown",
        "warnings": []
    }
    
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            names = z.namelist()
            
            # Try DataModelSchema first (uncompressed JSON)
            if "DataModelSchema" in names:
                try:
                    raw = z.read("DataModelSchema")
                    # Strip BOM if present
                    text = raw.decode("utf-16-le", errors="replace").lstrip("\ufeff")
                    model = json.loads(text)
                    metadata["raw_model"] = model
                    metadata["parse_method"] = "DataModelSchema (utf-16-le)"
                    _extract_from_model(model, metadata)
                    return metadata
                except Exception as e:
                    metadata["warnings"].append(f"DataModelSchema utf-16-le failed: {e}")
                
                # Retry utf-8
                try:
                    raw = z.read("DataModelSchema")
                    text = raw.decode("utf-8", errors="replace").lstrip("\ufeff")
                    model = json.loads(text)
                    metadata["raw_model"] = model
                    metadata["parse_method"] = "DataModelSchema (utf-8)"
                    _extract_from_model(model, metadata)
                    return metadata
                except Exception as e:
                    metadata["warnings"].append(f"DataModelSchema utf-8 failed: {e}")

            # Detect compressed DataModel (pbix binary) and warn user
            if "DataModel" in names and "DataModelSchema" not in names:
                metadata["warnings"].append(
                    "⚠️ This .pbix contains a compressed binary DataModel (XPress9 format) "
                    "which cannot be parsed directly. Please export as .pbit from Power BI Desktop "
                    "(File → Export → Power BI Template) and upload the .pbit file instead."
                )

            # Try Model/datamodel files
            model_candidates = [n for n in names if 
                "model" in n.lower() or "schema" in n.lower() or n.endswith(".json")]
            for candidate in model_candidates:
                try:
                    raw = z.read(candidate)
                    for enc in ["utf-16-le", "utf-8", "utf-16-be"]:
                        try:
                            text = raw.decode(enc, errors="replace").lstrip("\ufeff")
                            if "{" in text and "tables" in text.lower():
                                model = json.loads(text)
                                metadata["raw_model"] = model
                                metadata["parse_method"] = f"{candidate} ({enc})"
                                _extract_from_model(model, metadata)
                                if metadata["tables"]:
                                    return metadata
                        except:
                            pass
                except:
                    pass
            
            # Fallback: scan all entries
            metadata["warnings"].append("No DataModelSchema found; scanning all ZIP entries.")
            metadata["parse_method"] = "fallback scan"
            for name in names:
                try:
                    raw = z.read(name)
                    if len(raw) < 50 or len(raw) > 50_000_000:
                        continue
                    for enc in ["utf-16-le", "utf-8"]:
                        try:
                            text = raw.decode(enc, errors="replace")
                            if '"tables"' in text or '"Tables"' in text:
                                model = json.loads(text.lstrip("\ufeff"))
                                _extract_from_model(model, metadata)
                                if metadata["tables"]:
                                    metadata["parse_method"] = f"fallback: {name}"
                                    return metadata
                        except:
                            pass
                except:
                    pass

    except zipfile.BadZipFile:
        metadata["warnings"].append("File is not a valid ZIP/PBIX/PBIT.")
    except Exception as e:
        metadata["warnings"].append(f"Unexpected error: {e}")
    
    return metadata


def _extract_from_model(model: dict, metadata: dict):
    """Walk the model JSON and populate metadata dict."""
    # Navigate to tables
    tables_raw = None
    # Common paths
    for path in [
        ["model", "tables"],
        ["Model", "Tables"],
        ["tables"],
        ["Tables"],
    ]:
        try:
            obj = model
            for key in path:
                obj = obj[key]
            tables_raw = obj
            break
        except (KeyError, TypeError):
            pass
    
    if not tables_raw:
        # Try relationships at top level
        pass
    
    if tables_raw:
        for tbl in tables_raw:
            tname = tbl.get("name") or tbl.get("Name", "Unknown")
            is_hidden = tbl.get("isHidden") or tbl.get("IsHidden", False)
            metadata["tables"].append({
                "name": tname,
                "isHidden": is_hidden,
                "description": tbl.get("description", "")
            })
            
            # Columns
            cols = tbl.get("columns") or tbl.get("Columns", [])
            for col in cols:
                cname = col.get("name") or col.get("Name", "")
                if cname.startswith("RowNumber") or cname.startswith("_"):
                    continue
                metadata["columns"].append({
                    "table": tname,
                    "name": cname,
                    "dataType": col.get("dataType") or col.get("DataType", "string"),
                    "isHidden": col.get("isHidden") or col.get("IsHidden", False),
                    "expression": col.get("expression") or col.get("Expression", ""),
                })
            
            # Measures
            measures = tbl.get("measures") or tbl.get("Measures", [])
            for m in measures:
                mname = m.get("name") or m.get("Name", "")
                metadata["measures"].append({
                    "table": tname,
                    "name": mname,
                    "expression": m.get("expression") or m.get("Expression", ""),
                    "description": m.get("description") or m.get("Description", ""),
                })
    
    # Relationships
    rels_raw = None
    for path in [
        ["model", "relationships"],
        ["Model", "Relationships"],
        ["relationships"],
        ["Relationships"],
    ]:
        try:
            obj = model
            for key in path:
                obj = obj[key]
            rels_raw = obj
            break
        except (KeyError, TypeError):
            pass
    
    if rels_raw:
        for rel in rels_raw:
            metadata["relationships"].append({
                "fromTable": rel.get("fromTable") or rel.get("FromTable", ""),
                "fromColumn": rel.get("fromColumn") or rel.get("FromColumn", ""),
                "toTable": rel.get("toTable") or rel.get("ToTable", ""),
                "toColumn": rel.get("toColumn") or rel.get("ToColumn", ""),
                "crossFilteringBehavior": rel.get("crossFilteringBehavior", "singleDirection"),
            })


def metadata_to_prompt_context(metadata: dict) -> str:
    """Convert metadata dict to a concise text block for LLM prompts."""
    lines = ["=== Power BI Semantic Model Metadata ===\n"]
    
    visible_tables = [t for t in metadata["tables"] if not t.get("isHidden")]
    lines.append(f"TABLES ({len(visible_tables)} visible):")
    for t in visible_tables:
        lines.append(f"  - {t['name']}")
    
    lines.append(f"\nCOLUMNS:")
    for c in metadata["columns"]:
        if not c.get("isHidden"):
            expr = f" [Calculated: {c['expression'][:50]}]" if c.get("expression") else ""
            lines.append(f"  - {c['table']}[{c['name']}]  ({c['dataType']}){expr}")
    
    if metadata["measures"]:
        lines.append(f"\nEXISTING MEASURES:")
        for m in metadata["measures"]:
            lines.append(f"  - {m['table']}[{m['name']}] = {m['expression'][:80]}")
    
    if metadata["relationships"]:
        lines.append(f"\nRELATIONSHIPS:")
        for r in metadata["relationships"]:
            lines.append(f"  - {r['fromTable']}[{r['fromColumn']}] → {r['toTable']}[{r['toColumn']}]")
    
    return "\n".join(lines)


# ─────────────────────────────────────────────
# LLM Callers
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Power BI DAX developer. 
Given semantic model metadata and a user question, generate a precise DAX measure.
Respond ONLY with JSON in this exact format:
{
  "measure_name": "MeasureName",
  "dax": "MeasureName = <DAX expression>",
  "explanation": "Brief explanation of the logic",
  "confidence": 85
}
confidence is 0-100. Do not include markdown code fences in the dax field."""


def build_user_message(context: str, question: str) -> str:
    return f"{context}\n\nUSER QUESTION:\n{question}\n\nGenerate the DAX measure:"


def call_openai(api_key: str, model: str, context: str, question: str) -> dict:
    import openai
    client = openai.OpenAI(api_key=api_key)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(context, question)},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    latency = round(time.time() - t0, 2)
    raw = resp.choices[0].message.content.strip()
    parsed = safe_parse_json(raw)
    return {
        "model": model,
        "provider": "OpenAI",
        "raw_response": raw,
        "dax": parsed.get("dax", raw),
        "measure_name": parsed.get("measure_name", ""),
        "explanation": parsed.get("explanation", ""),
        "confidence": parsed.get("confidence", 0),
        "latency": latency,
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "total_tokens": resp.usage.total_tokens,
        "error": None,
    }


def call_gemini(api_key: str, model: str, context: str, question: str) -> dict:
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    full_prompt = f"{SYSTEM_PROMPT}\n\n{build_user_message(context, question)}"

    t0 = time.time()
    try:
        # Try with system_instruction (SDK >= 0.5, works with gemini-2.x)
        gmodel = genai.GenerativeModel(
            model_name=model,
            system_instruction=SYSTEM_PROMPT,
        )
        resp = gmodel.generate_content(build_user_message(context, question))
    except Exception:
        # Fallback: combine system + user into single prompt
        gmodel = genai.GenerativeModel(model_name=model)
        resp = gmodel.generate_content(full_prompt)

    latency = round(time.time() - t0, 2)
    raw = resp.text.strip()
    parsed = safe_parse_json(raw)

    try:
        usage = resp.usage_metadata
        ptok = usage.prompt_token_count
        ctok = usage.candidates_token_count
        ttok = usage.total_token_count
    except Exception:
        ptok = ctok = ttok = 0

    return {
        "model": model,
        "provider": "Gemini",
        "raw_response": raw,
        "dax": parsed.get("dax", raw),
        "measure_name": parsed.get("measure_name", ""),
        "explanation": parsed.get("explanation", ""),
        "confidence": parsed.get("confidence", 0),
        "latency": latency,
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "total_tokens": ttok,
        "error": None,
    }


def call_claude(api_key: str, model: str, context: str, question: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    t0 = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(context, question)}],
    )
    latency = round(time.time() - t0, 2)
    raw = resp.content[0].text.strip()
    parsed = safe_parse_json(raw)
    return {
        "model": model,
        "provider": "Claude",
        "raw_response": raw,
        "dax": parsed.get("dax", raw),
        "measure_name": parsed.get("measure_name", ""),
        "explanation": parsed.get("explanation", ""),
        "confidence": parsed.get("confidence", 0),
        "latency": latency,
        "prompt_tokens": resp.usage.input_tokens,
        "completion_tokens": resp.usage.output_tokens,
        "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
        "error": None,
    }


def safe_parse_json(text: str) -> dict:
    """Try to extract JSON from LLM response."""
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    # Find first { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return {}


# ─────────────────────────────────────────────
# Evaluation Metrics
# ─────────────────────────────────────────────

def evaluate_dax(dax: str, metadata: dict) -> dict:
    """Heuristic evaluation of DAX quality."""
    score = {}
    dax_upper = dax.upper()
    
    # 1. Syntactic completeness (has = sign, parentheses balanced)
    has_equals = "=" in dax
    open_p = dax.count("(")
    close_p = dax.count(")")
    balanced = open_p == close_p
    score["syntactic"] = 70 if has_equals else 20
    if balanced: score["syntactic"] += 20
    if len(dax) > 10: score["syntactic"] = min(score["syntactic"] + 10, 100)
    
    # 2. Semantic correctness – references valid table/column names
    table_names = [t["name"].upper() for t in metadata["tables"]]
    col_names = [c["name"].upper() for c in metadata["columns"]]
    refs_found = sum(1 for t in table_names if t in dax_upper)
    col_found = sum(1 for c in col_names if c in dax_upper)
    total_refs = len(table_names) + len(col_names)
    score["semantic"] = min(int((refs_found + col_found) / max(total_refs, 1) * 100) + 40, 100) if total_refs > 0 else 50
    
    # 3. DAX function usage
    dax_functions = ["CALCULATE", "SUMX", "SUM", "AVERAGE", "COUNTROWS", "FILTER", "ALL",
                     "RELATED", "DIVIDE", "IF", "SWITCH", "TOTALYTD", "SAMEPERIODLASTYEAR",
                     "DATEADD", "VALUES", "DISTINCTCOUNT", "MAXX", "MINX", "RANKX"]
    funcs_used = sum(1 for f in dax_functions if f in dax_upper)
    score["dax_usage"] = min(funcs_used * 15 + 30, 100)
    
    # 4. Clarity (length heuristic – not too short, not too long)
    l = len(dax)
    if 20 <= l <= 500:
        score["clarity"] = 80
    elif l < 20:
        score["clarity"] = 30
    else:
        score["clarity"] = max(80 - (l - 500) // 20, 40)
    
    # 5. Overall
    score["overall"] = int((score["syntactic"] * 0.25 + score["semantic"] * 0.35 +
                            score["dax_usage"] * 0.25 + score["clarity"] * 0.15))
    return score


# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────

if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "results" not in st.session_state:
    st.session_state.results = []
if "history" not in st.session_state:
    st.session_state.history = []


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div style="font-size:1.3rem;font-weight:700;color:#818cf8;margin-bottom:1.2rem;">⚙️ Configuration</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card-title">API Keys</div>', unsafe_allow_html=True)
    openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    gemini_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
    claude_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
    
    st.divider()
    
    st.markdown('<div class="card-title">Model Selection</div>', unsafe_allow_html=True)
    openai_model = st.selectbox("ChatGPT Model", [
        "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"
    ])
    gemini_model = st.selectbox("Gemini Model", [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-pro-preview-03-25",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
    ])
    claude_model = st.selectbox("Claude Model", [
        "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"
    ])
    
    st.divider()
    
    st.markdown('<div class="card-title">Active LLMs</div>', unsafe_allow_html=True)
    use_openai = st.checkbox("ChatGPT", value=True)
    use_gemini = st.checkbox("Gemini", value=True)
    use_claude = st.checkbox("Claude", value=True)
    
    st.divider()
    
    # Status indicators
    st.markdown('<div class="card-title">API Status</div>', unsafe_allow_html=True)
    st.markdown(f'{"🟢" if openai_key else "🔴"} OpenAI {"Configured" if openai_key else "Missing key"}')
    st.markdown(f'{"🟢" if gemini_key else "🔴"} Gemini {"Configured" if gemini_key else "Missing key"}')
    st.markdown(f'{"🟢" if claude_key else "🔴"} Anthropic {"Configured" if claude_key else "Missing key"}')


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
    <h1>⚡ DAX Benchmarker</h1>
    <p>Multi-LLM DAX generation · Semantic validation · Performance comparison dashboard</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──
tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload & Parse", "🤖 Generate DAX", "📊 Compare Results", "📜 History"])


# ────────────────────────────────────────────────────────────────
# TAB 1 – Upload & Parse
# ────────────────────────────────────────────────────────────────
with tab1:
    col_upload, col_preview = st.columns([1, 1.5], gap="large")
    
    with col_upload:
        st.markdown('<div class="card-title">Upload .pbix File</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop your Power BI file here",
            type=["pbix", "pbit"],
            help="Upload a .pbit (recommended) or .pbix file. .pbit files contain readable DataModelSchema JSON and parse reliably. .pbix DataModel is compressed binary and may not parse."
        )
        
        if uploaded:
            with st.spinner("🔍 Parsing PBIX metadata..."):
                file_bytes = uploaded.read()
                meta = parse_pbix(file_bytes)
                st.session_state.metadata = meta
            
            if meta["warnings"]:
                for w in meta["warnings"]:
                    st.warning(w)
            
            st.success(f"✅ Parsed via: `{meta['parse_method']}`")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tables", len(meta["tables"]))
            c2.metric("Columns", len(meta["columns"]))
            c3.metric("Measures", len(meta["measures"]))
            c4.metric("Relationships", len(meta["relationships"]))
        
        # Demo mode
        st.divider()
        st.markdown('<div class="card-title">Demo Mode</div>', unsafe_allow_html=True)
        if st.button("Load Sample Metadata"):
            st.session_state.metadata = {
                "tables": [
                    {"name": "Sales", "isHidden": False, "description": ""},
                    {"name": "Product", "isHidden": False, "description": ""},
                    {"name": "Customer", "isHidden": False, "description": ""},
                    {"name": "Date", "isHidden": False, "description": ""},
                ],
                "columns": [
                    {"table": "Sales", "name": "SalesAmount", "dataType": "decimal", "isHidden": False, "expression": ""},
                    {"table": "Sales", "name": "Quantity", "dataType": "int64", "isHidden": False, "expression": ""},
                    {"table": "Sales", "name": "ProductKey", "dataType": "int64", "isHidden": False, "expression": ""},
                    {"table": "Sales", "name": "CustomerKey", "dataType": "int64", "isHidden": False, "expression": ""},
                    {"table": "Sales", "name": "OrderDate", "dataType": "dateTime", "isHidden": False, "expression": ""},
                    {"table": "Sales", "name": "NetSales", "dataType": "decimal", "isHidden": False, "expression": ""},
                    {"table": "Product", "name": "ProductName", "dataType": "string", "isHidden": False, "expression": ""},
                    {"table": "Product", "name": "Category", "dataType": "string", "isHidden": False, "expression": ""},
                    {"table": "Customer", "name": "CustomerName", "dataType": "string", "isHidden": False, "expression": ""},
                    {"table": "Customer", "name": "Region", "dataType": "string", "isHidden": False, "expression": ""},
                    {"table": "Date", "name": "Date", "dataType": "dateTime", "isHidden": False, "expression": ""},
                    {"table": "Date", "name": "Year", "dataType": "int64", "isHidden": False, "expression": ""},
                    {"table": "Date", "name": "Month", "dataType": "string", "isHidden": False, "expression": ""},
                ],
                "measures": [
                    {"table": "Sales", "name": "Total Sales", "expression": "SUM(Sales[SalesAmount])", "description": ""},
                ],
                "relationships": [
                    {"fromTable": "Sales", "fromColumn": "ProductKey", "toTable": "Product", "toColumn": "ProductKey", "crossFilteringBehavior": "singleDirection"},
                    {"fromTable": "Sales", "fromColumn": "CustomerKey", "toTable": "Customer", "toColumn": "CustomerKey", "crossFilteringBehavior": "singleDirection"},
                    {"fromTable": "Sales", "fromColumn": "OrderDate", "toTable": "Date", "toColumn": "Date", "crossFilteringBehavior": "singleDirection"},
                ],
                "raw_model": None,
                "parse_method": "demo",
                "warnings": []
            }
            st.success("✅ Sample metadata loaded!")
            st.rerun()
    
    with col_preview:
        if st.session_state.metadata:
            meta = st.session_state.metadata
            st.markdown('<div class="card-title">Metadata Preview</div>', unsafe_allow_html=True)
            
            with st.expander("📋 Tables", expanded=True):
                df_tables = pd.DataFrame(meta["tables"])
                if not df_tables.empty:
                    st.dataframe(df_tables[["name", "isHidden"]], use_container_width=True, hide_index=True)
            
            with st.expander("📊 Columns"):
                df_cols = pd.DataFrame(meta["columns"])
                if not df_cols.empty:
                    st.dataframe(df_cols[["table", "name", "dataType", "isHidden"]], use_container_width=True, hide_index=True)
            
            with st.expander("📐 Measures"):
                df_meas = pd.DataFrame(meta["measures"])
                if not df_meas.empty:
                    st.dataframe(df_meas[["table", "name", "expression"]], use_container_width=True, hide_index=True)
                else:
                    st.info("No existing measures found.")
            
            with st.expander("🔗 Relationships"):
                df_rels = pd.DataFrame(meta["relationships"])
                if not df_rels.empty:
                    st.dataframe(df_rels, use_container_width=True, hide_index=True)
                else:
                    st.info("No relationships found.")


# ────────────────────────────────────────────────────────────────
# TAB 2 – Generate DAX
# ────────────────────────────────────────────────────────────────
with tab2:
    if not st.session_state.metadata:
        st.info("📁 Please upload a .pbix file or load sample metadata in the **Upload & Parse** tab first.")
    else:
        st.markdown('<div class="card-title">Analytical Question</div>', unsafe_allow_html=True)
        
        example_questions = [
            "Create a measure that calculates total net sales from the Sales table.",
            "Calculate year-over-year sales growth percentage.",
            "Compute the average sales per customer.",
            "Calculate the top 10 products by sales amount.",
            "Create a running total of sales ordered by date.",
        ]
        
        preset = st.selectbox("💡 Example questions (or write your own below)", 
                               ["-- Custom --"] + example_questions)
        
        question = st.text_area(
            "Your question",
            value=preset if preset != "-- Custom --" else "",
            height=100,
            placeholder="e.g. Create a measure that calculates total net sales from the Sales table."
        )
        
        active_llms = []
        if use_openai and openai_key: active_llms.append("openai")
        if use_gemini and gemini_key: active_llms.append("gemini")
        if use_claude and claude_key: active_llms.append("claude")
        
        # Warn about missing keys
        if use_openai and not openai_key:
            st.warning("⚠️ OpenAI key missing — ChatGPT will be skipped.")
        if use_gemini and not gemini_key:
            st.warning("⚠️ Gemini key missing — Gemini will be skipped.")
        if use_claude and not claude_key:
            st.warning("⚠️ Anthropic key missing — Claude will be skipped.")
        
        if not active_llms:
            st.error("❌ No LLMs configured. Please add at least one API key in the sidebar.")
        
        run_btn = st.button("🚀 Generate DAX with All LLMs", disabled=(not question.strip() or not active_llms))
        
        if run_btn and question.strip() and active_llms:
            meta = st.session_state.metadata
            context = metadata_to_prompt_context(meta)
            results = []
            
            progress = st.progress(0, text="Calling LLMs...")
            total = len(active_llms)
            
            for i, llm in enumerate(active_llms):
                try:
                    if llm == "openai":
                        st.info(f"🟢 Calling ChatGPT ({openai_model})...")
                        result = call_openai(openai_key, openai_model, context, question)
                    elif llm == "gemini":
                        st.info(f"🔵 Calling Gemini ({gemini_model})...")
                        result = call_gemini(gemini_key, gemini_model, context, question)
                    elif llm == "claude":
                        st.info(f"🟠 Calling Claude ({claude_model})...")
                        result = call_claude(claude_key, claude_model, context, question)
                    
                    result["scores"] = evaluate_dax(result["dax"], meta)
                    results.append(result)
                
                except Exception as e:
                    results.append({
                        "model": llm, "provider": llm.capitalize(),
                        "error": str(e),
                        "dax": "", "measure_name": "", "explanation": "",
                        "confidence": 0, "latency": 0,
                        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                        "scores": {"syntactic": 0, "semantic": 0, "dax_usage": 0, "clarity": 0, "overall": 0}
                    })
                    st.error(f"❌ {llm} failed: {e}")
                
                progress.progress((i + 1) / total, text=f"Done {i+1}/{total}")
            
            progress.empty()
            
            st.session_state.results = results
            st.session_state.history.append({
                "question": question,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "results": results,
            })
            
            st.success(f"✅ Generated DAX from {len([r for r in results if not r.get('error')])} LLMs!")
            st.info("👉 Switch to the **Compare Results** tab to see the dashboard.")


# ────────────────────────────────────────────────────────────────
# TAB 3 – Compare Results
# ────────────────────────────────────────────────────────────────
with tab3:
    results = st.session_state.results
    
    if not results:
        st.info("🤖 No results yet. Generate DAX in the **Generate DAX** tab first.")
    else:
        valid = [r for r in results if not r.get("error")]
        
        if not valid:
            st.error("All LLM calls failed. Check your API keys and try again.")
        else:
            # Sort by overall score
            valid_sorted = sorted(valid, key=lambda r: r["scores"]["overall"], reverse=True)
            
            COLORS = {"OpenAI": "#10b981", "Gemini": "#3b82f6", "Claude": "#f97316"}
            BADGES = {"OpenAI": "badge-openai", "Gemini": "badge-gemini", "Claude": "badge-claude"}
            EMOJIS = {"OpenAI": "🟢", "Gemini": "🔵", "Claude": "🟠"}
            
            # ── Rank Summary ──
            st.markdown('<div class="card-title">🏆 Rankings</div>', unsafe_allow_html=True)
            rank_cols = st.columns(len(valid_sorted))
            for i, (col, res) in enumerate(zip(rank_cols, valid_sorted)):
                prov = res["provider"]
                with col:
                    medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:2rem">{medal}</div>
                        <div style="font-size:1rem;font-weight:600;color:#e2e8f0;margin:0.3rem 0">{EMOJIS.get(prov,'')} {prov}</div>
                        <div class="metric-value">{res['scores']['overall']}</div>
                        <div class="metric-label">Overall Score</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.divider()
            
            # ── Side-by-side DAX ──
            st.markdown('<div class="card-title">📝 Generated DAX Comparison</div>', unsafe_allow_html=True)
            dax_cols = st.columns(len(valid_sorted))
            for col, res in zip(dax_cols, valid_sorted):
                prov = res["provider"]
                with col:
                    st.markdown(f"""
                    <span class="llm-badge {BADGES.get(prov,'')}">{prov} – {res['model']}</span>
                    <div style="margin:0.5rem 0;font-size:0.8rem;color:#8892a4">
                        ⏱ {res['latency']}s · 🎯 Confidence: {res['confidence']}% · 🪙 {res['total_tokens']} tokens
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f'<div class="dax-code">{res["dax"]}</div>', unsafe_allow_html=True)
                    if res.get("explanation"):
                        with st.expander("💬 Explanation"):
                            st.write(res["explanation"])
            
            st.divider()
            
            # ── Charts Row ──
            st.markdown('<div class="card-title">📊 Performance Dashboard</div>', unsafe_allow_html=True)
            chart_col1, chart_col2 = st.columns(2)
            
            providers = [r["provider"] for r in valid_sorted]
            prov_colors = [COLORS.get(p, "#818cf8") for p in providers]
            
            # Latency bar chart
            with chart_col1:
                fig_lat = go.Figure(go.Bar(
                    x=providers,
                    y=[r["latency"] for r in valid_sorted],
                    marker_color=prov_colors,
                    text=[f"{r['latency']}s" for r in valid_sorted],
                    textposition="outside",
                ))
                fig_lat.update_layout(
                    title="Response Latency (seconds)",
                    plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e",
                    font=dict(color="#e2e8f0"),
                    yaxis=dict(gridcolor="#2d3561"),
                    xaxis=dict(gridcolor="#2d3561"),
                    title_font_color="#818cf8",
                    margin=dict(t=50, b=20),
                )
                st.plotly_chart(fig_lat, use_container_width=True)
            
            # Token usage bar chart
            with chart_col2:
                fig_tok = go.Figure()
                fig_tok.add_trace(go.Bar(
                    name="Prompt Tokens",
                    x=providers,
                    y=[r["prompt_tokens"] for r in valid_sorted],
                    marker_color="#6366f1",
                ))
                fig_tok.add_trace(go.Bar(
                    name="Completion Tokens",
                    x=providers,
                    y=[r["completion_tokens"] for r in valid_sorted],
                    marker_color="#a78bfa",
                ))
                fig_tok.update_layout(
                    barmode="stack",
                    title="Token Consumption",
                    plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e",
                    font=dict(color="#e2e8f0"),
                    yaxis=dict(gridcolor="#2d3561"),
                    legend=dict(bgcolor="#1a1f2e"),
                    title_font_color="#818cf8",
                    margin=dict(t=50, b=20),
                )
                st.plotly_chart(fig_tok, use_container_width=True)
            
            chart_col3, chart_col4 = st.columns(2)
            
            # Radar chart
            with chart_col3:
                categories = ["Syntactic", "Semantic", "DAX Usage", "Clarity", "Overall"]
                fig_radar = go.Figure()
                for res in valid_sorted:
                    s = res["scores"]
                    vals = [s["syntactic"], s["semantic"], s["dax_usage"], s["clarity"], s["overall"]]
                    vals.append(vals[0])  # close loop
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals,
                        theta=categories + [categories[0]],
                        fill="toself",
                        name=res["provider"],
                        line_color=COLORS.get(res["provider"], "#818cf8"),
                        opacity=0.7,
                    ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], gridcolor="#2d3561", color="#8892a4"),
                        angularaxis=dict(gridcolor="#2d3561"),
                        bgcolor="#1a1f2e",
                    ),
                    title="Quality Radar (0–100)",
                    plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e",
                    font=dict(color="#e2e8f0"),
                    legend=dict(bgcolor="#1a1f2e"),
                    title_font_color="#818cf8",
                    margin=dict(t=50, b=20),
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Score breakdown table
            with chart_col4:
                st.markdown("**📋 Score Breakdown**")
                score_data = []
                for res in valid_sorted:
                    s = res["scores"]
                    score_data.append({
                        "Provider": f"{EMOJIS.get(res['provider'],'')} {res['provider']}",
                        "Syntactic": s["syntactic"],
                        "Semantic": s["semantic"],
                        "DAX Usage": s["dax_usage"],
                        "Clarity": s["clarity"],
                        "Overall ⭐": s["overall"],
                        "Latency (s)": res["latency"],
                        "Tokens": res["total_tokens"],
                    })
                df_scores = pd.DataFrame(score_data)
                st.dataframe(
                    df_scores.style.background_gradient(
                        cmap="Blues", subset=["Syntactic","Semantic","DAX Usage","Clarity","Overall ⭐"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            
            # Overall score bar
            st.divider()
            st.markdown('<div class="card-title">🎯 Overall Score Comparison</div>', unsafe_allow_html=True)
            fig_overall = go.Figure(go.Bar(
                x=[r["scores"]["overall"] for r in valid_sorted],
                y=providers,
                orientation="h",
                marker=dict(
                    color=[r["scores"]["overall"] for r in valid_sorted],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Score"),
                ),
                text=[f"{r['scores']['overall']}/100" for r in valid_sorted],
                textposition="auto",
            ))
            fig_overall.update_layout(
                title="Overall Quality Score (0–100)",
                plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e",
                font=dict(color="#e2e8f0"),
                xaxis=dict(range=[0, 100], gridcolor="#2d3561"),
                yaxis=dict(gridcolor="#2d3561"),
                title_font_color="#818cf8",
                height=200 + 60 * len(valid_sorted),
                margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig_overall, use_container_width=True)


# ────────────────────────────────────────────────────────────────
# TAB 4 – History
# ────────────────────────────────────────────────────────────────
with tab4:
    if not st.session_state.history:
        st.info("No benchmark runs yet. Generate some DAX to build history!")
    else:
        st.markdown(f'<div class="card-title">Benchmark History — {len(st.session_state.history)} runs</div>', unsafe_allow_html=True)
        
        for idx, entry in enumerate(reversed(st.session_state.history)):
            with st.expander(f"🕐 {entry['timestamp']} — {entry['question'][:80]}..."):
                valid = [r for r in entry["results"] if not r.get("error")]
                if valid:
                    best = max(valid, key=lambda r: r["scores"]["overall"])
                    st.markdown(f"**🏆 Winner:** {best['provider']} (score: {best['scores']['overall']}/100)")
                    
                    cols = st.columns(len(valid))
                    for col, res in zip(cols, valid):
                        with col:
                            st.markdown(f"**{res['provider']}**")
                            st.code(res["dax"], language="text")
                            st.caption(f"Score: {res['scores']['overall']} | Latency: {res['latency']}s | Tokens: {res['total_tokens']}")
        
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()