# Technical Report: AI-Powered Churn Prediction Chatbot

## 1. Business Motivation

Customer churn is one of the most critical metrics for telecom companies. Acquiring a new customer costs 5-25x more than retaining an existing one. By predicting which customers are likely to leave, the marketing team can proactively deploy targeted retention campaigns, reducing churn and protecting revenue.

This PoC demonstrates an AI-powered solution that makes churn prediction accessible to non-technical marketing staff through natural language conversation in both Arabic and English.

## 2. Model Selection

### Why XGBoost?

We evaluated multiple classical ML approaches on the Telco Customer Churn dataset (7,043 customers, 17 predictive features):

| Model | AUC-ROC | F1 (Churn) | Recall (Churn) | Notes |
|-------|---------|------------|----------------|-------|
| Logistic Regression | 0.84 | 0.60 | — | Simple baseline, fully interpretable |
| Random Forest | 0.84 | 0.62 | — | Good but overfits on small data |
| **XGBoost (tuned)** | **0.85** | **0.64** | **0.81** | **Best overall, recall-optimized** |
| Decision Tree (depth=5) | 0.83 | 0.58 | — | Used for explainability layer |

XGBoost was selected as the primary model because:
- Highest AUC-ROC (0.85) after Optuna hyperparameter tuning (50 trials, 5-fold CV)
- Recall-optimized threshold (0.81 recall) — in churn prediction, missing a churner (false negative) is far more costly than a false alarm, since acquiring a new customer costs 5-25x more than retaining one. The threshold is tuned to ensure at least 80% of actual churners are caught.
- Native handling of class imbalance via `scale_pos_weight`
- Compatible with SHAP for per-prediction explanations
- ONNX export for fast inference suitable for real-time chatbot responses

### Explainability: Dual Approach

1. **SHAP (per-prediction)**: For each customer, we compute the top 5 factors driving the churn prediction. This tells the marketing team *why* this specific customer is at risk.

2. **Decision Tree (global)**: A pruned CART model (max_depth=5) provides human-readable rules that describe the overall churn patterns. The marketing team can understand general risk factors without looking at individual predictions.

## 3. Feature Engineering

### Preprocessing Steps
1. Strip whitespace from column headers
2. Convert `Total_Charges` blanks (11 rows with tenure=0) to 0.0
3. Drop `customerID` (non-predictive identifier)
4. Drop `gender` and `Dual` (non-predictive after analysis)

### Engineered Features
- **tenure_bins**: Discretized tenure into 4 buckets (0-12, 13-24, 25-48, 49+ months) to capture the nonlinear relationship between tenure and churn
- **service_count**: Count of active add-on services (0-6). More services correlate with lower churn
- **avg_monthly_charge**: Total charges divided by tenure. Captures price trajectory

### Encoding Strategy
- Binary features (married, senior citizen, etc.): One-hot with binary drop
- Multi-class categoricals (contract, internet service, etc.): Full one-hot encoding
- Numeric features (tenure, charges): StandardScaler normalization

We preserved the "No internet service" and "No phone service" categories rather than collapsing them to "No" because they carry distinct predictive signal: a customer without internet is fundamentally different from one who has internet but declined an add-on.

## 4. Class Imbalance

The dataset has 73.5% non-churners vs 26.5% churners. We address this through:
- `scale_pos_weight=2.8` during XGBoost training (ratio of majority/minority)
- Post-hoc threshold optimization: the default 0.5 threshold is shifted to ensure at least 80% recall (catching churners) while maximizing F1

## 5. Key Churn Predictors (from SHAP analysis)

1. **Contract type**: Month-to-month contracts have dramatically higher churn risk
2. **Tenure**: New customers (< 12 months) are at highest risk
3. **Online Security**: Customers without online security churn more
4. **Internet Service type**: Fiber optic customers churn more than DSL
5. **Payment Method**: Electronic check users churn more than auto-pay users

## 6. LLM Integration

### Why Qwen 2.5?
- Open-source (meets client requirement of no closed-source LLMs or third-party APIs)
- Excellent multilingual support including Arabic and English
- Strong structured JSON output for tool calling
- Runs fully on local GPU (RTX 2050, 4GB VRAM) via Ollama — no cloud dependency
- No "thinking mode" overhead (unlike Qwen 3.x), ensuring clean JSON tool responses

### Tool-Use Agent Architecture
The chatbot uses a single LLM call pattern with manual JSON-based tool calling:
1. The LLM receives the system prompt with tool definitions and conversation history
2. It outputs either a JSON tool call or a natural language response
3. Tool results are fed back as context for the next LLM iteration
4. The loop continues until the LLM produces a final text response

Available tools:
- **update_customer_features**: Extracts and saves customer fields from natural language
- **get_collection_status**: Reports collected vs missing fields
- **predict_churn**: Runs the ML model when all 17 fields are collected
- **analyze_data**: Queries the historical churn dataset for analytics insights

### Model Configuration
- **Primary**: Qwen 2.5 3B (1.9 GB, runs 100% on GPU)
- **Fallback**: Qwen 2.5 1.5B (986 MB, lighter model for reliability)
- **Temperature**: 0.3 (low for consistent structured output)
- **Context window**: 4096 tokens
- **Latency optimizations**: Flash attention, KV cache quantization (q8_0), persistent model keep-alive

### Logical Defaults
When `internet_service = "No"`, six dependent fields are auto-filled, reducing the number of questions the bot needs to ask by up to 35%.

## 7. Solution Architecture

The solution follows SOLID, KISS, DRY, and YAGNI principles:
- **Single Responsibility**: Each module has one clear purpose
- **Shared Contract**: Pydantic models define the interface between all components
- **Zero Train/Serve Skew**: The same sklearn Pipeline is serialized and used in both training and inference
- **No Premature Abstractions**: Simple functions, no unnecessary design patterns

## 8. Client Requirements Alignment

| Requirement | How We Meet It |
|-------------|---------------|
| Open-source LLM | Qwen 2.5 via Ollama — fully local, no third-party API calls |
| Natural language interaction | Tool-use agent chatbot with Arabic + English |
| Classical ML model | XGBoost with sklearn pipeline, ONNX runtime inference |
| Clear results | SHAP explanations + Decision Tree rules |
| Simple solution | FastAPI + LangChain, minimal dependencies |
| Deployable as API | Docker-ready FastAPI with /chat and /predict endpoints |
| Data analytics | Historical churn dataset queries (segments, trends, patterns) |
