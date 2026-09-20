# System Architecture

## High-Level Architecture

```mermaid
graph TD
    User[Marketing Team<br/>Arabic / English] -->|Natural Language| API[FastAPI Server]

    subgraph API_Layer[API Layer]
        API --> Health[GET /health]
        API --> Predict[POST /predict]
        API --> Chat[POST /chat]
    end

    subgraph Chat_Pipeline[Chatbot Pipeline]
        Chat --> Session[Session Manager<br/>LRU Cache]
        Chat --> Chatbot[ChurnChatbot<br/>Tool-Use Agent]
        Chatbot --> Ollama[Ollama Server<br/>Qwen 2.5 3B / 1.5B]
        Chatbot --> Tools[Tool Dispatcher]
        Tools --> UpdateTool[update_customer_features]
        Tools --> StatusTool[get_collection_status]
        Tools --> PredictTool[predict_churn]
        Tools --> AnalyticsTool[analyze_data]
    end

    subgraph ML_Pipeline[ML Pipeline]
        PredictTool --> Predictor[ChurnPredictor]
        Predict --> Predictor
        Predictor --> Pipeline[sklearn Pipeline<br/>Clean + Encode]
        Pipeline --> XGBoost[XGBoost via ONNX<br/>AUC 0.85]
        Pipeline --> CART[Decision Tree<br/>Explainability]
        Predictor --> SHAP[SHAP Explainer<br/>Top 5 Factors]
    end

    subgraph Data_Layer[Data & Models]
        XGBoost -.->|ONNX| Models[(models/)]
        CART -.->|joblib| Models
        Pipeline -.->|joblib| Models
        SHAP -.->|joblib| Models
    end

    Chatbot -->|Partial Features| Accumulator[PartialCustomerFeatures<br/>17 fields]
    Accumulator -->|Complete| Predictor
    Predictor --> Result[PredictionResult<br/>probability + factors + rules]
    Result --> Chatbot
    AnalyticsTool --> Analytics[DatasetAnalytics<br/>7043 customers]
    Chatbot -->|Formatted Response| User
```

## Data Flow - Chat Interaction

```mermaid
sequenceDiagram
    participant U as Marketing User
    participant A as FastAPI /chat
    participant S as Session Store
    participant L as Qwen 2.5 3B (Ollama)
    participant T as Tool Dispatcher
    participant P as ChurnPredictor
    participant X as XGBoost + SHAP

    U->>A: POST {session_id, message}
    A->>S: get_or_create(session_id)
    S-->>A: ChatSession

    A->>L: System prompt + conversation history
    L-->>A: JSON tool call or text response

    alt Tool call detected
        A->>T: Execute tool (e.g. update_customer_features)
        T-->>A: Tool result
        A->>L: Feed tool result back
        L-->>A: Next action or text response
    end

    alt All 17 features collected
        A->>P: predict(CustomerFeatures)
        P->>X: transform + predict_proba + SHAP
        X-->>P: PredictionResult
        A->>L: Format result for user
        L-->>A: Conversational explanation
    else Missing fields
        L-->>A: Ask for missing fields naturally
    end

    A-->>U: {response, prediction?}
```

## Component Details

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API Server | FastAPI + Uvicorn | REST API with auto-generated docs |
| LLM | Qwen 2.5 3B via Ollama (primary), Qwen 2.5 1.5B (fallback) | Tool-use agent for conversation + extraction |
| ML Model | XGBoost via ONNX Runtime | Churn probability prediction |
| Explainability | SHAP + Decision Tree | Per-prediction and global explanations |
| Analytics | DatasetAnalytics | Historical churn pattern queries |
| Preprocessing | sklearn Pipeline | Reproducible feature engineering |
| Session Management | In-memory LRU dict | Multi-turn conversation state |
