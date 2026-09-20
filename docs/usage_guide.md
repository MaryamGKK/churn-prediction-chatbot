# Marketing Team - Chatbot Usage Guide

## What This Tool Does

The Churn Prediction Chatbot helps you predict whether a customer is likely to leave. You simply describe the customer in natural language (Arabic or English), and the bot will:

1. Extract customer details from your message automatically
2. Predict the likelihood of churn (0-100%)
3. Explain the key factors driving the prediction
4. Suggest retention actions

You can also ask questions about churn trends and high-risk segments from the historical dataset.

## How to Use

### Starting a Conversation

Send a message to the `/chat` endpoint with your question. You can start in English or Arabic:

**English example:**
> "I want to check if a customer might churn. He's been with us for 2 months, month-to-month contract, fiber optic internet."

**Arabic example:**
> "أريد التحقق مما إذا كان العميل سيغادر. عنده شهرين عقد شهري فايبر اوبتك."

### The Bot Will Ask Questions

The bot needs 17 pieces of information about the customer. It will ask for them naturally, 2-3 at a time:

- **Demographics**: Senior citizen status, married, dependents
- **Account**: How long they've been a customer (tenure in months)
- **Services**: Phone, internet type, online security, backup, device protection, tech support, streaming TV/movies
- **Billing**: Contract type, paperless billing, payment method, monthly/total charges

**Tip**: You can provide all details in one message to get an instant prediction:
> "Married customer, 24 months tenure, fiber optic, month-to-month contract, $85/month, electronic check, $2040 total, has online security and tech support, no streaming, no device protection, paperless billing, phone service, not senior, has dependents, no online backup"

### Understanding the Results

When all information is gathered, you'll receive:

1. **Churn Probability**: A percentage (e.g., 60.7%) indicating how likely the customer is to leave
2. **Prediction**: Whether the customer is likely to churn or stay
3. **Top Factors**: The 3-5 most important reasons driving this prediction
4. **Suggested Actions**: Specific retention strategies based on the risk factors

### Data Analytics

You can also ask about churn patterns in the dataset:
> "What are the high risk customer segments?"
> "Show me a data overview"
> "What's the churn rate by contract type?"

Available topics: overview, churn by contract, high risk segments, internet service, payment method, senior citizens, tech support, streaming.

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/chat` | POST | Natural language conversation |
| `/predict` | POST | Direct prediction with structured data |
| `/health` | GET | Check if the service is running |
| `/docs` | GET | Interactive API documentation |

## Tips for Best Results

- Be specific: "2 months" is better than "not long"
- Use the exact service names when you can: "Fiber optic" not just "fast internet"
- Contract types are: "Month-to-month", "One year", or "Two year"
- Payment methods: "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
