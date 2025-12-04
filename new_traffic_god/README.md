# 🚦 New Traffic God

**Advanced Traffic Intelligence Foundation Model for Indian Urban Scenarios**

A large-scale foundation model specifically designed for traffic simulation, prediction, and optimization in the Noida, Indirapuram, and NCR region.

## 🌟 Features

### Core Capabilities
- **Natural Language Interface** - Ask anything about traffic in plain English/Hindi
- **Traffic Flow Prediction** - Real-time and forecasted conditions
- **Route Optimization** - Best routes with alternatives considering live traffic
- **Scenario Simulation** - What-if analysis with physics-based world model
- **Infrastructure Suggestions** - Data-driven recommendations for improvements
- **AQI Impact Assessment** - Traffic's contribution to air quality

### Technical Highlights
- 🧠 **Transformer Architecture** - State-of-the-art foundation model
- 🔍 **RAG System** - Real-time knowledge retrieval and grounding
- 🌍 **World Model** - Neural traffic simulator for counterfactual reasoning
- 📊 **Continuous Learning** - RLHF and feedback-based improvement
- ⚡ **Optimized Inference** - Fast responses with caching and batching

## 📁 Project Structure

```
new_traffic_god/
├── __init__.py              # Main exports
├── traffic_god.py           # Central orchestrator
├── requirements.txt         # Dependencies
│
├── core/                    # Core model components
│   ├── foundation_model.py  # Transformer architecture
│   └── tokenizer.py         # Traffic-specific tokenizer
│
├── simulation/              # Traffic simulation
│   └── world_model.py       # Neural world model
│
├── retrieval/               # Knowledge retrieval
│   └── rag_system.py        # RAG with vector store
│
├── training/                # Training pipelines
│   └── trainer.py           # Pretraining, SFT, RLHF
│
├── inference/               # Inference engine
│   └── predictor.py         # Unified predictor
│
├── evaluation/              # Evaluation system
│   └── evaluator.py         # Benchmarks and metrics
│
├── data/                    # Data and knowledge
│   └── noida_traffic_data.py # Noida/NCR traffic knowledge
│
├── configs/                 # Configuration
│   ├── settings.py          # Python configs
│   └── config.yaml          # YAML config file
│
├── utils/                   # Utilities
│   └── helpers.py           # Common utilities
│
├── api/                     # REST API
│   └── routes.py            # FastAPI endpoints
│
└── models/                  # Model checkpoints
    └── (saved models)
```

## 🚀 Quick Start

### Installation

```bash
cd new_traffic_god
pip install -r requirements.txt
```

### Basic Usage

```python
from new_traffic_god import TrafficGod, create_traffic_god
import asyncio

async def main():
    # Initialize Traffic God
    god = await create_traffic_god(environment="development")
    
    # Ask a question
    response = await god.query("What's the traffic like on NH-24 right now?")
    print(response.answer)
    
    # Plan a route
    route = await god.plan_route(
        source="Sector 18 Noida",
        destination="Indirapuram"
    )
    print(f"Best route: {route['recommended']['name']}")
    print(f"Time: {route['recommended']['duration_min']} minutes")

asyncio.run(main())
```

### Quick Query (Synchronous)

```python
from new_traffic_god import quick_query

answer = quick_query("Best time to travel from Noida to Delhi?")
print(answer)
```

### API Server

```bash
# Start the API server
uvicorn new_traffic_god.api.routes:app --host 0.0.0.0 --port 8080

# Query the API
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Traffic prediction for Sector 62"}'
```

## 🔧 Configuration

### Environment Options

```python
# Development (smaller model, CPU)
god = await create_traffic_god(environment="development")

# Production (full model, GPU)
god = await create_traffic_god(environment="production")

# Testing (minimal model)
god = await create_traffic_god(environment="testing")
```

### Custom Configuration

```python
from new_traffic_god import TrafficGod

god = TrafficGod(
    environment="development",
    config_path="./my_config.yaml",
    device="cuda"  # or "cpu", "mps"
)
await god.initialize()
```

## 📊 Model Architecture

### Foundation Model
- **Architecture**: Transformer with multi-head attention
- **Parameters**: Configurable (512M - 2B+)
- **Context Length**: Up to 2048 tokens
- **Vocabulary**: 50,000 tokens (traffic-specific)

### Model Sizes
| Size | Parameters | d_model | Layers | Heads |
|------|------------|---------|--------|-------|
| Small | ~100M | 512 | 6 | 8 |
| Base | ~350M | 768 | 12 | 12 |
| Large | ~700M | 1024 | 24 | 16 |
| XL | ~1.5B | 1536 | 32 | 24 |

## 🗺️ Supported Regions

### Primary Coverage
- **Noida** - All sectors (1-168), expressways
- **Indirapuram** - Shakti Khand, Niti Khand, Vaibhav Khand
- **Greater Noida** - Tech zones, expressway

### Secondary Coverage
- **Ghaziabad** - Major roads and crossings
- **Delhi NCR** - Key corridors (DND, NH-24)

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Natural language query |
| `/routes/plan` | POST | Route planning |
| `/predict/traffic` | POST | Traffic prediction |
| `/analyze/scenario` | POST | Scenario analysis |
| `/analyze/infrastructure` | POST | Infrastructure suggestions |
| `/aqi/current` | GET | Current AQI |
| `/realtime/traffic` | GET | Real-time traffic |
| `/ws/traffic-updates` | WS | WebSocket updates |

## 🏋️ Training

### Pretraining
```python
from new_traffic_god import TrafficGod

god = await create_traffic_god()
await god.train(
    data_path="./traffic_data",
    epochs=100,
    batch_size=32
)
```

### Fine-tuning
```python
await god.fine_tune(
    dataset_path="./fine_tune_data",
    task="route_planning"
)
```

### RLHF
The model supports reinforcement learning from human feedback for improved responses.

## 📈 Evaluation

```python
from new_traffic_god import Evaluator

evaluator = Evaluator(model, tokenizer)
results = evaluator.evaluate_all(test_dataset)
print(results)
```

## 🛠️ Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Formatting
```bash
black new_traffic_god/
isort new_traffic_god/
```

### Type Checking
```bash
mypy new_traffic_god/
```

## 📝 Examples

### Traffic Prediction
```python
response = await god.query(
    "What will be the traffic like at Sector 18 at 6 PM today?"
)
```

### Route Planning
```python
route = await god.plan_route(
    source="Noida City Centre Metro",
    destination="Indirapuram Habitat Centre",
    departure_time="18:00",
    preferences={"avoid_tolls": True}
)
```

### Scenario Analysis
```python
response = await god.query(
    "What if we add a new flyover connecting Sector 62 to Film City?"
)
```

### Infrastructure Suggestions
```python
response = await god.suggest_infrastructure(
    area="Sector 62",
    problem="Heavy congestion during peak hours",
    budget=100  # crores
)
```

## 🤝 Integration

### With OVERHAUL Main App
```python
# In app.py
from new_traffic_god import TrafficGod

traffic_god = None

@app.on_event("startup")
async def startup():
    global traffic_god
    traffic_god = await create_traffic_god()

@app.post("/traffic/query")
async def traffic_query(query: str):
    response = await traffic_god.query(query)
    return response.to_dict()
```

## 📄 License

Part of the OVERHAUL project. See main LICENSE file.

## 🙏 Acknowledgments

- OpenStreetMap for road network data
- Noida Authority for traffic planning data
- DMRC for metro schedule information

---

**Built with ❤️ for smarter urban mobility in India**
