# FPT Cost Brain

Enterprise R&D Cost Estimation System using AI/ML for predicting project costs from Product Request (PR) documents.

## Prerequisites

- Docker and Docker Compose
- OpenRouter API key (for LLM features)

## Getting Started

### 1. Clone and Configure

```bash
git clone <repository-url>
cd fpt-cost-brain

# Copy environment template and configure
cp env.example .env
```

Edit `.env` and add your API key:
```
OPENROUTER_API_KEY=your_api_key_here
```

### 2. Start Services with Docker Compose

```bash
# Build and start all containers (backend, frontend, postgres, qdrant, redis)
docker-compose up -d

# Check if all services are running
docker-compose ps
```

### 3. Database Setup

```bash
# Run database migrations
docker-compose exec backend alembic upgrade head

# Create a new migration (when models change)
docker-compose exec backend alembic revision --autogenerate -m "description"
```

### 4. Access the Application

| Service            | URL                              |
| ------------------ | -------------------------------- |
| Frontend           | http://localhost:3000            |
| API Documentation  | http://localhost:8000/docs       |
| Qdrant Dashboard   | http://localhost:6333/dashboard  |

## Common Commands

```bash
# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f backend

# Restart a service
docker-compose restart backend

# Stop all services
docker-compose down

# Stop and remove volumes (clears database)
docker-compose down -v

# Rebuild containers after code changes
docker-compose up -d --build
```

## Project Structure

```
fpt-cost-brain/
├── backend/                    # FastAPI backend
│   ├── agents/                 # LangGraph agents & nodes
│   ├── api/                    # REST API endpoints
│   ├── ml/                     # ML models (HCQE predictor)
│   ├── models/                 # Trained model files
│   └── scripts/                # Training & benchmark scripts
│
├── frontend/                   # Next.js 15 frontend
│   ├── app/                    # App Router pages
│   ├── components/             # React components
│   └── stores/                 # Zustand state management
│
├── data/                       # Data files
├── data_prepared/              # Processed ML/RAG data
│   ├── ml_training/            # Training datasets
│   └── rag_knowledge/          # Knowledge base
│
├── Dataset/                    # Historical PR Excel/PPTX files
├── scripts/                    # Research & experiment scripts
│
├── docker-compose.yml          # Docker orchestration
└── .env                        # Environment variables
```

## Technology Stack

| Component  | Technology                           |
| ---------- | ------------------------------------ |
| Frontend   | Next.js 15, React 19, TailwindCSS v4 |
| Backend    | FastAPI 0.115+, Python 3.12          |
| ML         | scikit-learn, MAPIE                  |
| Agents     | LangChain 0.3, LangGraph 0.2         |
| Database   | PostgreSQL 16                        |
| Vector DB  | Qdrant                               |
| Cache      | Redis 7                              |
| LLM        | OpenRouter (DeepSeek, Gemini)        |

## Running Tests

```bash
# Backend tests
docker-compose exec backend pytest

# Frontend tests
docker-compose exec frontend npm test
```

## Authors

Advanced Data Science Program (ADSP) Project Team
University of Turin / Politecnico di Torino
