# FPT Cost Brain

Enterprise R&D Cost Estimation System using AI/ML for predicting project costs from Product Request (PR) documents.

## Problem Statement

FPT Industrial's R&D department faces a critical challenge: accurately estimating costs for new product development projects. Traditional estimation methods rely heavily on expert judgment, leading to:

- High variance in estimates (projects range from 50K to 5M euros)
- Time-consuming manual analysis of technical specifications
- Difficulty in leveraging historical project data systematically
- Inconsistent cost breakdowns across different estimators

This system addresses these challenges through a hybrid approach combining machine learning, case-based reasoning, and large language models.

## What This System Does

FPT Cost Brain automates the cost estimation workflow:

1. **Intake**: Parses Product Request Excel files to extract technical specifications
2. **Feature Extraction**: Converts PR specifications into 27 ML-ready features
3. **Prediction**: Uses HCQE model to predict total hours and cost with confidence intervals
4. **Breakdown Generation**: Distributes total estimate across PE02 activity categories
5. **Export**: Generates standard FPT PE02 PowerPoint/Excel reports

The system operates as a decision support tool, providing engineers with data-driven estimates that can be reviewed and adjusted.

## System Architecture

The system follows a three-tier architecture with an agentic orchestration layer:

```
                                 User Interface
                                      |
                    +-----------------+-----------------+
                    |                                   |
              Next.js Frontend                    REST API
              (Estimation Wizard)                 (FastAPI)
                    |                                   |
                    +-----------------+-----------------+
                                      |
                            Agentic Pipeline
                    +-------------------------------------+
                    |                                     |
                    |   +-----------+   +-----------+    |
                    |   | HCQE      |   | Similar   |    |
                    |   | Predictor |   | PR Search |    |
                    |   +-----------+   +-----------+    |
                    |         |               |          |
                    |         v               v          |
                    |       +-------------------+        |
                    |       |    Arbitrator     |        |
                    |       | (Decision Engine) |        |
                    |       +-------------------+        |
                    |                 |                  |
                    +-------------------------------------+
                                      |
                    +-----------------+-----------------+
                    |                 |                 |
              PostgreSQL          Qdrant            Redis
              (Main DB)         (Vectors)          (Cache)
```

### Component Responsibilities

**HCQE Predictor**: Machine learning model that predicts total hours and cost from PR features. Uses hierarchical sizing classification followed by sizing-specific quantile regression.

**Similar PR Search**: Retrieves historically similar projects using vector embeddings. Provides context for the arbitrator and enables case-based reasoning adjustments.

**Arbitrator**: Decision engine that combines ML prediction with historical context. Determines whether to use HCQE prediction directly, adjust based on similar PRs, or flag for manual review.

## ML Model: HCQE v8.2

HCQE (Hierarchical Conformal Quantile Ensemble) is a 4-stage prediction pipeline designed for high-variance cost estimation:

```
Input Features (27)
       |
       v
+------------------+
| Sizing Classifier|  --> Categorizes into 5 size classes
| (97% accuracy)   |      (X-Small, Small, Medium, Large, Full)
+------------------+
       |
       v
+------------------+
| Sizing-Specific  |  --> Separate model per size class
| Quantile Models  |      Predicts Q10, Q50, Q90
+------------------+
       |
       v
+------------------+
| Multi-Output     |  --> Predicts Hours and Cost
| Regression       |      with consistency constraints
+------------------+
       |
       v
+------------------+
| Conformal        |  --> Calibrates prediction intervals
| Calibration      |      to achieve 90% coverage
+------------------+
       |
       v
Output: {hours, cost, confidence_interval}
```

### Model Performance (LOOCV, n=33)

| Metric            | Hours   | Cost  |
| ----------------- | ------- | ----- |
| Within 30%        | 66.7%   | 57.6% |
| MAE               | 2,237 h | 619 K |
| Sizing Accuracy   | 97.0%   | 97.0% |
| Interval Coverage | 90.8%   | 90.8% |

### Input Features

The model uses 27 features extracted from PR documents:

| Category       | Features                                                 | Count |
| -------------- | -------------------------------------------------------- | ----- |
| Change Flags   | ATS, calibration, hardware, software changes             | 4     |
| Power/Torque   | Power increase (kW), torque increase (Nm)                | 2     |
| Complexity     | Number of functions, emissions level, affected functions | 3     |
| Product Family | E0N0, E5F0, E0C0, E8S0, E0V0 encodings                   | 5     |
| ATS Components | DOC, SCR, DPF presence                                   | 3     |
| Project Flags  | New engine, BOM, homologation, CE marking                | 4     |
| Sizing Scores  | PE base, system, install, program scores                 | 4     |
| Other          | Application type, RD type                                | 2     |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenRouter API key (for LLM features)

### Installation

```bash
git clone <repository-url>
cd fpt-cost-brain

cp env.example .env
# Add OPENROUTER_API_KEY to .env

docker-compose up -d
docker-compose exec backend alembic upgrade head
```

### Access Points

- Frontend: http://localhost:3000
- API Documentation: http://localhost:8000/docs
- Qdrant Dashboard: http://localhost:6333/dashboard

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
├── docs/                       # Documentation
│   ├── architecture/           # System architecture
│   ├── research/               # ML research & IEEE papers
│   └── figures/                # Visualization assets
│
├── data/                       # Data files
├── data_prepared/              # Processed ML/RAG data
│   ├── ml_training/            # Training datasets
│   └── rag_knowledge/          # Knowledge base
│
├── Dataset/                    # Historical PR Excel/PPTX files
├── scripts/                    # Research & experiment scripts
├── models/                     # Benchmark results
│
├── docker-compose.yml          # Docker orchestration
├── .env                        # Environment variables
└── archive/                    # Historical versions
    ├── v1-prototype/           # Streamlit prototype
    └── scripts-history/        # Old script versions
```

## Ablation Study

Systematic analysis of component contributions:

| Experiment                     | Within 30% | Delta  | Significance         |
| ------------------------------ | ---------- | ------ | -------------------- |
| Full Model (Gradient Boosting) | 39.4%      | -      | Baseline             |
| Remove Complexity Features     | 18.2%      | -21.2% | Critical             |
| Remove Change Flags            | 30.3%      | -9.1%  | Important            |
| Remove Power/Torque            | 30.3%      | -9.1%  | Important            |
| Remove Product Family          | 36.4%      | -3.0%  | Minor                |
| Linear Regression (baseline)   | 15.2%      | -24.2% | Architecture matters |

Key finding: Complexity features (num_functions, emissions_level, num_affected_functions) are the most critical predictors, consistent with domain knowledge that project complexity drives cost.

## Technology Stack

| Component | Technology                           | Purpose            |
| --------- | ------------------------------------ | ------------------ |
| Frontend  | Next.js 15, React 19, TailwindCSS v4 | User interface     |
| Backend   | FastAPI 0.115+, Python 3.12          | REST API           |
| ML        | scikit-learn, MAPIE                  | Prediction models  |
| Agents    | LangChain 0.3, LangGraph 0.2         | Orchestration      |
| Database  | PostgreSQL 16                        | Persistent storage |
| Vector DB | Qdrant                               | Similarity search  |
| Cache     | Redis 7                              | Session management |
| LLM       | OpenRouter (DeepSeek, Gemini)        | Text generation    |

## Development

### Running Experiments

```bash
# Ablation study
python scripts/ablation_study.py

# Generate IEEE figures
python scripts/generate_ieee_figures.py

# Benchmark agentic system
python scripts/benchmark_agentic_system.py

# Train HCQE model
python scripts/train_hcqe_autonomous.py
```

### Running Tests

```bash
cd backend && pytest
cd frontend && npm test
```

## Key Scripts

| Script                                | Purpose                     |
| ------------------------------------- | --------------------------- |
| `scripts/ablation_study.py`           | Feature importance analysis |
| `scripts/generate_ieee_figures.py`    | Publication-quality figures |
| `scripts/benchmark_agentic_system.py` | System component analysis   |
| `scripts/train_hcqe_autonomous.py`    | Train production model      |

## Limitations

- Small training dataset (33 samples with complete data)
- Domain-specific to FPT Industrial R&D projects
- Breakdown distribution uses rule-based allocation, not ML prediction
- Requires manual review for edge cases (flagged by system)

## Future Work

- Expand training dataset through continued data collection
- Function-level prediction when sample size exceeds 100
- Transfer learning to other industrial domains
- A/B testing framework for continuous improvement

## Authors

Advanced Data Science Program (ADSP) Project Team
University of Turin / Politecnico di Torino
