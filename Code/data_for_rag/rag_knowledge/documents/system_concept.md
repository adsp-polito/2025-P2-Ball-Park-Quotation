# FPT Cost Brain - System Concept

## Purpose
AI-powered ballpark quotation tool for FPT R&D cost estimation.

## Process Flow

### Step 1: PR Intake
- Upload Product Request (PR) Excel file
- System parses and validates PR content
- Extract key metadata: Platform, Engine, Tier, Plant, Region

### Step 2: Feature Extraction
- **Automatic Features**: Extracted directly from PR text
  - Hardware changes (turbo, injectors, fuel rail, EGR, cooling)
  - Software/calibration changes
  - ATS (After Treatment System) modifications
  - Installation requirements
  
- **Suggested Features**: System recommends based on similar PRs
- **Missing Features**: Identified gaps requiring human input

### Step 3: ML Prediction
- Ensemble model combining:
  - Gradient Boosting (main predictor)
  - Neural Network (non-linear patterns)
  - Gaussian Process (uncertainty)
  - KNN (similar project matching)
  - LLM context (reasoning)

### Step 4: Output Generation
- Cost breakdown by function
- Confidence intervals
- Similar project references
- Risk factors

### Step 5: Human Review
- Engineer reviews and adjusts estimates
- Corrections feed back into learning system

## Data Sources
- Historical PRs (48 unique)
- R&D Output records (761 line items)
- 50 defined features
- 35 project clusters
- 45 sizing rules
- 22 product types
