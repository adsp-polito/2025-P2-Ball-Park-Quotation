# Rules of Engagement (ROE) - FPT Cost Brain

## Overview

The Ballpark Quotation Tool uses machine learning to estimate FPT R&D costs and efforts for CNH product requests. It analyzes historical quotes for similar projects to determine the effort required for new projects, broken down by function. The goal is faster, more accurate quotes that improve competitiveness and strengthen customer trust.

## Business Case Parameters

| Parameter | Value | Unit | Assumption |
|-----------|-------|------|------------|
| Average PR/year | 97 | Nr. | Source: CNH budget file (year 2025) |
| Average Ballpark/year | ~24 | Nr. | 25% of PRs are preceded by a Ballpark request |
| Ballpark preparation lead time | ~45 | man-hours | 45 man-hours per Ballpark |
| Hourly Cost | 90 | EUR/hour | EUR 90 per hour |
| Ballpark success rate | 20% | % | 20% of Ballparks turn into an official PR |
| Average Value of Awarded Project | 150,000 | EUR | Source: CNH budget file (avg R&D 2020-25) |
| Estimation errors | 20% | % | Gap between Ballpark and Final Quote |
| Average Renegotiation Time | ~25 | man-hours | Per renegotiation due to errors |

## Business Case Contributors

1. **Competitiveness Increase**: Reducing preparation time allows more Ballparks to be sent, increasing probability of winning projects.
2. **Internal Efficiency**: Reduction in man-hours translates to direct cost savings.
3. **Estimation Accuracy**: Fewer errors mean less over/under quoting and fewer escalations/renegotiations.

## System Concept

### Input Flow
1. PR from Excel file
2. Feature selection
   - System extracts features from PR
   - System suggests additional features for evaluation
   - System identifies missing features in PR
   - System requests human feedback for:
     - Features suggested
     - Features missing from PR
     - Features to be added manually
   - Human inputs missing features

### ML Algorithm Processing
- Analyze historical data
- Match similar projects
- Generate cost estimates by function

### Output
- Ballpark quotation breakdown
- Confidence scores
- Similar project references

## Key Metrics

### Historical Data Summary (2010-2025)
- Total PRs analyzed: 1,487+
- Average R&D per year: ~186K EUR
- Sectors: AG (Agricultural), CE (Construction Equipment)

### Success Criteria
- Estimation accuracy within 20%
- Preparation time reduction of 50%+
- ROE potential: payback in 7-14 months
