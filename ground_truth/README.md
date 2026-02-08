# Ground Truth Labeling Framework

This framework enables creating human-labeled relevance judgments for evaluating and improving the Academic Matcher system.

## Why Ground Truth Matters

Without human-labeled relevance data, we cannot:
- Know if our system is actually good
- Compare different approaches fairly
- Fine-tune models for our specific domain
- Identify systematic failures

## Relevance Scale

| Score | Label | Definition |
|-------|-------|------------|
| **3** | Highly Relevant | This researcher is an excellent match. Their primary research focus directly addresses the query. You would confidently recommend them. |
| **2** | Relevant | This researcher works in a related area. They have some relevant expertise but it's not their main focus. A reasonable recommendation. |
| **1** | Marginally Relevant | Weak connection. The researcher might have tangential knowledge but wouldn't be a strong recommendation. |
| **0** | Not Relevant | No meaningful connection to the query. Wrong field entirely. |

## Labeling Process

### Step 1: Generate Candidates
```bash
python ground_truth/generate_candidates.py --num-queries 50
```

This creates `labeling_tasks.json` with query-researcher pairs to judge.

### Step 2: Label the Pairs
Open `labeling_tasks.csv` in a spreadsheet and fill in the `relevance_score` column (0-3).

Or use the interactive labeler:
```bash
python ground_truth/interactive_labeler.py
```

### Step 3: Analyze Labels
```bash
python ground_truth/analyze_labels.py
```

### Step 4: Evaluate System
```bash
python ground_truth/evaluate_with_ground_truth.py
```

## File Structure

```
ground_truth/
├── README.md                    # This file
├── generate_candidates.py       # Generate pairs for labeling
├── interactive_labeler.py       # Terminal-based labeling tool
├── analyze_labels.py            # Analyze label distribution
├── evaluate_with_ground_truth.py # Evaluate system with labels
├── labeling_tasks.json          # Generated tasks (machine-readable)
├── labeling_tasks.csv           # Generated tasks (spreadsheet)
└── labels/
    └── ground_truth_v1.json     # Completed labels
```

## Labeling Guidelines

### Do Consider:
- The researcher's **primary** research focus
- Recent publications and projects
- Specific expertise mentioned in their profile

### Don't Consider:
- Whether you personally like the researcher
- Department prestige
- How many results you've already marked as relevant

### Edge Cases:
- **Interdisciplinary queries**: A query about "machine learning for materials" could match both ML experts and materials scientists. Score based on how well they bridge both areas.
- **Broad queries**: "energy research" is broad. Score based on how central energy is to their work.
- **Very specific queries**: If the query is very specific (e.g., "gallium nitride epitaxy"), require a strong match for score 3.

## Target: 200 Labeled Pairs

For meaningful evaluation and potential fine-tuning, aim for:
- 50 unique queries
- 4 researchers judged per query (mix of system results + random)
- = 200 total judgments

This takes approximately 1-2 hours of focused labeling time.
