# User Stories: Credit Risk Prediction System

## Overview

This document contains all user stories for the Credit Risk Prediction System project. Each story follows the standard format: "As a [role], I want [feature], so that [benefit]."

**Epic**: Credit Risk Prediction System  
**Total Stories**: 8  
**Total Tasks**: 42  
**Status**: Sprint 1 - In Progress  
**Last Updated**: February 2, 2026

---

## User Story 1: Risk Score Prediction

**Story ID**: US-001  
**Priority**: High  
**Story Points**: 13  
**Sprint**: Sprint 1-2

### User Story
**As a** credit analyst  
**I want** to see a risk score for each loan application  
**So that** I can focus my attention on the ones that look problematic

### Acceptance Criteria
- [ ] Model generates probability score (0-1) for each application
- [ ] Scores calibrated (predicted probability matches actual default rate)
- [ ] Predictions generated for all test set applications
- [ ] Output includes case_id and predicted default probability
- [ ] ROC-AUC score > 0.75 on validation set

### Tasks

| Task ID | Task Description | Assignee | Status | Est. Hours |
|---------|-----------------|----------|--------|------------|
| 1.1 | Get the dataset downloaded from Kaggle | Venkat Dinesh | To Do | 2h |
| 1.2 | Open the main training file and see how defaults are distributed | Sai Charan | To Do | 3h |
| 1.3 | Divide data into training and testing portions | Lokesh Reddy | To Do | 2h |
| 1.4 | Build a simple logistic regression as starting point | Sunny Sunny | To Do | 4h |
| 1.5 | Build a LightGBM model | Pranav Dhara | To Do | 6h |
| 1.6 | Build an XGBoost model | Sai Charan | To Do | 6h |
| 1.7 | Check which model works best and go with that | Lokesh Reddy | To Do | 4h |

**Total Estimate**: 27 hours

---

## User Story 2: Understanding Predictions

**Story ID**: US-002  
**Priority**: High  
**Story Points**: 8  
**Sprint**: Sprint 2

### User Story
**As a** loan officer  
**I want** to know the reasons behind a high-risk prediction  
**So that** I can have a real conversation with the applicant about it

### Acceptance Criteria
- [ ] SHAP values calculated for all predictions
- [ ] Top 5 contributing features identified per prediction
- [ ] Force plots generated for sample predictions
- [ ] Feature importance ranking available
- [ ] Explanations in non-technical language

### Tasks

| Task ID | Task Description | Assignee | Status | Est. Hours |
|---------|-----------------|----------|--------|------------|
| 2.1 | Pull out feature importance from the model | Sai Charan | To Do | 3h |
| 2.2 | Get SHAP working in our environment | Lokesh Reddy | To Do | 4h |
| 2.3 | Make SHAP plots that show overall feature impact | Pranav Dhara | To Do | 5h |
| 2.4 | Show example explanations for a few predictions | Sunny Sunny | To Do | 4h |
| 2.5 | Write up what the important features actually mean | Venkat Dinesh | To Do | 3h |

**Total Estimate**: 19 hours

---

## User Story 3: Combining Data Tables

**Story ID**: US-003  
**Priority**: Critical  
**Priority**: 13  
**Sprint**: Sprint 1

### User Story
**As a** data engineer  
**I want** all the scattered data tables merged into one usable dataset  
**So that** we can actually train a model on it

### Acceptance Criteria
- [ ] All 68 tables successfully loaded
- [ ] Relationships between tables documented
- [ ] Static tables (1 row per case) merged correctly
- [ ] Dynamic tables (multiple rows) aggregated appropriately
- [ ] No duplicate case_id in final dataset
- [ ] Memory-efficient processing (chunking if needed)

### Tasks

| Task ID | Task Description | Assignee | Status | Est. Hours |
|---------|-----------------|----------|--------|------------|
| 3.1 | Figure out how all 68 tables connect to each other | Lokesh Reddy | To Do | 6h |
| 3.2 | Link the simple one-row-per-person tables to the main table | Venkat Dinesh | To Do | 5h |
| 3.3 | For tables with multiple rows per person, calculate summaries | Sai Charan | To Do | 8h |
| 3.4 | Put everything together into one final table | Pranav Dhara | To Do | 6h |
| 3.5 | Make sure we didn't mess up any joins | Sunny Sunny | To Do | 4h |

**Total Estimate**: 29 hours

---

## User Story 4: Handling Missing Data

**Story ID**: US-004  
**Priority**: High  
**Story Points**: 8  
**Sprint**: Sprint 1

### User Story
**As a** data analyst  
**I want** a solid strategy for dealing with missing information  
**So that** incomplete applications don't break the system

### Acceptance Criteria
- [ ] Missing value analysis report generated
- [ ] Columns with >80% missing values dropped
- [ ] Numerical features imputed with median
- [ ] Categorical features imputed with mode
- [ ] Missingness indicator flags created
- [ ] No NaN values in final training data

### Tasks

| Task ID | Task Description | Assignee | Status | Est. Hours |
|---------|-----------------|----------|--------|------------|
| 4.1 | Count how much data is missing in each column | Pranav Dhara | To Do | 3h |
| 4.2 | Get rid of columns that are mostly empty | Lokesh Reddy | To Do | 2h |
| 4.3 | Fill in missing numbers with the middle value | Sai Charan | To Do | 4h |
| 4.4 | Fill in missing categories with the most common one | Venkat Dinesh | To Do | 4h |
| 4.5 | Add flags showing which values were originally blank | Sunny Sunny | To Do | 3h |

**Total Estimate**: 16 hours

---

## User Story 5: Model Evaluation

**Story ID**: US-005  
**Priority**: High  
**Story Points**: 8  
**Sprint**: Sprint 2

### User Story
**As a** data scientist  
**I want** proper evaluation metrics calculated  
**So that** we can trust the model's predictions

### Acceptance Criteria
- [ ] ROC-AUC calculated on validation and test sets
- [ ] Confusion matrix generated
- [ ] Precision, Recall, F1-Score calculated
- [ ] 5-fold cross-validation completed
- [ ] Results compared across all models
- [ ] Best threshold identified for classification

### Tasks

| Task ID | Task Description | Assignee | Status | Est. Hours |
|---------|-----------------|----------|--------|------------|
| 5.1 | Measure ROC-AUC on held-out data | Sai Charan | To Do | 3h |
| 5.2 | Build a confusion matrix | Lokesh Reddy | To Do | 2h |
| 5.3 | Get precision, recall, and F1 numbers | Venkat Dinesh | To Do | 3h |
| 5.4 | Do cross-validation to check consistency | Sunny Sunny | To Do | 5h |
| 5.5 | Summarize all results in one place | Pranav Dhara | To Do | 4h |

**Total Estimate**: 17 hours

---

## User Story 6: Feature Importance

**Story ID**: US-006  
**Priority**: Medium  
**Story Points**: 8  
**Sprint**: Sprint 2

### User Story
**As a** business analyst  
**I want** to understand which factors drive defaults the most  
**So that** I can report meaningful insights to management

### Acceptance Criteria
- [ ] Feature importance rankings generated
- [ ] Top 20 features identified and analyzed
- [ ] Correlation analysis completed
- [ ] Distribution comparisons (defaulters vs non-defaulters) created
- [ ] Business interpretation document prepared
- [ ] Visualizations created for presentations

### Tasks

| Task ID | Task Description | Assignee | Status | Est. Hours |
|---------|-----------------|----------|--------|------------|
| 6.1 | Dig into the top features with charts and stats | Pranav Dhara | To Do | 5h |
| 6.2 | See how features correlate with defaults | Sai Charan | To Do | 4h |
| 6.3 | Compare feature distributions for defaulters vs non-defaulters | Lokesh Reddy | To Do | 5h |
| 6.4 | Pick the top 20 features that matter most | Sunny Sunny | To Do | 3h |
| 6.5 | Explain what these features mean in business terms | Venkat Dinesh | To Do | 4h |

**Total Estimate**: 21 hours

---

## User Story 7: Fair Assessment

**Story ID**: US-007  
**Priority**: Medium  
**Story Points**: 8  
**Sprint**: Sprint 2-3

### User Story
**As** someone applying for a loan with no credit history  
**I want** the system to consider my actual financial habits  
**So that** I'm not automatically rejected

### Acceptance Criteria
- [ ] Alternative data features created (payment behavior, banking history)
- [ ] Model performance on thin-file applicants measured
- [ ] Comparison with credit-score-only approach
- [ ] Fairness metrics calculated (disparate impact)
- [ ] Feature set includes non-traditional credit indicators

### Tasks

| Task ID | Task Description | Assignee | Status | Est. Hours |
|---------|-----------------|----------|--------|------------|
| 7.1 | Find columns related to payment behavior | Sai Charan | To Do | 4h |
| 7.2 | Create features from past loan application data | Lokesh Reddy | To Do | 6h |
| 7.3 | Create features from credit bureau records | Venkat Dinesh | To Do | 6h |
| 7.4 | Check how the model does on people with thin credit files | Sunny Sunny | To Do | 5h |
| 7.5 | See if our approach beats traditional methods | Pranav Dhara | To Do | 4h |

**Total Estimate**: 25 hours

---

## User Story 8: Documentation

**Story ID**: US-008  
**Priority**: Medium  
**Story Points**: 5  
**Sprint**: Sprint 3

### User Story
**As a** project evaluator  
**I want** everything properly documented  
**So that** future teams can pick up where we left off

### Acceptance Criteria
- [ ] Data cleaning process documented
- [ ] Feature engineering process documented
- [ ] Model training process documented
- [ ] Results and findings documented
- [ ] README.md complete with setup instructions
- [ ] Code comments and docstrings added
- [ ] Jupyter notebooks cleaned and annotated

### Tasks

| Task ID | Task Description | Assignee | Status | Est. Hours |
|---------|-----------------|----------|--------|------------|
| 8.1 | Write up how we cleaned the data | Venkat Dinesh | To Do | 4h |
| 8.2 | Write up how we built features | Sai Charan | To Do | 4h |
| 8.3 | Write up how we trained models | Lokesh Reddy | To Do | 4h |
| 8.4 | Write up our results and findings | Pranav Dhara | To Do | 5h |
| 8.5 | Put together the GitHub README | Sunny Sunny | To Do | 3h |

**Total Estimate**: 20 hours

---

## Summary Statistics

### By Priority
- **Critical**: 1 story (13 story points)
- **High**: 4 stories (37 story points)
- **Medium**: 3 stories (21 story points)

### By Sprint
- **Sprint 1**: Stories 1, 3, 4 (focus on data preparation)
- **Sprint 2**: Stories 2, 5, 6, 7 (focus on modeling and analysis)
- **Sprint 3**: Story 8 (focus on documentation)

### Effort Distribution

| Team Member | Assigned Tasks | Estimated Hours |
|-------------|----------------|-----------------|
| Venkat Dinesh | 8 tasks | 36 hours |
| Sai Charan | 9 tasks | 44 hours |
| Lokesh Reddy | 9 tasks | 44 hours |
| Pranav Dhara | 8 tasks | 41 hours |
| Sunny Sunny | 8 tasks | 35 hours |

**Total**: 42 tasks, ~200 hours

---

## Definition of Done

A user story is considered "Done" when:
- [ ] All tasks completed
- [ ] Code reviewed by at least one team member
- [ ] Unit tests written (where applicable)
- [ ] Documentation updated
- [ ] Acceptance criteria met
- [ ] Demo prepared for stakeholders
- [ ] Merged to main branch

---

## Retrospective Template

After each sprint, we'll review:
1. **What went well?**
2. **What could be improved?**
3. **Action items for next sprint**

---

**Document Version**: 1.0  
**Last Updated**: February 2, 2026  
**Next Review**: End of Sprint 1
