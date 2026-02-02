# Epic Definition: Credit Risk Prediction System

## Epic Overview

**Epic Name**: Credit Risk Prediction System  
**Epic ID**: CP-001  
**Created**: February 2, 2026  
**Status**: In Progress  
**Priority**: High  
**Team**: Capstone Project Team

---

## Epic Statement

As a **financial institution**, we want to **develop an AI-powered credit risk prediction system** so that we can **make fair, accurate, and explainable lending decisions for applicants with varying levels of credit history**.

---

## Epic Description

Our goal is to create a machine learning model that analyzes loan applicant information and predicts the probability of loan default. We're working with a comprehensive dataset from the 2024 Kaggle "Home Credit Credit Risk Model Stability" competition, which contains 68 different data tables and approximately 1.5 million loan records.

The model will analyze multiple dimensions of applicant information:
- **Demographics**: Age, employment, family status
- **Financial Information**: Income, assets, liabilities
- **Credit History**: Past loans, credit bureau records, payment behavior
- **Application Data**: Loan amount, purpose, terms
- **Behavioral Patterns**: Transaction history, banking relationships

By leveraging advanced machine learning techniques (LightGBM, XGBoost, CatBoost) and modern interpretability tools (SHAP), we aim to build a system that not only predicts risk accurately but also explains why certain applicants are flagged as high-risk.

---

## Business Problem

### The Lender's Dilemma

Financial institutions face a critical balancing act:

**Scenario 1: Too Conservative**
- Reject many loan applications to minimize defaults
- **Result**: Miss out on creditworthy customers → Lost revenue opportunities
- **Impact**: Reduced market share, customer dissatisfaction

**Scenario 2: Too Lenient**
- Approve most applications to maximize lending volume
- **Result**: High default rates → Financial losses, increased bad debt
- **Impact**: Profit erosion, regulatory scrutiny

### The Credit Invisibility Problem

**Challenge**: Approximately 26 million Americans (and billions globally) are "credit invisible" — they lack sufficient credit history to generate traditional credit scores.

**Current Impact**:
- Automatic rejection despite potentially responsible financial behavior
- Exclusion from financial services
- Perpetuation of financial inequality

**Our Approach**:
- Utilize alternative data sources (bank transactions, payment histories, application data)
- Focus on behavioral patterns rather than historical credit scores
- Enable financial inclusion while maintaining risk management

---

## Business Value

### Quantifiable Benefits

1. **Reduced Default Rate**
   - Target: Reduce false negatives by 15-20%
   - Expected impact: Millions in prevented losses

2. **Increased Approval Rate for Creditworthy Applicants**
   - Target: Increase true positives by 10-15%
   - Expected impact: Expanded customer base, increased revenue

3. **Operational Efficiency**
   - Automated risk assessment
   - Faster decision-making (minutes vs. days)
   - Reduced manual review workload

4. **Regulatory Compliance**
   - Explainable AI satisfies regulatory requirements
   - Audit trail for lending decisions
   - Fair lending documentation

### Intangible Benefits

1. **Financial Inclusion**: Enable access to credit for underserved populations
2. **Customer Trust**: Transparent, fair decision-making process
3. **Competitive Advantage**: Modern AI-driven approach to credit risk
4. **Data-Driven Culture**: Build organizational ML capabilities

---

## Success Criteria

### Technical Metrics

| Metric | Baseline | Target | Stretch Goal |
|--------|----------|--------|--------------|
| ROC-AUC Score | 0.50 (random) | 0.75+ | 0.80+ |
| Precision (Default) | TBD | 0.70+ | 0.75+ |
| Recall (Default) | TBD | 0.65+ | 0.70+ |
| Model Explainability | None | SHAP implemented | Dashboard |

### Business Metrics

| Metric | Current State | Target State |
|--------|--------------|--------------|
| Manual Review Time | 2-3 days | < 1 hour |
| Default Rate | Industry avg ~5% | Reduced by 15% |
| Approval Rate (Creditworthy) | Baseline | +10% approvals |
| Model Confidence | N/A | 85%+ decisions with >70% confidence |

### Acceptance Criteria

✅ Model trained on 1.5M+ loan applications  
✅ All 68 data tables successfully integrated  
✅ Missing data strategy implemented and validated  
✅ Multiple models compared (LR, LightGBM, XGBoost, CatBoost)  
✅ ROC-AUC score > 0.75 on held-out test set  
✅ SHAP explanations generated for all predictions  
✅ Documentation complete for all processes  
✅ Reproducible pipeline from raw data to predictions  

---

## Scope

### In Scope

✅ Data exploration and quality assessment  
✅ Data merging and preprocessing (68 tables)  
✅ Feature engineering and aggregation  
✅ Missing value imputation strategy  
✅ Class imbalance handling (SMOTE, class weights)  
✅ Model training (Logistic Regression, LightGBM, XGBoost, CatBoost)  
✅ Hyperparameter tuning  
✅ Model evaluation (ROC-AUC, Precision, Recall, F1)  
✅ Model interpretability (SHAP, feature importance)  
✅ Prediction generation for test set  
✅ Comprehensive documentation  

### Out of Scope

❌ Real-time API deployment  
❌ Production infrastructure setup  
❌ User interface development  
❌ Integration with existing loan origination systems  
❌ A/B testing in production environment  
❌ Ongoing model monitoring and retraining  
❌ Data collection from new sources  

### Future Considerations

- **Phase 2**: Deploy model as REST API service
- **Phase 3**: Build decision support dashboard for loan officers
- **Phase 4**: Implement automated model retraining pipeline
- **Phase 5**: Expand to additional credit products (credit cards, personal loans)

---

## Stakeholders

### Primary Stakeholders

| Role | Name/Department | Interest | Influence |
|------|----------------|----------|-----------|
| Project Sponsor | Academic Advisor | Project success, learning outcomes | High |
| End Users | Credit Analysts | Usable risk scores, clear explanations | Medium |
| Data Owner | Kaggle/Home Credit | Data usage compliance | Low |

### Secondary Stakeholders

- **Loan Officers**: Will use predictions for decision-making
- **Compliance Team**: Ensure regulatory adherence
- **IT Department**: Potential future deployment support
- **Business Analysts**: Use insights for strategy
- **Loan Applicants**: Benefit from fair assessment

---

## Risks and Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| Data quality issues | High | High | Comprehensive data validation, robust imputation |
| Model overfitting | Medium | High | Cross-validation, regularization, early stopping |
| Class imbalance issues | High | Medium | SMOTE, class weights, stratified sampling |
| Computational constraints | Medium | Medium | Use efficient algorithms, cloud resources |
| Feature engineering complexity | Medium | Medium | Iterative approach, domain research |

### Business Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| Model bias against protected groups | Medium | High | Fairness analysis, disparate impact testing |
| Lack of model interpretability | Low | High | SHAP implementation, feature importance |
| Insufficient model performance | Medium | High | Multiple model comparison, ensemble methods |
| Timeline delays | Medium | Medium | Agile sprints, regular progress reviews |

---

## Dependencies

### Data Dependencies
- Kaggle dataset availability and download permissions
- Data storage capacity (~10GB+ required)
- Parquet/CSV file reading capabilities

### Technical Dependencies
- Python 3.8+ environment
- Machine learning libraries (scikit-learn, LightGBM, XGBoost)
- Sufficient computational resources (16GB+ RAM)
- SHAP library for model explanations

### Team Dependencies
- All team members available and committed
- Knowledge sharing and collaboration
- Code review and quality assurance

---

## Linked User Stories

1. **US-001**: Risk Score Prediction (7 tasks)
2. **US-002**: Understanding Predictions (5 tasks)
3. **US-003**: Combining Data Tables (5 tasks)
4. **US-004**: Handling Missing Data (5 tasks)
5. **US-005**: Model Evaluation (5 tasks)
6. **US-006**: Feature Importance (5 tasks)
7. **US-007**: Fair Assessment (5 tasks)
8. **US-008**: Documentation (5 tasks)

**Total**: 8 User Stories, 42 Tasks

---

## Timeline

| Milestone | Date | Deliverables |
|-----------|------|--------------|
| **Milestone 1**: Project Approval | Completed | Concept approved |
| **Milestone 2**: Technical Planning | Feb 2, 2026 | Epic, user stories, technical plan |
| **Milestone 3**: Data Ready | Week of Feb 9 | Cleaned, merged dataset |
| **Milestone 4**: Features Ready | Week of Feb 16 | Feature engineering complete |
| **Milestone 5**: Models Trained | Week of Feb 23 | All models compared |
| **Milestone 6**: Evaluation Complete | Week of Mar 2 | Final metrics, SHAP analysis |
| **Milestone 7**: Documentation | Week of Mar 9 | Complete documentation |
| **Final Presentation** | Mid-March 2026 | Project demo and results |

---

## Budget and Resources

### Human Resources
- 5 team members × 8 weeks = 40 person-weeks effort
- Estimated 15-20 hours per person per week

### Computational Resources
- Local development machines (existing)
- Potential cloud compute if needed (budget: $0-100)
- Storage: 10-20GB (local drives)

### Software/Tools
- All open-source tools (no licensing costs)
- Kaggle account (free)
- GitHub repository (free)
- Jira board (free tier)

**Total Estimated Budget**: $0-100

---

## Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Project Sponsor | TBD | _________ | _____ |
| Team Lead | Venkat Dinesh | _________ | 02/02/2026 |
| Technical Lead | Sai Charan | _________ | 02/02/2026 |

---

**Document Version**: 1.0  
**Last Updated**: February 2, 2026  
**Next Review**: February 9, 2026
