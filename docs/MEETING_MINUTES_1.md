Meeting Minutes 3
Home Credit Credit Risk Model Stability Project

---

When: February 24, 2026 (2:00 PM - 4:15 PM)  
Where: Microsoft Teams (virtual)  
Who showed up: Venkat Dinesh, Sai Charan, Lokesh Reddy, Pranav Dhara, Sunny Sunny  
(Nobody was absent)

---

What This Meeting Was About

Venkat started things off around 2:03 PM by saying we've made some serious progress since last time. We basically went from just planning stuff to actually building and running code, which feels like a big deal.

What we covered:
1. Walking through everything we built since Meeting 2
2. Problems we ran into and how we fixed them
3. Actually showing the pipeline work (live demo)
4. The model results (this was exciting)
5. What's next
6. Who's doing what going forward

---

Progress Since Last Time

The Data Pipeline (All Done ✅)

Venkat and Lokesh walked through all the data processing stuff we built. After we finished the planning in Milestone 2, we just jumped in and started writing the actual scripts.

Step 1: Loading the Data
This was straightforward. We loaded all 32 parquet files from Kaggle. Built a DataLoader class that's pretty smart about memory. Here's what we got:
- 1,526,659 loan records (that's a lot)
- Started at 72.8 MB in memory, got it down to 17.5 MB (saved 76%)
- Lokesh mentioned this only took 2 hours instead of the 4 we thought it would

Done: Yes  
Saved as: step1_base_collected.parquet

Step 2: Merging Everything Together
This was trickier. We had to combine all 68 tables into one big table we could actually use.
- Some tables were simple (one row per person, just join them)
- Other tables had multiple rows per person, so we had to summarize them first
- Final table: 1,526,659 rows × 376 columns

Venkat said figuring out how the tables related to each other was the hard part. Sai mentioned we built a monitoring script that showed progress in real-time, which was super helpful because this took a while to run.

Done: Yes  
Saved as: step2_data_merged.parquet

Step 3: Cleaning Up the Data
This is where we dealt with all the missing values and junk.
- Looked at how much data was missing in each column
- Dropped 140 columns that were mostly empty (80%+ missing)
- Filled in missing numbers with the median value
- Filled in missing categories with the most common one
- Checked for duplicates (didn't find any)
- End result: 1,526,659 rows × 236 columns with ZERO missing values

Pranav said this script needed a couple rewrites because Pandas 2.0 changed how things work, but we got it figured out.

Done: Yes  
Saved as: step3_data_cleaned.parquet

Side note: Sunny asked if dropping 140 columns was throwing away useful info. Lokesh pulled up the analysis showing those columns were basically 80%+ blank, so they weren't going to help the model anyway.

---

### 2.2 Feature Engineering & Model Preparation (Completed)

**Presented by:** Sai Charan & Pranav Dhara

Sai walked the team through the feature engineering process, which he said was more straightforward than expected once the data was cleaned.

#### Step 4: Feature Engineering
- Separated features (X) and target variable (y)
- Handled categorical variables:
  - Dropped date columns (too high cardinality)
  - Applied one-hot encoding to remaining categorical features
  - Final feature count after encoding: 245 features
- Created stratified train-test split (80/20)
  - Training set: 1,221,327 samples
  - Test set: 305,332 samples
- Applied StandardScaler to numerical features
- Preserved class distribution in both sets (96.86% class 0, 3.14% class 1)

Sai mentioned they decided NOT to use SMOTE for oversampling because it would be too memory-intensive. Instead, they opted to use class weights in the models to handle the imbalance, which worked really well.

Status: ✅ Completed  
Outputs: 
- step4_X_train.parquet
- step4_X_test.parquet  
- step4_y_train.parquet
- step4_y_test.parquet
- scaler.pkl

Pranav added that they also saved the scaler and numerical column list so they can apply the same transformations to new data in the future.

Training the Models (The Fun Part! ✅)

Sai and Lokesh showed off the model results. This is where everyone got pretty excited even though we were just on a video call.

Step 5: Actually Training Models

What we built:

Model 1: Logistic Regression
- This was our baseline (the simplest thing we could try)
- Used balanced class weights to handle the imbalance
- Trained super fast (like 3 seconds)

Model 2: LightGBM
- This is our main model (gradient boosting, fancy stuff)
- Way more memory-efficient than other options
- Took about 2 minutes to train
- We used 20% of the data (stratified sample) because using everything was taking forever

Sai explained they tried the full dataset first but it was just taking way too long. The 20% sample kept the same ratio of defaults to non-defaults, and the model learned the patterns just fine.

Note on XGBoost: We tried it but ran into memory problems. Decided to just stick with what works (Logistic Regression and LightGBM).

Done: Yes  
Saved:
- logistic_regression.pkl
- lightgbm.pkl

Step 6: Model Evaluation

Lokesh took over to present the evaluation results, and this is where things got really interesting.

Model Performance Comparison:

| Model | AUC-ROC | F1-Score | Accuracy | Precision | Recall |
|-------|---------|----------|----------|-----------|--------|
| LightGBM | 0.8030 | 0.1476 | 74.74% | 0.0826 | 69.58% |
| Logistic Regression | 0.5000 | 0.0610 | 3.14% | 0.0314 | 100.00% |

Confusion Matrix - LightGBM (Best Model):
```
                    Predicted
                    No Default   Default
Actual  No Default    166,143    55,657
        Default         2,190     5,009
```

Key Metrics Explained by Lokesh:
- True Negatives: 166,143 (correctly identified non-defaulters)
- True Positives: 5,009 (correctly caught defaulters)
- False Positives: 55,657 (flagged as risky but actually okay)
- False Negatives: 2,190 (missed defaulters - the dangerous ones)

Winner: LightGBM with 0.8030 AUC-ROC score

Lokesh explained that an AUC-ROC of 0.8030 is actually quite good for this type of problem, especially considering the severe class imbalance. It means the model can distinguish between defaulters and non-defaulters 80% better than random guessing.

Discussion: 

Venkat asked about the low F1-score (0.1476). Lokesh explained that F1-score is heavily impacted by the class imbalance. While the model catches ~70% of actual defaulters (good recall), many non-defaulters also get flagged as risky (lower precision), which is expected given the 31:1 imbalance.

Sunny raised an important business question: "What about those 2,190 false negatives? Those are the ones who will default but we're approving." 

Lokesh responded that this is the inherent trade-off in credit risk - you can never catch 100% of defaulters without also rejecting many good customers. The bank needs to decide their risk tolerance. The model gives them a probability score, and they can adjust the threshold based on their risk appetite.

Status: ✅ Completed  
Outputs:
- step6_model_comparison.csv
- step6_evaluation_results.json
- step6_best_model.txt

---

Problems We Hit and How We Fixed Them

Venkat said, "Let's be real about what didn't work on the first try, because that's where we actually learned stuff."

Problem 1: Memory Blowing Up
Who hit it: Sai

What happened: When we tried merging all tables at once, everything crashed with "Memory Error: Requested 3.21 GB". Just ran out of RAM.

Why: Loading too many columns at the same time

How we fixed it:
- Processed tables in batches (50 at a time) instead of all at once
- Used `pd.concat()` instead of adding columns one by one (prevents fragmentation)
- Changed data types to smaller ones (like using float32 instead of float64)
- Added a progress bar so we could see what was happening

After these changes, it worked fine and we could watch the progress in real-time.

Problem 2: Pandas Being Weird
Who hit it: Pranav

What happened: We were filling in missing values but they were still there after running the code. The fillna() wasn't actually filling anything.

Why: Pandas 2.0 changed how things work. It doesn't modify data directly anymore unless you explicitly tell it to.

How we fixed it:
Changed from this (doesn't work):
```python
df[col].fillna(df[col].median(), inplace=True)
```

To this (works):
```python
df[col] = df[col].fillna(df[col].median())
```

Pranav said it was annoying at first because the code looked right. Once we figured out Pandas changed, it made sense.

Problem 3: XGBoost Taking Forever
Who hit it: Lokesh

What happened: XGBoost training was taking over 30 minutes and eating up way too much memory.

How we dealt with it:
- Just decided to skip XGBoost for now
- LightGBM is working great and uses way less memory
- If we need XGBoost later, we can run it on cloud resources
- LightGBM delivered awesome results anyway (0.8030 AUC-ROC)

Lokesh said LightGBM uses a smarter algorithm that's more efficient, which is why it's faster on big datasets.

Problem 4: Dealing with Imbalance
Who hit it: Sai

What happened: We planned to use SMOTE (creates fake samples to balance classes), but with 1.2M samples, that would create millions more fake ones and blow up memory.

How we fixed it:
- Used class weights instead - tells the model "pay more attention to defaults"
- Way more memory-efficient (no extra data created)
- Got great results without making fake data
- Set `class_weight='balanced'` in our models

Sai said this is actually better for big datasets anyway, and looking at our results, it definitely worked.

---

Live Demo

Venkat shared his screen and walked through everything:

1. Showed the `data_processed/` folder - all our intermediate files are there
2. Ran `python scripts/step6_model_evaluation.py` live to show it actually works
3. Opened the `models/` folder with our saved models
4. Showed the JSON file with all the detailed results
5. Pulled up the CSV comparing both models

The cool part: anyone on the team can run the scripts in order and get the exact same results. Venkat said "This is what proper MLOps looks like."

---

Documentation and Code Quality

Pranav talked about all the docs we've been writing:

What we documented:
- ✅ PROJECT_SUMMARY_LAYMAN.md - explained everything in non-tech language (Pranav said even his roommate who isn't in tech could understand it)
- ✅ README_WORKFLOW.py - step-by-step how to run everything
- ✅ Comments in all our code
- ✅ Proper function documentation
- ✅ Decent git commit messages

Code quality stuff:
- Kept the same style across all scripts
- Made things modular so we can reuse code
- Added error checks so things don't silently break
- Progress bars for stuff that takes a while

Sunny said from a testing perspective, the code is pretty clean and easy to check.

---

What's Going Well

Venkat asked everyone to say one thing that's working. Here's what people said:

Venkat: "The way we organized everything is really paying off. We're not hunting for files."

Sai: "Having all our settings in config.py is awesome. Change one thing and it updates everywhere."

Lokesh: "Those data quality tools we built at the start saved us so much time. We knew what we were dealing with upfront."

Pranav: "Team communication is solid. Someone gets stuck, we hop on a call and fix it."

Sunny: "The documentation doesn't suck! Usually docs are an afterthought but ours actually help."

---

What Could Be Better

Venkat also asked what we could improve:

Lokesh: "We should test edge cases better. That Pandas fillna thing could've been caught sooner."

Sai: "Training takes kind of long. Maybe we can speed things up somehow."

Sunny: "We don't have automated tests. We're just manually checking everything."

Pranav: "Our git workflow could be cleaner. Sometimes people forget to pull before starting."

Venkat: "Fair points. Let's work on these next sprint."

---

How We're Doing On Our Goals

Lokesh showed where we are vs. where we wanted to be:

The Numbers

| What We Wanted | Target | What We Got | Did We Hit It? |
|----------------|--------|-------------|----------------|
| AUC-ROC Score | 0.75+ | 0.8030 | ✅ Beat it |
| Train 3+ models | 3 models | 2 models (LR, LightGBM) | 🟡 Close enough |
| Merge all tables | All merged | Done | ✅ Yes |
| Handle missing data | Good strategy | 0 missing values | ✅ Yes |
| Reproducible pipeline | Automated | All scripts done | ✅ Yes |

Process Stuff

| Thing | Status |
|-------|--------|
| Documentation | ✅ Done and actually useful |
| Code Quality | ✅ Clean and modular |
| Git Usage | ✅ Using it regularly |
| Team Communication | ✅ Talking often |

Lokesh said beating the AUC-ROC target by 7% is actually a big deal - means the model's better than we hoped.

---

Risks We're Watching

Venkat talked about things that could become problems:

Risk 1: XGBoost Not Done
- How bad: Medium
- What it means: We're missing one model comparison
- What we're doing: LightGBM is working great, we can add XGBoost later if we have time
- Status: We're okay with this

Risk 2: No SHAP Yet
- How bad: Medium
- What it means: Can't explain why the model makes certain predictions
- What we're doing: Scheduled for next sprint
- Status: Tracked, not a blocker right now

Risk 3: Not Enough Testing
- How bad: Low
- What it means: Might have bugs we haven't found
- What we're doing: Add proper tests next sprint
- Status: On the list

Risk 4: Only Running on One Computer
- How bad: Low
- What it means: Can't easily scale to way bigger data
- What we're doing: Current data fits fine, we documented how to move to cloud if needed
- Status: Fine for now

Venkat said overall nothing's critical and we're managing risks pretty well.

---

What's Next

Here's what we need to do in the next couple weeks:

Top Priorities

Priority 1: Make the Model Explainable (SHAP stuff)
Who's doing what:
- Lokesh: Get SHAP calculations working (5 hours)
- Sai: Make visualizations showing feature importance (4 hours)
- Pranav: Create example explanations for some predictions (4 hours)
- Venkat: Write up what the features actually mean (3 hours)

Due: March 8

Priority 2: Make Nice Charts
Who's doing what:
- Sai: ROC curves (3 hours)
- Lokesh: Confusion matrix as a heatmap (2 hours)
- Pranav: Feature distribution plots (4 hours)
- Sunny: Model comparison charts (3 hours)

Due: March 10

Priority 3: Finish All Docs
Who's doing what:
- Venkat: Update the README with final numbers (2 hours)
- Everyone: Help make the presentation slides (6 hours)
- Venkat & Sai: Record a demo video (4 hours)
- Lokesh: Get Kaggle submission ready (3 hours)

Due: March 15

If We Have Time
- Try to get XGBoost working
- Add automated tests
- Build an API for predictions
- Deploy to cloud

---

Questions People Had

Sunny asked: "What if we find a bug in the preprocessing after models are trained?"

Venkat answered: "We'd just rerun from Step 3 onward. That's why we made it modular - can restart from wherever. The parquet files make it fast."

Pranav asked: "Should we retrain on the full dataset instead of the 20% sample?"

Sai answered: "For final submission, yeah. But for now, the sample works fine for testing stuff. We'll do a full training run before we submit."

Lokesh asked: "Are we actually submitting this to Kaggle?"

Venkat answered: "Yep, that's the plan! We'll make predictions on their test set and submit. Even if we don't win, it's cool to see how we rank."

---
Quick Summary of Action Items

| What Needs Doing | Who's On It | When | How Important |
|------------------|-------------|------|---------------|
| SHAP analysis | Lokesh | Mar 8 | High |
| Visualization plots | Sai | Mar 10 | High |
| ROC curves | Sai | Mar 10 | High |
| Confusion matrix heatmap | Lokesh | Mar 10 | Medium |
| Explain features | Venkat | Mar 8 | High |
| Update docs | Venkat | Mar 15 | High |
| Presentation slides | Everyone | Mar 15 | High |
| Retrain on full data | Sai | Mar 12 | High |
| Kaggle submission | Lokesh | Mar 15 | Medium |
| Add tests | Sunny | Mar 18 | Low |

---

Important Dates

- Next Meeting: March 10, 2:00 PM (Sprint Demo & Final Review)
- Project Demo: March 15 (exact time TBD)
- Final Submission: March 20

---

How the Meeting Went

Good stuff:
- Actually showed demos, not just talked about it
- People were honest about problems
- Everyone chimed in
- Team's in a good mood

Could be better:
- Went 15 minutes over (should've booked more time)
- Could've sent results ahead of time

Shoutouts:
- Sai for the model training
- Lokesh for explaining metrics clearly
- Pranav for fixing the Pandas stuff
- Sunny for asking good questions
- Venkat for keeping us organized

---

Venkat's Closing Comments

"Just want to say thanks for everyone's hard work. Few weeks ago we had 68 tables of messy data and weren't sure if this would work. Now we have a real ML model with 0.80 AUC-ROC that actually predicts loan defaults.

We built a complete pipeline from raw data to predictions. Dealt with memory issues, figured out Pandas weirdness, and made smart trade-offs. Most important - we helped each other out when someone got stuck.

Home stretch now. Let's finish strong with the explainability stuff and final docs. Two more weeks and we'll have something we're all proud of."

Everyone agreed and gave positive feedback.

---

Meeting ended: 4:16 PM

These minutes written by: Venkat Dinesh  
Date: February 24, 2026  
Next meeting: March 10, 2026

---

Useful Links

- GitHub: https://github.com/Venkatdinesh20/Capstone_Project
- Jira: https://venkatdinesh60.atlassian.net/jira/core/projects/CP/board
- Docs folder: /docs
- Model files: /models
- Results: /outputs/reports

---

How Much We Got Done

Sprint 2 stats:
- Story points done: 26
- Tasks finished: 15
- Estimated time: 78 hours
- Actual time: ~72 hours (came in under!)
- Velocity: Good pace, bit ahead of schedule