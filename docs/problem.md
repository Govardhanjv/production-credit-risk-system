# Problem statement

Team: Individual submission (Govardhan Vyloppully)

## 1. Who uses the model, and for what decision?

A lender's credit-risk team can use the score to decide whether an application should be approved, sent for manual review, or rejected. The score is one input to the decision; it should not make the final decision by itself.

## 2. The options

Applicants with lower estimated risk can be approved. Applications near a decision boundary, or cases needing more information, can be referred to a reviewer. Applications with high estimated risk can be rejected. A referral takes staff time, so the lender should track review volume and cost before setting a referral range.

## 3. The cost of approving someone who defaults

The task card assigns 10 cost units to approving an applicant who later has payment difficulties.

## 4. The cost of rejecting someone who would have repaid

The task card assigns 1 cost unit to rejecting an applicant who would have repaid. An approval mistake is therefore ten times as costly as a rejection mistake. The lender should be cautious about approving high-risk applications, while using a review option for uncertain cases rather than treating one probability cutoff as certain.

## 5. Success metrics, and why

I will report ROC-AUC, PR-AUC, and KS on the fixed validation set. ROC-AUC measures ranking across thresholds, PR-AUC focuses on finding the less common default class, and KS measures the largest separation between the score distributions for the two classes. Accuracy alone is misleading: a rule that predicts no defaults would be correct for about 92% of applicants while finding no defaulters. The baseline is useful if it beats that trivial ranking behavior on validation without using TARGET as an input or fitting preprocessing on validation data. Expected cost should be considered once decision thresholds and referral costs are defined.

## 6. Risks and sensitive features

CODE_GENDER and DAYS_BIRTH (age can be derived from it) are sensitive characteristics. Other financial, geographic, household, and employment fields may act as proxies. Using these fields can produce different error rates across groups or reproduce existing inequities. The baseline is for analysis, not automatic approval or rejection; group-level performance and applicable lending rules need review before deployment.

## 7. Assumptions

This work assumes TARGET is a consistent indicator of repayment difficulty, that the historical sample is relevant to future applicants, and that the task card's 10-to-1 error-cost ratio is a reasonable starting point. The assumption that historical outcomes represent future applicants matters most; if lending policy or the applicant population changes, both model performance and the estimated costs may change.
