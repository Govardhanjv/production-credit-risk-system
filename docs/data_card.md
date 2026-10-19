# Data card: Home Credit Default Risk (Week 1 sample)

## 1. Source and license terms

The data is from the Home Credit Default Risk Kaggle competition. The competition rules govern its use. The original files and derived sample are kept out of GitHub; `data/` is ignored by this repository. The project uses the data only for this course exercise.

## 2. What each table holds

| Table | One row is... | Links by |
|---|---|---|
| application_train | One applicant's current loan application and target | SK_ID_CURR |
| bureau | One previous credit reported by another financial institution | SK_ID_CURR; individual credit is keyed by SK_ID_BUREAU |
| bureau_balance | One month's status for a bureau credit | SK_ID_BUREAU (links to bureau, then SK_ID_CURR) |
| previous_application | One prior Home Credit application | SK_ID_CURR; application record is keyed by SK_ID_PREV |
| POS_CASH_balance | One monthly POS or cash-loan balance | SK_ID_CURR and SK_ID_PREV |
| installments_payments | One installment payment record | SK_ID_CURR and SK_ID_PREV |
| credit_card_balance | One monthly credit-card balance | SK_ID_CURR and SK_ID_PREV |

## 3. Full size versus sample size

The full competition files contain hundreds of thousands to tens of millions of rows. The working sample contains 5,000 applicants in `application_train`; the other tables retain all records linked to those applicants. The sample table counts are: 23,680 bureau records, 231,091 bureau-balance records, 23,009 previous applications, 138,350 POS/cash-balance records, 186,311 installment-payment records, and 50,668 credit-card-balance records. The fixed seed and instructor-provided split IDs make the applicant sample reproducible. Sampling whole applicants keeps their related records together across tables. The summary report records each sampled table's memory use before and after dtype reduction.

## 4. Target definition and class balance

TARGET is 1 when the applicant had payment difficulties and 0 otherwise. Defaults are the minority class: 8.07% in the full training file and 8.08% in the 5,000-applicant sample. The fixed split contains 4,000 training applicants and 1,000 validation applicants; the validation default rate is 8.1%. The fixed split keeps the same applicants assigned to each set on every run.

## 5. Known issues

The application data contains missing values, including columns with substantial missingness; the EDA notebook reports the most-missing columns in the 5,000-person sample. DAYS_EMPLOYED includes the sentinel value 365243, which is not a plausible number of days employed. The baseline replaces this value with a missing value and adds an indicator. DAYS_BIRTH and other DAYS_* fields use negative day counts relative to the application date.

## 6. Sensitive columns

CODE_GENDER and DAYS_BIRTH are sensitive or can reveal sensitive information. Other household, employment, and regional fields may be proxies. The baseline is an educational benchmark, not a deployed lending decision system; these features and group-level errors require review before any real use.

## 7. Limitations of a small sample

A 5,000-applicant sample has only about 400 defaults, and the 1,000-person validation split has about 80. Metrics can move when a small number of defaults change, and rare applicant types may be absent or poorly represented. Results from this sample should not be treated as a reliable estimate of performance on all future applicants.
