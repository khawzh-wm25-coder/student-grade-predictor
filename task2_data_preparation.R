# ==============================================================================
# TASK 2: DATA PREPARATION (UCI Student Performance Dataset)
# Aligned with CLO1, CLO2, CLO3 & CRISP-DM Phase 3
# ==============================================================================

# 1. Load Required Libraries
# Install if missing: install.packages(c("tidyverse", "tidymodels", "skimr"))
library(tidyverse)
library(tidymodels)

# Set seed for full reproducibility across team members
set.seed(42)

# ==============================================================================
# PART 1: DATASET SELECTION & SCOPE
# ==============================================================================
# Note: UCI files use semicolons (;) as delimiters
mat_raw <- read_delim("student-mat.csv", delim = ";", show_col_types = FALSE)
por_raw <- read_delim("student-por.csv", delim = ";", show_col_types = FALSE)

cat("--- DATASET DIMENSIONS & COVERAGE ---\n")
cat("Math dataset (student-mat.csv):      ", nrow(mat_raw), "rows x", ncol(mat_raw), "columns\n")
cat("Portuguese dataset (student-por.csv):", nrow(por_raw), "rows x", ncol(por_raw), "columns\n")

# Verify the 382 overlapping students across both courses using the 13 shared attributes
merge_keys <- c("school", "sex", "age", "address", "famsize", "Pstatus", 
                "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet")

overlapping_students <- inner_join(mat_raw, por_raw, by = merge_keys, suffix = c("_mat", "_por"))
cat("Overlapping students in both files:  ", nrow(overlapping_students), "students\n\n")

# Select primary modeling dataset (Math course: 395 independent student records)
# Using a single subject avoids violating observation independence from the 382 duplicates
df_raw <- mat_raw


# ==============================================================================
# PART 2: DATA INSPECTION & QUALITY ASSESSMENT (BEFORE CHECKS)
# ==============================================================================
cat("--- [BEFORE] DATA STRUCTURE & QUALITY AUDIT ---\n")

# 2A. Data Types Audit
cat("Column types BEFORE cleaning:\n")
print(table(sapply(df_raw, class)))

# 2B. Missing Values Audit
total_na <- sum(is.na(df_raw))
cat("\nTotal Missing Values (NA) in raw data:", total_na, "\n")

# 2C. Exact Row Duplicates Audit
exact_dups <- sum(duplicated(df_raw))
cat("Exact duplicate rows in student-mat.csv:", exact_dups, "\n")

# 2D. Implausible / Outlier Value Checks (against UCI Data Dictionary ranges)
cat("\n--- RANGE & OUTLIER VALIDATION ---\n")
range_checks <- tibble(
  Variable = c("age (15-22)", "absences (0-93)", "G1 (0-20)", "G2 (0-20)", "G3 (0-20)"),
  Min_Val  = c(min(df_raw$age), min(df_raw$absences), min(df_raw$G1), min(df_raw$G2), min(df_raw$G3)),
  Max_Val  = c(max(df_raw$age), max(df_raw$absences), max(df_raw$G1), max(df_raw$G2), max(df_raw$G3)),
  Mean_Val = round(c(mean(df_raw$age), mean(df_raw$absences), mean(df_raw$G1), mean(df_raw$G2), mean(df_raw$G3)), 2),
  SD_Val   = round(c(sd(df_raw$age), sd(df_raw$absences), sd(df_raw$G1), sd(df_raw$G2), sd(df_raw$G3)), 2)
)
print(range_checks)

# Inspect zero-grade anomalies in G3 (students with G3 == 0 despite positive G1/G2)
zero_g3 <- df_raw %>% filter(G3 == 0) %>% select(school, sex, age, failures, absences, G1, G2, G3)
cat("\nNumber of students with G3 == 0:", nrow(zero_g3), "\n")
print(head(zero_g3, 5))


# ==============================================================================
# PART 3: PREPROCESSING & CLEANING OPERATIONS
# ==============================================================================
# Convert character nominal & binary variables to proper R factors
# (Note: Type casting does not leak statistical parameters across rows)
df_clean <- df_raw %>%
  mutate(
    # Nominal & Binary categorical variables -> factors
    across(where(is.character), as.factor),
    
    # Optional: Create binary classification target alongside numeric G3
    # Standard Portuguese pass threshold: G3 >= 10
    pass_status = factor(ifelse(G3 >= 10, "Pass", "Fail"), levels = c("Fail", "Pass"))
  )

cat("\n--- [AFTER TYPE CASTING] DATA STRUCTURE CHECK ---\n")
cat("Column types AFTER factor conversion:\n")
print(table(sapply(df_clean, class)))
cat("\nClass balance for binary target (pass_status):\n")
print(table(df_clean$pass_status))


# ==============================================================================
# PART 4: REPRODUCIBLE PIPELINE & DATA LEAKAGE PREVENTION (TIDYMODELS)
# ==============================================================================
# CRITICAL STEP: Split into 80% Training and 20% Testing BEFORE any scaling,
# centering, or dummy encoding is learned from the data.
data_split <- initial_split(df_clean, prop = 0.80, strata = G3)
train_data <- training(data_split)
test_data  <- testing(data_split)

cat("\n--- TRAIN / TEST SPLIT SUMMARY ---\n")
cat("Training set observations (80%):", nrow(train_data), "\n")
cat("Testing set observations  (20%):", nrow(test_data), "\n")

# Define the preprocessing recipe strictly on training data
# (Predictors for regression target G3; exclude pass_status to prevent target leakage)
student_recipe <- recipe(G3 ~ ., data = train_data) %>%
  # Remove alternative classification target so it is not used as a predictor
  step_rm(pass_status) %>%
  # Remove any zero-variance predictors if present
  step_zv(all_predictors()) %>%
  # One-hot / dummy encode all nominal factor predictors
  step_dummy(all_nominal_predictors(), one_hot = FALSE) %>%
  # Normalize (center & scale: mean = 0, sd = 1) all numeric predictors
  step_normalize(all_numeric_predictors())

# Fit ('prep') the recipe ONLY on train_data so means/SDs come strictly from training set
trained_recipe <- prep(student_recipe, training = train_data)

# Apply ('bake') the learned transformations to train_data and test_data separately
train_processed <- bake(trained_recipe, new_data = NULL)
test_processed  <- bake(trained_recipe, new_data = test_data)


# ==============================================================================
# PART 5: BEFORE-AND-AFTER PIPELINE VERIFICATION & LEAKAGE PROOF
# ==============================================================================
cat("\n--- [AFTER PIPELINE] BEFORE-AND-AFTER VERIFICATION ---\n")
cat("Raw training dimensions:      ", nrow(train_data), "rows x", ncol(train_data), "cols\n")
cat("Processed training dimensions:", nrow(train_processed), "rows x", ncol(train_processed), "cols (after dummy encoding)\n")
cat("Processed testing dimensions: ", nrow(test_processed), "rows x", ncol(test_processed), "cols\n")

# Proof of zero data leakage:
# Training predictors have exact Mean = 0 and SD = 1, whereas Test predictors
# are scaled using Training statistics (so Test Mean != 0 and Test SD != 1).
leakage_proof <- tibble(
  Feature = c("absences", "G1", "G2"),
  Raw_Train_Mean       = round(c(mean(train_data$absences), mean(train_data$G1), mean(train_data$G2)), 3),
  Processed_Train_Mean = round(c(mean(train_processed$absences), mean(train_processed$G1), mean(train_processed$G2)), 3),
  Processed_Train_SD   = round(c(sd(train_processed$absences), sd(train_processed$G1), sd(train_processed$G2)), 3),
  Processed_Test_Mean  = round(c(mean(test_processed$absences), mean(test_processed$G1), mean(test_processed$G2)), 3),
  Processed_Test_SD    = round(c(sd(test_processed$absences), sd(test_processed$G1), sd(test_processed$G2)), 3)
)

cat("\n--- DATA LEAKAGE PREVENTION PROOF (SCALING STATS) ---\n")
print(leakage_proof)

# Save cleaned datasets and prepped recipe for Task 3 (EDA) and Part B (Modelling)
saveRDS(df_clean, "student_mat_clean_eda.rds")
saveRDS(list(train = train_processed, test = test_processed, recipe = trained_recipe), "student_mat_model_ready.rds")
cat("\nSaved 'student_mat_clean_eda.rds' and 'student_mat_model_ready.rds' successfully.\n")
