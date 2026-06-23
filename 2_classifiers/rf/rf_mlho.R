# rf_mlho.R


library(mlho)
library(pROC)
if (!require(pacman)) install.packages("pacman")
pacman::p_load(
  data.table, devtools, backports, Hmisc, tidyr, dplyr, ggplot2,
  plyr, scales, readr, httr, DT, lubridate, tidyverse, reshape2,
  foreach, doParallel, caret, gbm, praznik, DALEX
)

# Reproducibility seed
set.seed(40)

# ============================================================
# Paths — edit here
# ============================================================
TRAIN_FILE   <- "data/features_rf_10k.csv"
VAL_FILE     <- "data/features_rf_validation.csv"
ALLCORRS_FILE <- "data/features_rf.csv"           
OUTPUT_DIR   <- "data/"

# ============================================================
# Load
# ============================================================
dat_all <- read_csv(TRAIN_FILE)
val_all <- read_csv(VAL_FILE)

duration_map <- c(
  "0-14 days"  = 1,
  "15-30 days" = 2,
  "30-90 days" = 3,
  "90+ days"   = 4
)

dat_all$sequence_id <- paste0(as.character(dat_all$sequence),
                               duration_map[dat_all$duration])
dat_all$label       <- ifelse(dat_all$label == "NOISE", 0, 1)
dat_all$label       <- as.factor(dat_all$label)

# ============================================================
# Stratified train / test split (80 / 20)
# ============================================================
idx       <- createDataPartition(dat_all$label, p = 0.8, list = FALSE)
dat.train <- dat_all[idx, ]
dat.test  <- dat_all[-idx, ]

drop_cols      <- c("sequence_id", "sequence", "startPhen_def",
                    "endPhenx_def", "duration", "duration_map")
drop_cols_test <- c("sequence", "startPhen_def", "endPhenx_def",
                    "duration", "duration_map")

dat.train <- dat.train[, !names(dat.train) %in% drop_cols]
dat.test  <- dat.test[,  !names(dat.test)  %in% drop_cols_test]
names(dat.test)[names(dat.test) == "sequence_id"] <- "patient_num"

# ============================================================
# Remove near-zero-variance features (on training partition only)
# ============================================================
nzv_cols   <- nearZeroVar(dat.train[, names(dat.train) != "label"])
nzv_names  <- names(dat.train)[names(dat.train) != "label"][nzv_cols]
message("NZV features removed: ", paste(nzv_names, collapse = ", "))

dat.train <- dat.train[, !names(dat.train) %in% nzv_names]
dat.test  <- dat.test[,  !names(dat.test)  %in% nzv_names]

dat.train <- dat.train %>% select(where(~ !any(is.na(.)))) %>% as.data.frame()
dat.test  <- dat.test  %>% select(where(~ !any(is.na(.)))) %>% as.data.frame()

# ============================================================
# Train RF (MLHO, 5-fold CV, Youden-optimal threshold)
# ============================================================
model_rf <- mlearn(
  dat.train  = dat.train,
  dat.test   = dat.test,
  dems       = NULL,
  save.model = TRUE,
  classifier = "rf",
  note       = "final_train",
  cv         = "cv",
  nfold      = 5,
  aoi        = "non-noise",
  multicore  = FALSE,
  preProc    = TRUE,
  calSHAP    = FALSE,
  n_incidence = 200,
  counterfactual = FALSE
)

save(model_rf, file = paste0(OUTPUT_DIR, "model_rf_final_train.RData"))
write.csv(data.frame(model_rf$ROC),
          paste0(OUTPUT_DIR, "train_rf_metrics.csv"), row.names = FALSE)
message("Training AUROC: ", round(model_rf$ROC$roc, 4),
        "  CV AUROC: ", round(model_rf$ROC$cv_roc, 4))

# ============================================================
# Evaluate on 200-row development set
# ============================================================
rf <- readRDS(paste0(getwd(), "/results/model_rf_final_train_non-noise.rds"))

val_all$sequence_id <- paste0(as.character(val_all$sequence),
                               duration_map[val_all$duration])
val_all$label       <- factor(ifelse(val_all$label_jt == "NOISE", "0", "1"),
                               levels = c("0", "1"))

probs_rf  <- predict(rf, newdata = val_all, type = "prob")
preds_rf  <- factor(ifelse(probs_rf$Y >= model_rf$ROC$thresholdj, "1", "0"),
                    levels = c("0", "1"))
roc_val   <- pROC::roc(val_all$label, probs_rf$Y, levels = c("0", "1"))
auc_val   <- as.numeric(roc_val$auc)

cm <- caret::confusionMatrix(preds_rf, val_all$label, positive = "1")
print(cm)
message("Validation AUROC: ", round(auc_val, 4))

# Save validation predictions
val_all$rf_label <- preds_rf
val_all$rf_prob  <- probs_rf$Y
write.csv(val_all, paste0(OUTPUT_DIR, "val_rf_predictions.csv"), row.names = FALSE)

# Save validation metrics (including ROC AUC)
cm_stats <- rbind(
  data.frame(metric = c(names(cm$overall), names(cm$byClass)),
             value  = c(cm$overall, cm$byClass)),
  data.frame(metric = "ROC_AUC_val", value = auc_val),
  data.frame(metric = "MCC",         value = {
    tp <- cm$table[2, 2]; tn <- cm$table[1, 1]
    fp <- cm$table[2, 1]; fn <- cm$table[1, 2]
    (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  })
)
write.csv(cm_stats, paste0(OUTPUT_DIR, "val_rf_metric.csv"), row.names = FALSE)

# ============================================================
# Score full 435k corpus
# ============================================================
corr_all <- read_csv(ALLCORRS_FILE)
corr_all$sequence_id <- paste0(as.character(corr_all$sequence),
                                duration_map[corr_all$duration])
probs_all  <- predict(rf, newdata = corr_all, type = "prob")
preds_all  <- factor(ifelse(probs_all$Y >= model_rf$ROC$thresholdj, "1", "0"),
                     levels = c("0", "1"))
corr_all$rf_label <- preds_all
corr_all$rf_prob  <- probs_all$Y
write.csv(corr_all, paste0(OUTPUT_DIR, "rf_predictions_allcorrs.csv"), row.names = FALSE)
message("Saved rf_predictions_allcorrs.csv  (", nrow(corr_all), " rows)")
