library(tidyverse)
library(glue)
library(broom)
library(broom.mixed)
library(langcog)
library(stringr)
library(lme4)
library(modelr)
library(purrr)

load("../../../data/aoa_predictors/uni_joined.RData")
uni_joined_eng <- uni_joined %>% filter(language == "English (American)")

predictors <- c("frequency", "MLU", "final_frequency", "solo_frequency",
                "num_phons", "concreteness", "valence", "arousal", "babiness")
.alpha <- 0.05
set.seed(42)

model_data <- uni_joined_eng %>% 
  select(uni_lemma, words, lexical_classes, !!predictors) %>%
  distinct() %>%
  mutate(lexical_category = if_else(
    str_detect(lexical_classes, ","), "other", lexical_classes
  ) %>%
    as_factor() %>%
    fct_collapse("predicates" = c("verbs", "adjectives", "adverbs"))) %>%
  select(-lexical_classes)

df.model_data<- model_data %>% 
  mutate(word_clean = gsub(" [(].*$","",words)) %>%
  mutate(word_clean = str_remove(word_clean, "['`*]")) %>% 
  filter(!str_detect(word_clean, " ")) 

df.uni_lemma <- df.model_data %>% select(word_clean, uni_lemma)

write.table(df.uni_lemma, file=paste("../../../data/aoa_predictors/aoa_words.csv"), sep = "\t", quote = FALSE, col.names = TRUE, row.names = FALSE)

save(model_data, file = "../../../data/aoa_predictors/model_data.RData")
