install.packages('ggplot2')
install.packages('dplyr')
library(ggplot2)
library(dplyr)

df = read.csv(file.choose())

head(df)
str(df)


df <- df %>%
  filter(!is.na(Calories.Burned)) %>%
  mutate(
    Workout.Type = as.factor(Workout.Type),
    Workout.Intensity = as.factor(Workout.Intensity),
    Workout.Duration.Category = factor(ifelse(
      Workout.Duration..mins. <= 30, "<= 30 min",
      ifelse(Workout.Duration..mins. > 30 & Workout.Duration..mins. <= 60, "30-60 min",
             ifelse(Workout.Duration..mins. > 60 & Workout.Duration..mins. <= 120, "60-120 min",
                    "More than 120 min"))))
  )

head(df)


ggplot(df, aes(x = Workout.Type, y = Calories.Burned, fill = Workout.Intensity)) +
  geom_boxplot() +
  labs(title = "Calories Burned by Workout Type & Intensity", x = "Workout Type", y = "Calories Burned") +
  theme_minimal()



ggplot(df, aes(x = Workout.Duration.Category, y = Calories.Burned, fill = Workout.Duration.Category)) +
  geom_boxplot() +
  labs(title = "Boxplot of Calories Burned by Workout Duration Category",
       x = "Workout Duration Category", 
       y = "Calories Burned") +
  theme_minimal() +
  theme(legend.position = "none") 



ggplot(df, aes(x = Calories.Burned)) +
  geom_histogram(fill = "blue", color = "black", bins = 30) +
  labs(title = "Distribution of Calories Burned", x = "Calories Burned", y = "Count") +
  theme_minimal()

ggplot(df, aes(x = Workout.Type, y = Calories.Burned, fill = Workout.Type)) +
  geom_boxplot() +
  labs(title = "Calories Burned by Workout Type", x = "Workout Type", y = "Calories Burned") +
  theme_minimal()

ggplot(df, aes(x = Workout.Intensity, y = Calories.Burned, fill = Workout.Intensity)) +
  geom_boxplot() +
  labs(title = "Calories Burned by Workout Intensity", x = "Workout Intensity", y = "Calories Burned") +
  theme_minimal()


mean_calories_workout_type <- df %>%
  group_by(Workout.Type) %>%
  summarise(Mean_Calories_Burned = mean(Calories.Burned, na.rm = TRUE)) %>%
  arrange(desc(Mean_Calories_Burned))

mean_calories_workout_intensity <- df %>%
  group_by(Workout.Intensity) %>%
  summarise(Mean_Calories_Burned = mean(Calories.Burned, na.rm = TRUE)) %>%
  arrange(desc(Mean_Calories_Burned))

mean_calories_workout_duration <- df %>%
  group_by(Workout.Duration.Category) %>%
  summarise(Mean_Calories_Burned = mean(Calories.Burned, na.rm = TRUE)) %>%
  arrange(desc(Mean_Calories_Burned))

mean_calories_workout_duration
mean_calories_workout_intensity
mean_calories_workout_type


anova_results = aov(Calories.Burned ~ Workout.Type * Workout.Intensity * Workout.Duration.Category, data = df)

anova_summary = summary(anova_results)
anova_summary

tukey_results = TukeyHSD(anova_results)
tukey_results

anova_summary <- summary(anova_results)

# Extract the p-values
p_values <- anova_summary[[1]][["Pr(>F)"]]

# Print only significant p-values (alpha = 0.05)
significant_p <- p_values[p_values < 0.05]
significant_p

lm1 = lm(Calories.Burned ~ Workout.Duration..mins., df)
summary(lm1)

