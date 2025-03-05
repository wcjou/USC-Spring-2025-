install.packages('dplyr')
install.packages('ggplot2')
library(dplyr)
library(ggplot2)

str(sleep)

t.test(extra ~ group, sleep)

str(InsectSprays)

anova_results = aov(count ~ spray, InsectSprays)
tukey_results = TukeyHSD(anova_results)
tukey_results

str(rock)

lm1 = lm(area ~ peri, rock)
summary(lm1)

lm2 = lm(area ~ peri + shape + perm, rock)
summary(lm2)

hotdog = read.csv(file.choose())

ggplot(hotdog, aes(x = Year, y = Dogs.eaten, fill = New.record>0)) +
  geom_col() +
  scale_fill_manual(values = c('lightblue', 'red')) +
  xlab('Year') + 
  ylab('') + 
  ggtitle('Number of Hot Dogs Eaten')



gss = read.csv(file.choose())

head(gss)

gss_party = gss %>%
  group_by(partyid) %>%
  summarise(count = n()) %>%
  filter(count >= 2000)

gss_party

gss_party %>%
  mutate(percent = count / sum(count) * 100) %>%
  ggplot(aes(x = reorder(partyid, -percent), y = percent)) +
  geom_col(fill = c('purple', 'lightblue', 'lightblue', 'red', 'purple', 'red')) +
  xlab('')


gss %>%
  filter(!is.na(age)) %>%
  filter(race %in% c('Black', 'White')) %>%
  ggplot(aes(x = age, fill = race)) +
  geom_density(alpha = 0.4)

gss %>%
  filter(!is.na(age)) %>%
  group_by(partyid) %>%
  mutate(Age = factor(ifelse(age>=50, ">=50", "<50"))) %>%
  ggplot(aes(x = partyid, fill = Age)) +
  geom_bar(position = 'dodge') +
  scale_fill_manual(values = c('lightblue', 'pink')) +
  coord_flip() +
  xlab('') +
  ylab('') +
  ggtitle('Number of People: Party Aff. vs. Age (<50:Blue, >=50: Pink)')
  


gss %>%
  filter(!is.na(tvhours)) %>%
  group_by(relig) %>%
  summarize(avg_tvhours = mean(tvhours)) %>%
  ggplot(aes(x = avg_tvhours, y = reorder(relig, avg_tvhours))) + 
  geom_point(size=3) +
  theme_bw() +
  xlab('Avg. TV Hours') + 
  ylab('Religion')
