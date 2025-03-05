install.packages('ggplot2')
install.packages('dplyr')
library(ggplot2)
library(dplyr)


sleep

t.test(extra ~ group, sleep)

InsectSprays

anova = aov(count ~ spray, InsectSprays)
summary(anova)
tukey_result = TukeyHSD(anova)
tukey_result

rock

lin_reg= lm(area ~ peri, rock)
summary(lin_reg)

lin_reg2 = lm(area ~ peri + shape + perm, rock)
summary(lin_reg2)

hotdogdf = read.csv(file.choose())

ggplot(hotdogdf, aes(x = Year, y = Dogs.eaten, fill = New.record > 0)) +
  geom_col() + scale_fill_manual(values=c('lightblue', 'red')) + 
  ggtitle('Number of Hotdogs Eaten')

gss = read.csv(file.choose())

gss_party = gss %>%

gss_party

