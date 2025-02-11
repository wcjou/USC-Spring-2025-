# ANOVA = Analysis of Variance
# compares the means of several groups

npk

levels(npk$block)

npkdf = npk

t.test(yield ~ N, npkdf)

t.test(yield ~ P, npkdf)

t.test(yield ~ K, npkdf)

model1 = aov(yield ~ block, npkdf) # technically offically anova

summary(model1) # gives you overall result

TukeyHSD(model1) # posthoc that tells you where the sig findings are

model0 = lm(yield ~ block, npkdf) # technically linear model which is used for regression but also can be used for anova

summary(model0) 

model2 = aov(yield ~ block + N, npkdf)

summary(model2)

TukeyHSD(model2)

model2b = lm(yield ~ block + N, npkdf)
summary(model2b)

model3 = aov(yield ~ block + N + block*N, npkdf)
summary(model3)

TukeyHSD(model3)

model3b = lm(yield ~ block + N + block*N, npkdf)
summary(model3b)
