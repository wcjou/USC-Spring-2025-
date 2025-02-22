dfsleep = sleep

#write.csv(dfsleep, "dfsleep.csv" )
#?write.csv
#getwd()

sleep
str(sleep)

# t tests requires IV that is categorical in other words factor type with 2 levels and 1 cont DV in other words numeric dv

model0 = t.test(extra ~ group, dfsleep) #t.test(y ~ x, dataframe) #assumed unequal variance Welches
model0
model1 = t.test(extra ~ group, dfsleep, var.equal = T) #og independent t test
model1


#model2 = t.test(extra ~ group, paired = TRUE, data = dfsleep)
?t.test()


model3 = t.test(dfsleep$extra) #one sample t test comparison to a benchmark
