install.packages("ggplot2")
library(ggplot2)

## CASE 1 ##

# Reading csv file in and storing it in tipsdf
tipsdf <- read.csv("C:/Users/willi/OneDrive/Desktop/tips.csv")

# Q1
num_female <- sum(tipsdf$sex == "F")
print(num_female)

# Q2
num_female_nonsmoker <- sum(tipsdf$sex == "F" & tipsdf$smoker == "No")
print(num_female_nonsmoker)

# Q3
ggplot(tipsdf, aes(x = sex)) + 
  geom_bar(aes(fill = smoker)) +
  scale_fill_manual(values = c('lightblue', 'pink')) +
  xlab('Gender') +
  ylab('Count') +
  ggtitle('Count of Smokers/Non-smokers for Each Gender')

# Q4
tipsdf$tipCategory <- ifelse(tipsdf$tip <= 2.5, 'Low', 'High')

ggplot(tipsdf, aes(x = tipCategory)) +
  geom_bar() +
  xlab('Tip Category')

## CASE 2 ##

# Reading in syria csv and storing it in syriadf
syriadf <- read.csv("C:/Users/willi/OneDrive/Desktop/syria_refugees.csv")

# Q5
num_asian_host <- sum(syriadf$continent == 'Asia')
print(num_asian_host)

# Q6
num_asian_euro_host <- sum(syriadf$continent == 'Asia' | syriadf$continent == 'Europe')
print(num_asian_euro_host)

# Q7
NA_refugee_pop <- sum(syriadf$refugees[syriadf$continent == 'North America'])
print(NA_refugee_pop)

# Q8
syriadf_sorted <- syriadf[order(-syriadf$refugees),]
top20_refugee_countries <- head(syriadf_sorted, 20)
ggplot(top20_refugee_countries, aes(x = refugees, y = reorder(Country, refugees))) + 
  geom_point() +
  xlab('Number of Refugees') + 
  ylab('Country') +
  ggtitle('Top 20 Countries with Largest Refugee Population')


