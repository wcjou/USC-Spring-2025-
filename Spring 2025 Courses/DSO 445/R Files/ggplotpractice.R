install.packages("ggplot2")
library(ggplot2)


# plot(anscombe$mpg x1, anscombe$y1) # plotting without ggplot

mpg
#data(mpg)
#?mpg

ggplot(data = mpg, aes(x = drv, y = hwy)) + geom_col()
ggplot(data = mpg, aes(x = drv, y = hwy)) + geom_bar(stat = "identity")

# barchart, boxplot, or column chart work best for this data
# doing geom_bar() however will give an error, instead use col()
# or you can do geom_bar(stat = "identity")


# In Class Exercise
mpgdf <- mpg
ggplot(data = mpgdf, aes(x = displ, y = hwy, color = class)) + geom_point() 

#or

ggplot(data = mpgdf, aes(x = displ, y = hwy)) + geom_point(aes(color = class))


ggplot(data = mpgdf, aes(x = displ, y = hwy, size = class)) + geom_point() 
# size should not be used for discrete variables

ggplot(data = mpgdf, aes(x = displ, y = hwy)) + 
  geom_point(aes(color = class)) + 
  facet_wrap(~class)

diamondsdf <- diamonds

str(diamondsdf)
summary(diamondsdf)

ggplot(data = diamondsdf, aes(x = cut)) + geom_bar(color = 'green', fill = 'deepskyblue') 
# since the color is not representing a variable, it is not inside aes()
colors()
ggplot(data = diamondsdf, aes(x = carat)) + 
  geom_bar()

install.packages("gcookbook")
library(gcookbook)

BODdf <- BOD

ggplot(data = BODdf, aes(x = Time, y = demand)) + 
  geom_col()



BODdf$Time = as.factor(BODdf$Time) # converting the time variable into a factor (categorical) variable
ggplot(data = BODdf, aes(x = Time, y = demand)) + 
  geom_col()

BODdf$Time = as.numeric(BODdf$Time) 

# converting time back to numeric

ggplot(BODdf, aes(factor(Time), y = demand)) + 
  geom_col()
# making the same change without modifying data set and only changing the graph
# !! note that it removes the gap, but changes 7 to 6

ggplot(BODdf, aes(x = Time, y = demand)) + 
  geom_col(width = 0.5)

ggplot(BODdf, aes(x = Time, y = demand)) + 
  geom_col(width = 0.5, fill = "lightblue", color = "black")


ggplot(BODdf, aes(x = Time, y = demand)) + 
  geom_col(width = 0.5, fill = "lightblue", color = "black") +
  geom_text(aes(label = demand), vjust = 10, color = 'red')
# whenever you deal with a variable you use aes()
# for vjust and hjust, it's opposite signed, so negative is up/right, and positive is down/left

ggplot(BODdf, aes(x = Time, y = demand)) + 
  geom_col(width = 0.5, fill = "lightblue", color = "black") +
  geom_text(aes(label = demand), hjust = -.05, color = 'red') +
  coord_flip()

ggplot(diamondsdf, aes(x = cut)) + 
  geom_bar(aes(fill = clarity))

ggplot(diamondsdf, aes(x = cut)) + 
  geom_bar(aes(fill = clarity), position = "dodge")


upcdf = uspopchange

upcsorteddf = upcdf[order(-uspopchange$Change),]

top10uspsorteddf = head(upcsorteddf, 10)

ggplot(top10uspsorteddf, aes(x = Abb, y = Change)) +
  geom_col(aes(fill = Region))


ggplot(top10uspsorteddf, aes(x = Abb, y = Change)) +
  geom_col(aes(fill = Region)) + scale_fill_manual(values = c('pink', 'lightblue'))


ggplot(top10uspsorteddf, aes(x = reorder(Abb, -Change), y = Change)) +
  geom_col(aes(fill = Region)) + scale_fill_manual(values = c('pink', 'lightblue'))


baseballdf = tophitters2001

str(baseballdf)

baseballdfsorted = baseballdf[order(-baseballdf$avg),]

top10baseballdfsorted = head(baseballdfsorted, 10)

subsetbaseballdf = top10baseballdfsorted[, c('name', 'avg')]

ggplot(subsetbaseballdf, aes(x = avg, y = reorder(name, -avg))) + 
  geom_point(size = 3) +
  theme_minimal() +
  xlab('Average') +
  ylab('Name') + 
  ggtitle("Baseball Players Hitting Average")

  