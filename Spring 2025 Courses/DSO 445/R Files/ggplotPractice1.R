install.packages('ggplot2')
library(ggplot2)

mpg

#2

ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point()

#3
# NOTE WHEN USING SCATTER USE COLOR NOT FILL USE FILL ON BAR CHARTS OR COLUMN CHARTS

ggplot(mpg, aes(x = displ, y = hwy, color = class)) + 
  geom_point() +
  geom_smooth()

#4

ggplot(mpg, aes(x = displ, y = hwy)) + 
  geom_point(aes(color = class)) +
  geom_smooth()

#5


ggplot(mpg, aes(x = displ, y = hwy)) + 
  geom_point(aes(size = class)) +
  geom_smooth()

#6

ggplot(mpg, aes(x = displ, y = hwy)) + 
  geom_point(aes(color = class)) +
  facet_wrap(~class)

#7

str(diamonds)

#8

ggplot(diamonds, aes(x = cut)) + 
  geom_bar()

#9

ggplot(diamonds, aes(x = carat)) + 
  geom_bar()

#10

install.packages('gcookbook')
library(gcookbook)

#11

str(BOD)

#12

ggplot(BOD, aes(x = Time, y = demand)) +
  geom_col()

ggplot(BOD, aes(x = Time, y = demand)) + 
  geom_bar(stat = 'identity')

#13

ggplot(BOD, aes(x = as.factor(Time), y = demand)) +
  geom_col()

#14

ggplot(BOD, aes(x = as.factor(Time), y = demand)) +
  geom_col(width = 0.5)

#15

ggplot(BOD, aes(x = as.factor(Time), y = demand)) +
  geom_col(width = 0.5, fill = 'lightblue')

#16

ggplot(BOD, aes(x = as.factor(Time), y = demand)) +
  geom_col(width = 0.5, fill = 'lightblue') +
  geom_text(aes(label = demand))

#17

ggplot(BOD, aes(x = as.factor(Time), y = demand)) +
  geom_col(width = 0.5, fill = 'lightblue') +
  geom_text(aes(label = demand)) + 
  coord_flip()

#18

ggplot(diamonds, aes(x = cut)) +
  geom_bar(aes(fill = clarity))

ggplot(diamonds, aes(x = cut,fill = clarity)) +
  geom_bar(position = "dodge")

#19

df = uspopchange

dftop = head(df[order(-df$Change),], 10)

ggplot(dftop, aes(x = reorder(State, -Change), y = Change, fill = Region)) + 
  geom_col()

#20

ggplot(dftop, aes(x = reorder(State, -Change), y = Change, fill = Region)) + 
  geom_col() +
  scale_fill_manual(values = c('pink', 'lightblue'))

# 21
# already reordered

#22

tophitters2001

baseball = tophitters2001

str(baseball)

#23

baseball10 = head(baseball[order(-baseball$avg),],10)

ggplot(baseball10, aes(x = reorder(name, avg), y = avg)) +
  geom_point()
