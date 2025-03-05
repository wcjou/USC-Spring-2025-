install.packages('dplyr')
install.packages('tidyverse')
install.packages('hflights')
library(dplyr)
library(tidyverse)
library(hflights)

dfflights = hflights

head(dfflights)

#2

dfflights2 = select(dfflights, c(ActualElapsedTime, AirTime, ArrDelay, DepDelay))

head(dfflights2)

#3

dfflights3 = select(dfflights, Origin:Cancelled)
head(dfflights3)

#4

df4 = select(dfflights, -(DepTime:AirTime))
head(df4)

#5

# dplyr helper functions
# starts_with()
# ends_with()
# contains()
# matches()
# num_range()
# one_of()

df5 = select(dfflights, ends_with('Delay'))
head(df5)

#6

df6 = select(dfflights, UniqueCarrier:TailNum | contains('Cancel'))
head(df6)  

#7

df7 = mutate(dfflights, GroundTime = TaxiIn + TaxiOut)
head(df7)

#8

df8 = filter(dfflights, Distance >= 3000)
head(df8)

#9

df9 = filter(dfflights, UniqueCarrier %in% c('AA', 'AS', 'B6'))
head(df9)

#10

df10 = filter(dfflights, TaxiIn > AirTime)
head(df10)

#11

df11 = filter(dfflights, DayOfWeek %in% c(6, 7) & Cancelled == 1)
head(df11)

#12

df12 = arrange(dfflights, UniqueCarrier, desc(DepDelay))
head(df12)

#13

df13 = mutate(dfflights, TotDelay = ArrDelay + DepDelay)
df13 = arrange(df13, TotDelay)
head(df13)

#14

df14 = filter(dfflights, Dest == 'DFW', DepTime < 0800)
df14 = arrange(df14, desc(AirTime))
head(df14)

#15

df15 = dfflights
df15b = summarise(dfflights, min_dist = min(Distance), max_dist = max(Distance))
head(df15b)

#16

df16 = dfflights

df16 %>%
  filter(Diverted == 1) %>%
  summarise(max_div = max(Distance))

#17

# aggregate functions
# first(x) : the first element of vector x
# last(x) : the last element of vector x
# nth(x, n) : the nth element of vector x
# n(x) : the number of rows in the dataframe or group of observations that summarise() describes
# n_distinct(x) :" the number of unqiue values in vector x


dfflights %>%
  summarise(num_obs = n(), n_carrier = n_distinct(UniqueCarrier), n_dest = n_distinct(Dest), dest100 = nth(Dest, 100))

#18

dfflights %>%
  mutate(diff = TaxiIn - TaxiOut) %>%
  filter(!is.na(diff)) %>%
  summarise(avg = mean(diff))

#19


d = dfflights %>%
  select(Dest, UniqueCarrier, Distance, ActualElapsedTime) %>%
  mutate(RealTime = ActualElapsedTime + 100)

head(d)

#20

dfflights %>%
  group_by(Dest) %>%
  summarise(num_flights = n(), 
            avg_dist = mean(Distance),
            avg_ArrDelay = mean(ArrDelay),) %>%
  ggplot(aes(x = avg_dist, y = avg_ArrDelay)) + geom_point()
  
