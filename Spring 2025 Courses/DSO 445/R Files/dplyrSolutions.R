install.packages("tidyverse")
library(tidyverse)
install.packages("hflights")
library(hflights)

dfflights = hflights
head(dfflights)

#1. 
#filter, select, summarize

#2. 
df2 = select(dfflights, ActualElapsedTime, AirTime,ArrDelay, DepDelay) 
head(df2)



#3.
df3 = select(dfflights, Origin:Cancelled)
head(df3)

#4.
df4 = select(dfflights, -(DepTime:AirTime))
head(df4)

#5
df5 = select(dfflights, ends_with("Delay"))
head(df5)



#6. 
df6 = select(dfflights, UniqueCarrier:TailNum | contains("Cancel"))
head(df6)
df6b = select(dfflights, UniqueCarrier, ends_with("Num"), starts_with(("Cancel")))
head(df6b)


#7. 
df7 = mutate(dfflights, GroundTime = TaxiIn + TaxiOut)
head(df7)

#8.
df8 = filter(dfflights, Distance >= 3000)
head(df8)


#9. 

df9 = filter(dfflights, UniqueCarrier %in% c("AA", "AS", "B6"))
head(df9)

df9b = filter(dfflights, UniqueCarrier == "AA" | UniqueCarrier == "AS"| UniqueCarrier =="B6")
head(df9b)


#10.
df10 = filter(df7, GroundTime > AirTime)
head(df10)
df10b = filter(dfflights, TaxiIn + TaxiOut > AirTime)


#11.
df11 = filter(dfflights, (DayOfWeek == 6 | DayOfWeek == 7) & Cancelled ==1)
df11b = filter(dfflights, DayOfWeek %in% c(6,7) & Cancelled == 1)

#12.
df12 = arrange(dfflights, UniqueCarrier, desc(DepDelay))
head(df12)

#13.
df13 = mutate(dfflights, TotalDelay = ArrDelay + DepDelay)
head(df13)

#14. 
df14 = arrange(filter(dfflights, Dest == "DFW" & DepTime < 0800), desc(AirTime))
head(df14)


filter(dfflights, Dest == "DFW" & DepTime < 0800)
arrange(, desc(AirTime))

df14b = dfflights %>%
  filter(Dest == "DFW" & DepTime < 800) %>%
  arrange(desc(AirTime))


#15.
df15 = dfflights %>%
  summarise(min_dist= min(Distance), max_dist = max(Distance))
df15

df15b = summarise(dfflights, min_dist = min(Distance),max_dist = max(Distance))
head(df15b)

#16.
df16 = dfflights %>%
  filter(Diverted == 1)%>%
  summarise(max_div = max(Distance))
head(df16)



  