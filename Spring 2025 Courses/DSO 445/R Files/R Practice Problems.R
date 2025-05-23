x <- 10

x <- x + 2

y <- 5

x + y

x - y

x**2

sqrt(x)

grades <- c(100, 90, 85, 95)

classes <- c('DSO510', 'DSO545', 'DSO552','GSBA545')

class(grades)

class(classes)

grades <- grades + 5

x <- c(5, 10, '15', 20)

grades[2]
grades[4]

grades[c(2,4)]

# grades <- grades[-3]

grades[2] <- 97

student <- data.frame(Course = c(classes), Grade = c(grades))

lettergrade <- c('B', 'A+', 'A-', 'A+')
student <- cbind(student, lettergrade = lettergrade)

colnames(student)

colnames(student) <- c('Class', 'Grade', 'Lettergrade')

student$Grade[3]

student[2,]

student$Lettergrade

student[student$Class != 'DSO510', ]

student[, -2]

student[student$Class != 'GSBA545', c('Class', 'Grade')]

length(student$Class)

