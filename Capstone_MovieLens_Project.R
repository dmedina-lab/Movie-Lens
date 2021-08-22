##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


### Data exploration
##Overall profile of the dataset
# check for NA value 
anyNA(edx)
#Let’s first have a general overview of the dataset:
str(edx)
head(edx)
# summary of edx dataset
summary(edx)

# number of movies and users in data set 
edx %>% summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

# Top 5 movies with most ratings
edx %>% group_by(title) %>%
  summarize(count = n()) %>%
  top_n(5) %>%
  arrange(desc(count))

# let's look for the most  ratings
edx %>% group_by(rating) %>% 
  summarize(count = n()) %>% 
  top_n(5) %>%
  arrange(desc(count))  

#Number of occurence of each rating
edx %>% group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line() +
  ggtitle("Number of occurence of each rating")

#rating distribution
edx %>% 
  ggplot(aes(x = rating, y = ..prop..)) +
  geom_bar() + 
  ggtitle("Number of occurence of each rating") +
  labs(x="Movie Rating", y="Relative Frequency") + 
  geom_vline(xintercept=mean(edx$rating), color = "red")

#number of rating per movies 
edx %>% count(movieId) %>% ggplot(aes(n))+
  geom_histogram(bins= 30, color = "black" )+
  scale_x_log10()+
  ggtitle("Rating Number Per Movie")+
  theme_gray()+
  labs(x="Number Rating", y="Number Movie")

#number of rating per user
edx %>% count(userId) %>% ggplot(aes(n))+
  geom_histogram(bins=30, color = "black" )+
  ggtitle(" Number of Rating Per User")+
  scale_x_log10()+
  theme_gray()+
  labs(x="Number Rating", y="Number User")

#Top 10 most popular genres
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>%
  top_n(10) %>%
  ggplot(aes(count,reorder(genres,count))) + 
  geom_bar(stat = "identity")+ 
  labs(title = " Top 10 most popular Genre", y="Genre")+
  theme(axis.text.x  = element_text(angle= 0, vjust = 50 ))+
  theme_gray()

#Average of rating for each movie genres
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(avg_rating = mean(rating)) %>%
  ggplot(aes(avg_rating,reorder(genres,avg_rating))) + 
  geom_bar(stat = "identity")+ 
  labs(title = " Average of rating for each Genre", x="Average Rating", y="Genre")+
  theme(axis.text.x  = element_text(angle= 0, vjust = 50 ))+
  theme_gray()+
  geom_vline(xintercept=mean(edx$rating), color = "red")


##Data wrangling

#Set seed to 1
set.seed(1, sample.kind="Rounding")

##Split the edx data set to train set(80%) and test set(20%)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

#to make sure we do not include users and movies in the test set that do not appear in 
#the training set, we remove these entries using the semi_join function:
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

#RMSE calculation Function 
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2, na.rm = TRUE))
}

## First model: use average ratings for all movies regardless of user 

mu <- mean(train_set$rating)
mu #3.512493

#If we predict all unknown ratings with ˆμ we obtain the following RMSE:
naive_rmse <- RMSE( mu,test_set$rating)
naive_rmse #1.060526


## Second model, Movie Effect

#As we saw on the exploratory analysis some are rated more than other We can augment our previous model by adding the term  b_i to represent average ranking for movie i Yu,i = μ + bi + εu,i
#We can again use least squared to estimate the movie effect 
mu <- mean(train_set$rating)
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

#we can see that variability in the estimate as plotted here 
qplot(b_i, data = movie_avgs, bins = 10, color = I("black"))

#Predict ratings and apply RMSE to predicted ratigns 
predicted_ratings <- mu + test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
model_rmse_2 <- RMSE(predicted_ratings, test_set$rating)
model_rmse_2 #0.9434747

## Third model, Movie + User Effect

#Similar to the movie effect, intrinsic features of a given user could also affect the ratings of a movie
# Let's compute the user U for , for those who ratedover 100 movies 
train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

#The model will be Yu,i = μ + bi + bu + εu,i
#We could fit this model by using use the lm() function but it would be very slow
#lm(rating ~ as.factor(movieId) + as.factor(userId))
#We now further add the bias of user (b_u) to the movie effect model.
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
user_avgs

#now let's see how RMSE improved this time 
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_rmse_3 <- RMSE(predicted_ratings, test_set$rating)
model_rmse_3

##Fourth model, Regularization of both movie and user effects (use the same lambda for both movie and user effects)

#Perform cross validation to determine the parameter lambda

#We perform regularization on both movie and user effects. Using 10-fold cross validation, we will train one single lambda value for both movie and user effects.

# Split the data into 10 parts
set.seed(1, sample.kind = "Rounding")
cv_splits <- createFolds(edx$rating, k=10, returnTrain =TRUE)

# Define a matrix to store the results of cross validation
lambdas <- seq(0, 8, 0.1)
rmses <- matrix(nrow=10,ncol=length(lambdas))
# Perform 10-fold cross validation to determine the optimal lambda, it will take several minutes.
for(k in 1:10) {
  train_set_k <- edx[cv_splits[[k]],]
  test_set_k <- edx[-cv_splits[[k]],]
  
  # Make sure userId and movieId in test set are also in the train set
  test_final <- test_set_k %>% 
    semi_join(train_set_k, by = "movieId") %>%
    semi_join(train_set_k, by = "userId")
  
  # Add rows removed from validation set back into edx set
  removed <- anti_join(test_set_k, test_final)
  train_final <- rbind(train_set_k, removed)
  
  mu <- mean(train_final$rating)
  
  rmses[k,] <- sapply(lambdas, function(l){
    b_i <- train_final %>% 
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n()+l))
    b_u <- train_final %>% 
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n()+l))
    predicted_ratings <- 
      test_final %>% 
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      mutate(pred = mu + b_i + b_u) %>%
      pull(pred)
    return(RMSE(predicted_ratings, test_final$rating))
  })
}

rmses
#Perform the colum average of rmses
rmses_cv <- colMeans(rmses)
rmses_cv
#Plot the rmses cross validation vs lambdas
qplot(lambdas,rmses_cv)
#Get the minimal RMSE as the optimized lambda for model 
lambda <- lambdas[which.min(rmses_cv)]   
lambda #4.9

## Model generation and prediction
#Now we use this parameter lambda to predict the validation dataset and evaluate the RMSE.
mu <- mean(edx$rating)
movie_avg_reg <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))
user_avg_reg <- edx %>% 
  left_join(movie_avg_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
predicted_ratings_4 <- 
  test_set %>% 
  left_join(movie_avg_reg, by = "movieId") %>%
  left_join(user_avg_reg, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
model_rmse_4 <- RMSE(predicted_ratings_4, test_set$rating)   
model_rmse_4 #0.8569379

#Create a tibble for the RMSE result to store all the result from each method to compare
rmse_results <- tibble(method = c("Just the average","Movie Effect Model","Movie + User Effects Model","Regularized Movie + User Effect Model")
                       , RMSE = c(naive_rmse,model_rmse_2,model_rmse_3,model_rmse_4))
rmse_results 

## RMSE of the validation set

valid_pred_rating <- validation %>%
  left_join(movie_avg_reg, by = "movieId") %>%
  left_join(user_avg_reg, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

model_final <- RMSE(validation$rating, valid_pred_rating)
model_final

# Add the validations results to the tibble
rmse_results <- bind_rows( rmse_results, 
                           tibble(method = "Validation Results" , RMSE = model_final))
rmse_results
