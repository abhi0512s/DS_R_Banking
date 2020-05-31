setwd("D:/AI & Data Science/R/Project 5 Banking")

bank_test=read.csv("bank-full_test.csv",stringsAsFactors = F)
bank_train=read.csv("bank-full_train.csv", stringsAsFactors = F)

library(dplyr)
library(car)

bank_test$y=NA

bank_test$data="test"
bank_train$data="train"

bank_all=rbind(bank_train,bank_test)

lapply(bank_all,function(x) sum(is.na(x)))

View(bank_all)
glimpse(bank_all)
#numeric-age,balance,duration,campaign,day
#dummy-job,marital,education
#boolean-default,housing,loan
#remove-ID,poutcome,contact,pdays,previous
#doubt-month

sort(table(bank_all$loan))

bank_all=bank_all %>% 
  select(-ID,-poutcome,-contact,-month,-pdays,-previous,)

bank_all=bank_all %>% 
  mutate(default=as.numeric(default=="yes"))

bank_all=bank_all %>% 
  mutate(housing=as.numeric(housing=="yes"))

bank_all=bank_all %>% 
  mutate(loan=as.numeric(loan=="yes"))

CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]

  for( cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)

    data[,name]=as.numeric(data[,var]==cat)
  }

  data[,var]=NULL
  return(data)
}

bank_all=CreateDummies(bank_all ,"job",500)
bank_all=CreateDummies(bank_all ,"marital",5000)
bank_all=CreateDummies(bank_all ,"education",5000)
#bank_all=CreateDummies(bank_all ,"month",1500)

bank_train=bank_all %>% filter(data=='train') %>% select(-data)
bank_test=bank_all %>% filter(data=='test') %>% select(-data,-y)

bank_train=bank_train %>% 
  mutate(y=as.numeric(y=="yes"))

set.seed(2)
s=sample(1:nrow(bank_train),.75*nrow(bank_train))
bank_train1=bank_train[s,]
bank_train2=bank_train[-s,]

library(pROC)

#Logistic Regression Model
for_vif=lm(y~.,data=bank_train1)

#alias(lm(y~.,data=bank_train1))

sort(vif(for_vif),decreasing = T)

for_vif=lm(y~.-job_management,data=bank_train1)

log_fit=glm(y~.,data=bank_train1)

log_fit
log_fit=step(log_fit)

summary(log_fit)

formula(log_fit)

val.score=predict(log_fit,newdata = bank_train2,type='response')

auc(roc(bank_train2$y,val.score))

#DTree Model
library(rpart)
library(rpart.plot)
library(tidyr)
require(rpart)
library(tree)

dtModel = tree(y~.,data=bank_train1)
plot(dtModel)
dtModel

val.score=predict(dtModel,newdata = bank_train2)
auc(roc(bank_train2$y,val.score))

#Random Forest Model
library(randomForest)
randomForestModel = randomForest(y~.,data=bank_train1)
d=importance(randomForestModel)
d
names(d)
d=as.data.frame(d)
d$IncNodePurity=rownames(d)
d %>% arrange(desc(IncNodePurity))

val.score=predict(randomForestModel,newdata = bank_train2)
auc(roc(bank_train2$y,val.score))

val.score.test=predict(randomForestModel,newdata = bank_test)

#GBM Model
library(gbm)
library(cvTools)

param=list(interaction.depth=c(1:10),
           n.trees=c(700),
           shrinkage=c(.1,.01,.001),
           n.minobsinnode=c(1,2,5,10))

subset_paras=function(full_list_para,n=10){
  
  all_comb=expand.grid(full_list_para)
  set.seed(2)
  s=sample(1:nrow(all_comb),n)
  
  subset_para=all_comb[s,]
  
  return(subset_para)
}

num_trials=10
my_params=subset_paras(param,num_trials)

mycost_auc=function(y,yhat){
  roccurve=pROC::roc(y,yhat)
  score=pROC::auc(roccurve)
  return(score)
}

myauc=0

for(i in 1:num_trials){
  print(paste0('starting iteration:',i))
  
  params=my_params[i,]
  
  k=cvTuning(gbm,y~.,data=bank_train,
             tuning =params,
             args = list(distribution="bernoulli"),
             folds = cvFolds(nrow(bank_train), K=10, type = "random"),
             cost =mycost_auc, seed =2,
             predictArgs = list(type="response",n.trees=params$n.trees)
  )
  score.this=k$cv[,2]
  
  if(score.this>myauc){
    print(params)
    
    myauc=score.this
    print(myauc)
    
    best_params=params
  }
  print('DONE')
}

myauc

best_params

best_params=data.frame(interaction.depth=2,
                       n.trees=700,
                       shrinkage=0.1,
                       n.minobsnode=2)

product.gbm.final=gbm(y~.,data=bank_train,
                      n.trees = best_params$n.trees,
                      n.minobsinnode = best_params$n.minobsnode,
                      shrinkage = best_params$shrinkage,
                      interaction.depth = best_params$interaction.depth,
                      distribution = "bernoulli")

product.gbm.final

test.pred=predict(product.gbm.final,newdata=bank_test,
                  n.trees = best_params$n.trees,type="response")

#Cutoff
train.score=predict(randomForestModel,newdata = bank_train,
                    n.trees = best_params$n.trees,type='response')

real=bank_train$y
cutoffs=seq(0.001,0.999,0.001)

cutoff_data=data.frame(cutoff=99,Sn=99,Sp=99,KS=99,F5=99,F.1=99,M=99)

for(cutoff in cutoffs){
  
  predicted=as.numeric(train.score>cutoff)
  
  TP=sum(real==1 & predicted==1)
  TN=sum(real==0 & predicted==0)
  FP=sum(real==0 & predicted==1)
  FN=sum(real==1 & predicted==0)
  
  P=TP+FN
  N=TN+FP
  
  Sn=TP/P
  Sp=TN/N
  precision=TP/(TP+FP)
  recall=Sn
  
  KS=(TP/P)-(FP/N)
  
  #print(paste0('KS Score: ',KS))
  
  F5=(26*precision*recall)/((25*precision)+recall)
  F.1=(1.01*precision*recall)/((.01*precision)+recall)
  
  M=(4*FP+FN)/(5*(P+N))
  
  cutoff_data=rbind(cutoff_data,
                    c(cutoff,Sn,Sp,KS,F5,F.1,M))
}

cutoff_data=cutoff_data[-1,]

max(cutoff_data$KS)

View(cutoff_data)

#### visualise how these measures move across cutoffs
library(ggplot2)
ggplot(cutoff_data,aes(x=cutoff,y=M))+geom_line()

library(tidyr)

cutoff_long=cutoff_data %>% 
  gather(Measure,Value,Sn:M)

ggplot(cutoff_long,aes(x=cutoff,y=Value,color=Measure))+geom_line()

my_cutoff=cutoff_data$cutoff[which.max(cutoff_data$KS)]

my_cutoff

# now that we have our cutoff we can convert score to hard classes

test.predicted=(val.score.test>my_cutoff)
test.predicted
class(test.predicted)
test1=c('No', 'Yes')[test.predicted + 1]
test1
write.table(test1,file ="Abhilash_Singh_P5_part2.csv",
            row.names = F,col.names="y")

############################Quiz############################################
#1
round(mean(bank_train$age),2)

#2
OutVals = boxplot(bank_train$balance)$out
length(which(bank_train$balance %in% OutVals))

OutVals = boxplot(bank_train$balance, plot=FALSE)$out
length(OutVals)

IQR=IQR(bank_train$balance)
summary(bank_train$balance)
q1=summary(bank_train$balance)[["1st Qu."]]
q3=summary(bank_train$balance)[["3rd Qu."]]
lower=(q1-1.5*IQR)
upper=(q3+1.5*IQR)
n1=sum(as.numeric(bank_train$balance<lower))
n2=sum(as.numeric(bank_train$balance>upper))
n1+n2

#3
var(bank_train$balance)

#8
n1=dim(bank_train)[1]
n2=table(bank_train$y)[1]
n3=table(bank_train$y)[2]
n2*100/n1
n3*100/n1
table(bank_train$loan)
############################################################################