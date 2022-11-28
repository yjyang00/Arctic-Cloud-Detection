## split data first
## input training data that has been split--with fold

systematic.split = function(image, i){
  # divide the image into 2T*2T pieces
  T = 4
  image = image %>%
    mutate(level.x = as.factor(cut(image$x, breaks = 2*T)),
           level.y = as.factor(cut(image$y, breaks = 2*T)),)
  levels(image$level.x)=as.character(1:(2*T))
  levels(image$level.y)=as.character(1:(2*T))
  
  image = image %>%
    mutate(
      mod = (as.integer(level.x) + as.integer(level.y)) %% T) %>%
    mutate(fold = mod+4*i-3) 
  
  return(image %>% select(-c(level.x, level.y, mod)))
}
imagem1.syst = systematic.split(imagem1, i=1)
imagem2.syst = systematic.split(imagem2, i=2)
imagem3.syst = systematic.split(imagem3, i=3)
# combine the three images
image.syst = rbind(imagem1.syst, imagem2.syst, imagem3.syst)
# create the test data
image.syst.test = NULL
set.seed(111111)
idx.test = c()
for (i in 1:2){
  idx = sample(1:length(unique(image.syst$fold)), size = 1)
  image.syst.test = rbind(image.syst.test, image.syst[image.syst$fold==idx, ])
  idx.test = c(idx.test, idx)
}
# create training data
image.syst.train = image.syst[!(image.syst$fold %in% idx.test), ]


buf.split = function(image, width, i){
  xcut = quantile(image$x, c(4/9, 5/9, 1))
  ycut = quantile(image$y, c(4/9, 5/9, 1))
  
  image = image %>%
    mutate(
      level.x = case_when(
        ( x > xcut[1] & x < xcut[2]) ~ 0,
        ( x < xcut[1] - width ) ~ 1,
        ( x > xcut[2] + width )  ~ 2,
        TRUE ~ -1),
      level.y = case_when(
        ( y > ycut[1] & y < ycut[2]) ~ 0,
        ( y < ycut[1] - width) ~ 1,
        ( y > ycut[2] + width) ~ 2,
        TRUE ~ -1)
    )
  
  image = image %>%
    mutate(fold = case_when(
      (level.x*level.y == 0) ~ 0,
      (level.x == 1 & level.y == 1) ~ 4*i-3,
      (level.x == 2 & level.y == 1) ~ 4*i-2,
      (level.x == 1 & level.y == 2) ~ 4*i-1,
      (level.x == 2 & level.y == 2) ~ 4*i,
      TRUE ~ -1)
    )
  return (image %>% select(-c(level.x, level.y)))
}
imagem1.buf = buf.split(imagem1, width = 3, i = 1)
imagem2.buf = buf.split(imagem2, width = 3, i = 2)
imagem3.buf = buf.split(imagem3, width = 3, i = 3)
image.buf.test = rbind(imagem1.buf[imagem1.buf$fold==0,],
                       imagem2.buf[imagem2.buf$fold==0,],
                       imagem3.buf[imagem3.buf$fold==0,]) %>%
  select(-fold) 

# create training data
image.buf.train = rbind(imagem1.buf[!(imagem1.buf$fold %in% c(0,-1)), ],
                        imagem2.buf[!(imagem2.buf$fold %in% c(0,-1)), ],
                        imagem3.buf[!(imagem3.buf$fold %in% c(0,-1)), ])


CVmaster2=function(classifier,xtrain,ytrain,K,loss){
  ## set up
  classifiers=c("logistic","LDA","QDA","Naive Bayes","knn")
  losses=c("accuracy")
  if(!(classifier%in%classifiers)){
    print("Please choose split classifiers: logistic, LDA, QDA, Naive Bayes, knn.")
    break
  }
  if(!(loss%in%losses)){
    print("Please choose loss functions: accuracy.")
    break
  }
  
  df=data.frame(xtrain,label=ytrain)
  data=df%>%filter(label!=0)
  data$label[data$label==-1]=0
  fold=unique(data$fold)
  fold.sample=sample(fold)
  
  ## matrix to save result
  cvresult=matrix(NA,ncol=2,nrow=K)
  colnames(cvresult)=c("fold","acc")
  
  ## K-folds
  for(i in 1:K){
    datatrain=data%>%filter(fold!=fold.sample[i%%K+1])
    dataval=data%>%filter(fold==fold.sample[i%%K+1])
    ## classifiers
    if(classifier=="logistic"){
      glm.fit=glm(label~NDAI+SD+CORR+DF+CF+BF+AF+AN,data=datatrain,family=binomial)
      glm.probs=predict(glm.fit,dataval,type="response")
      glm.pred=rep(0,length(glm.probs))
      glm.pred[glm.probs>0.5]=1
      pred=glm.pred
    }
    if(classifier=="LDA"){
      lda.fit=lda(label~NDAI+SD+CORR+DF+CF+BF+AF+AN,data=datatrain)
      lda.pred=predict(lda.fit,dataval)
      pred=lda.pred$class
    }
    if(classifier=="QDA"){
      qda.fit=qda(label~NDAI+SD+CORR+DF+CF+BF+AF+AN,data=datatrain)
      qda.pred=predict(qda.fit,dataval)
      pred=qda.pred$class
    }
    if(classifier=="Naive Bayes"){
      nb.fit=naiveBayes(label~NDAI+SD+CORR+DF+CF+BF+AF+AN,data=datatrain)
      nb.pred=predict(nb.fit,dataval)
      pred=nb.pred
    }
    if(classifier=="knn"){
      knn.pred=knn(datatrain[,4:11],dataval[,4:11],datatrain$label,k=5)
      pred=knn.pred
    }
    ## loss functions
    if(loss=="accuracy"){
      acc=mean(pred==dataval$label)
    }
    ## save results
    cvresult[i,1]=i
    cvresult[i,2]=acc
  }
  
  return(cvresult)
}