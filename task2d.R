CVmaster=function(method,classifier,xtrain,ytrain,K,loss){
  ## set up
  methods=c("systematic","buffering")
  classifiers=c("logistic","LDA","QDA","Naive Bayes","knn")
  losses=c("accuracy")
  if(!(method%in%methods)){
    print("Please choose split methods: systematic, buffering.")
    break
  }
  if(!(classifier%in%classifiers)){
    print("Please choose split classifiers: logistic, LDA, QDA, Naive Bayes, knn.")
    break
  }
  if(!(loss%in%losses)){
    print("Please choose loss functions: accuracy.")
    break
  }
  
  ## get split data
  if(method=="systematic"){
    data=image.syst.train%>%filter(label!=0)
    fold.syst=unique(image.syst.train$fold)
    fold.sample=sample(fold.syst)
  }
  if(method=="buffering"){
    data=image.buf.train%>%filter(label!=0)
    fold.buf=unique(image.buf.train$fold)
    fold.sample=sample(fold.buf)
  } 
  data$label[data$label==-1]=0
  
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