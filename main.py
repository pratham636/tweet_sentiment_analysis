import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
def clean_text(text):
    text=text.lower()
    text=text.strip()
    text=re.sub(r"http\S+|www\S+|@\S+","",text)
    text=re.sub(r"[^a-z,A-Z\s]","",text)
    return text
    
def pred_sentiment(text):
    text=clean_text(text)
    vector=vectorizer.transform([text])
    # return model.predict(vector)
    label=[lr,nb,svm]
    answer=[]
    for i in label:
        if i.predict(vector)[0]==1:
            answer.append("Positive")
        else:
            answer.append("Negative")
    return answer

def len_text(text):
    return len(text)

df=pd.read_csv("data/tweets.csv")

df["clean_tweet"]=df["text"].apply(clean_text)
df["target"]=df["target"].replace(4,1)

print(df.shape)
print(df['target'].value_counts())


df["length"]=df["clean_tweet"].apply(len_text)
count=df.groupby("target")["length"].mean()
print(count)

# print(df["length"])

labels=df.groupby("target")['clean_tweet'].count()
my_labels=["Negative","Positive"]
labels.plot(kind="pie",autopct="%1.1f%%",labels=my_labels)
plt.title("Labels in Dataset")
# plt.legend()
plt.savefig("charts/pie_chart.png")
plt.show()




vectorizer= TfidfVectorizer(ngram_range=(1,2),max_features=50000)


X=vectorizer.fit_transform(df["clean_tweet"])

Y=df["target"]


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

lr=LogisticRegression(class_weight="balanced")
lr.fit(x_train,y_train)

nb=MultinomialNB(class_prior=[0.5,0.5])
nb.fit(x_train,y_train)

svm=LinearSVC(class_weight="balanced")
svm.fit(x_train,y_train)

pred1=lr.predict(x_test)
print("Accuracy of Logistic Regression:",accuracy_score(y_test,pred1))
print("Precision of Logistic Regression:",precision_score(y_test,pred1,average="binary"))
print("Recall of Logistic Regression:",recall_score(y_test,pred1,average="binary"))
print("F1-Score of Logistic Regression:",f1_score(y_test,pred1,average="binary"))
print("Confusion Matric of Logistic Regression:", confusion_matrix(y_test,pred1))

pred2=nb.predict(x_test)
print("Accuracy of Naive Bayes:",accuracy_score(y_test,pred2))
print("Precision of Naive Bayes:",precision_score(y_test,pred2,average="binary"))
print("Recall of Naive Bayes:",recall_score(y_test,pred2,average="binary"))
print("F1-Score of Naive Bayes:",f1_score(y_test,pred2,average="binary"))
print("Confusion Matric of Naive Bayes:", confusion_matrix(y_test,pred2))

pred3=svm.predict(x_test)
print("Accuracy of SVM:",accuracy_score(y_test,pred3))
print("Precision of SVM:",precision_score(y_test,pred3,average="binary"))
print("Recall of SVM:",recall_score(y_test,pred3,average="binary"))
print("F1-Score of SVM:",f1_score(y_test,pred3,average="binary"))
print("Confusion Matric of SVM:", confusion_matrix(y_test,pred3))

# print(df.head())

answers=pred_sentiment("I love this product")
print(f"Accuracy of Logistic Regression:{answers[0]}\nAccuracy of Naive Bayes:{answers[1]}\nAccuracy of SVM:{answers[2]}")

