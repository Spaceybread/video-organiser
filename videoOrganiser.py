import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import tree
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

vt = CountVectorizer()

#userInput = input('Enter a video title: ')

filePath = '/Users/vivekaggarwal/Desktop/AI Workshop/Video Organiser/GBvideos.csv'
data = pd.read_csv(filePath)

#data = data.drop([38916], axis=0)

y = 0
outcomes = []
titleName = []

while y != (len(data) - 2):
    outcomes.append(data['category_id'][y])
    titleName.append(data['title'][y])
    y = y + 1

x = 0

titleName = vt.fit_transform(titleName)

numTitle = []
splitTitle = []

vocab_size = 10000

print('Pre-processing is complete')

trainingVal = 0.75*len(numTitle)
trainData = []
trainOut = []

x = 0 

x_train, x_test, y_train, y_test = train_test_split(titleName, outcomes, train_size=.75)
print('Training data and testing data has been processed')

clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)

yPrediction = clf.predict(x_test)

score = accuracy_score(y_test, yPrediction)
print(score)



# Option A -> userInput = 'Everything Wrong With Fantastic Beasts: The Crimes of Grindelwald'
# Option B -> userInput = 'Cuphead'
# Option C -> userInput = 'Cuphead Cartoon'
# Option D -> userInput = 'BTS'
# Option E -> userInput = 'The iPhone'
# Option F -> userInput = 'Life Noggin'

userInput = 'Life Noggin'
userInput = [userInput]
name = vt.transform(userInput)
prediction = clf.predict(name)
prob = clf.predict_proba(name[0])
print('This video is category:', prediction[0],'The probability of this being correct is:', sum(sum(prob))*100)