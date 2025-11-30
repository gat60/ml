import csv
print("\n the most general hypothesis : ['?','?','?','?','?','?']")
print(" the most specific hypothesis : ['0','0','0','0','0','0']\n")
a=[]
with open('enjoysport.csv','r') as csvFile:
    reader=csv.reader(csvFile)
    for row in reader:
        a.append(row)
        print(row)
    csvFile.close()

num_attributes=len(a[0])-1
print('\ninitial value of the hypothesis:')
hypothesis=['0']* num_attributes
print(hypothesis)

for j in range (0,num_attributes):
    hypothesis[j]=a[1][j]
    
print('\n Find S: finding maximally specific hypothesis ')
for i in range(1,len(a)):
    if a[1][num_attributes]=='yes':
        for j in range(num_attributes):
            if a[i][j] != hypothesis[j]:
                hypothesis[j]='?'
            else:
                hypothesis[j]==a[i][j]
    print('for training example no :',format(i),'the hypothesis is ',hypothesis)
    
print('\nthe maximally specific hypothesis is:')
print(hypothesis)
