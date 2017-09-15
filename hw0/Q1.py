import sys
path = sys.argv[1]

file_words = open(path,'r')
content = file_words.read()
content_split = content.split( )

labels = []
counts = []

for i in range(len(content_split)):
    if content_split[i] in labels:
        counts[labels.index(content_split[i])] += 1
    else:
        labels.append(content_split[i])
        counts.append(1)

file_output = open('Q1.txt','w')
for i in range(len(labels)):
    file_output.write(str(labels[i]) + " " + str(i) + " " + str(counts[i]))
    if i != len(labels)-1:
        file_output.write("\n")       
file_output.close()



    

