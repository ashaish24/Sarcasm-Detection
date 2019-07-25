import numpy as np
import csv
import re

def preprocessing(csv_file_object):
    
    data=[]
    length=[]
    remove_hashtags = re.compile(r'#\w+\s?')
    remove_friendtag = re.compile(r'@\w+\s?')
     

    for row in csv_file_object:
        if len(row[0:])==1:
            temp=row[0:][0]
            temp=remove_hashtags.sub('',temp)
            if len(temp)>0 and 'http' not in temp and temp[0]!='@' and '\x' not in temp: 
                temp=remove_friendtag.sub('',temp)
                
                temp=' '.join(temp.split()) 
                if len(temp.split())>2:
                    data.append(temp)
                    length.append(len(temp.split()))
    data=list(set(data))
    data = np.array(data)    
           

    
    print(data)

    print (re.compile(".*:(.*)\x.*").match(data).groups())

print ('Extracting data')


csv_file_object_pos = csv.reader(open('D:\LSPapers\TwitterData\BCCI_tweets.csv', 'rU',encoding="UTF-8"),delimiter='\n')
preprocessing(csv_file_object_pos)  


