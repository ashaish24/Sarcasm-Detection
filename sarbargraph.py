import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('Precision', 'Recall', 'F1-score')
y_pos = np.arange(len(objects))
performance = [0.61,0.57,0.59]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Percentage')
plt.ylim(0,1)
plt.title('Performance of classifier on sarcastic tweets dataset')
 
plt.show()
