import numpy as np
import matplotlib.pyplot as plt


plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Precision-Recall curve')
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#plt.axis([40, 160, 0, 0.03])
plt.plot([0.92,0.57,0.86],[0.91,0.61,0.86])

plt.grid(True)
plt.show()
