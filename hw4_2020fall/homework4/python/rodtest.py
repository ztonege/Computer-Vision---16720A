from submission import rodrigues
from submission import invRodrigues
import numpy as np

a = np.array(([[1/2], 
               [1/3], 
               [1/4]]))

               # pi/2??


print(invRodrigues(rodrigues(a)))

