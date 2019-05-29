 import scipy as sci
 import matplotlib.pyplot as plt
 from tqdm import tqdm
    
 dict = {}

 # Change the value of your training folder directory.
 folderDirectory = "C:\\Users\\ettag\\Desktop\\HDA\\training2017\\"
    
 for i in tqdm(range(1,8529,1)):
     
     index = "A" + str(i).zfill(5)
    
     path = folderDirectory + index
     mat = scipy.io.loadmat(path, squeeze_me = True)
    
     values = mat['val']
     dict[index] = values

# Check the length of the dictionary corresponds to the number of files in the folder.
len(dict)

# Plot a random ECG in order to ensure the loading was performed correctly.
plt.plot(dict['A00004'])