
# Python program to demonstrate 
# glob using different wildcards 
import glob 
  
print('Named explicitly:') 
for name in glob.glob("C:\\Users\\maria\\OneDrive\\Documentos\\Coding\\Github\\computer-vision-pattern-recognition\\images\\FD_01.jpg"): 
    print(name) 
  
# Using '*' pattern  
print('\nNamed with wildcard *:') 
for name in glob.glob("C:\\Users\\maria\\OneDrive\\Documentos\\Coding\\Github\\computer-vision-pattern-recognition\\images\\*.jpg"): 
    print(name) 
  
# Using '?' pattern 
print('\nNamed with wildcard ?:') 
for name in glob.glob("C:\\Users\\maria\\OneDrive\\Documentos\\Coding\\Github\\computer-vision-pattern-recognition\\images\\FD_01?.jpg"): 
    print(name) 
  
# Using [0-9] pattern 
print('\nNamed with wildcard ranges:') 
for name in glob.glob("C:\\Users\\maria\\OneDrive\\Documentos\\Coding\\Github\\computer-vision-pattern-recognition\\images\\*[0-9].*"): 
    print(name) 