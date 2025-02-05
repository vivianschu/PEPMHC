import numpy as np
import re



pep_path = "/media/patrick/DATA/MHCAttnNet_dataset/1-gram-vectors.txt"

res =[]
with open(pep_path, 'r') as f:
   for line in f:
       for i in range(len(line)):
           if i < len(line) - 1:
                if line[i].isupper():
                    if not line[i+1].isupper() and not line[i-1].isupper():
                        res += line[i]
                    elif line[i+1].isupper():
                        res += [line[i]+line[i+1]]
                    else:
                        res +=[]

                else:
                    res +=[]

print((res))

# # Count the number of words
# num_words = len(words)
#
# print(num_words)



# seq = 'UBVACAFGDHSAHSKASHASKSABGGGGGRRR'
# def tokenize_pep(seq):
#     return list(map("".join, zip(*[iter(seq)]*3)))
#
# print(tokenize_pep(seq))

with open("/home/patrick/Desktop/numbers.txt", "r") as file:
    # Read the contents of the file
    content = file.read()



import re
floats = re.findall(r"[-+]?\d*\.\d+|\d+", content)

numbers = [float(num) for num in floats]
print(len(np.array(numbers)))
# Convert each line into a number and store it in a list

# print(np.array(lines).sum())











#
