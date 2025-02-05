import matplotlib.pyplot as plt
import re

'''
# open and read the output file
with open('/home/patrick/Desktop/pretraining_LLM_48.txt', 'r') as f:
    data = f.read()

# use regular expressions to find all losses and epochs
loss_pattern = "'loss': (\d+\.\d+)"
epoch_pattern = "'epoch': (\d+\.\d+)"

losses = re.findall(loss_pattern, data)
epochs = re.findall(epoch_pattern, data)

# convert strings to floats
losses = [float(i) for i in losses]
epochs = [float(i) for i in epochs]

# plot the data
plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, marker='*')
plt.title('mlm_0.25_maxlen_48')
#plt.title('Loss vs Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.show()
'''

epochs = []
roc_aucs = []

with open("/home/patrick/Desktop/prot_bert_48_350_mlm_0.15_216000_new_split_scratch.txt", 'r') as f:  # replace with your output file path
    for line in f:
        if "'eval_ROC_AUC'" in line:
            match = re.search("'epoch': (\d+.\d+)", line)
            if match:
                epochs.append(float(match.group(1)))
            match = re.search("'eval_ROC_AUC': (\d+.\d+)", line)
            if match:
                roc_aucs.append(float(match.group(1)))

plt.plot(epochs, roc_aucs)
plt.xlabel('Epoch')
plt.ylabel('ROC_AUC')
plt.title('prot_bert_48_350_mlm_0.15_96000')
plt.show()





















