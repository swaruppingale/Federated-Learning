


import matplotlib.pyplot as plt

# Original labels
original_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Flipping labels
label = original_label.copy()
for i in range(len(label)):
    if label[i] == 0:
        label[i] = 4
    elif label[i] == 1:
        label[i] = 5
    elif label[i] == 2:
        label[i] = 7
    elif label[i] == 3:
        label[i] = 6
    elif label[i] == 4:
        label[i] = 0
    elif label[i] == 5:
        label[i] = 3
    elif label[i] == 6:
        label[i] = 8
    elif label[i] == 7:
        label[i] = 9
    elif label[i] == 8:
        label[i] = 1
    elif label[i] == 9:
        label[i] = 2
    else:
        print("byeee")





# Plotting the original and flipped labels on a single line graph
plt.figure(figsize=(10, 6))

plt.plot(range(len(original_label)), original_label, label='Original Labels', marker='o', color='blue')
plt.plot(range(len(label)), label, label='Flipped Labels', marker='o', color='red',linestyle="dotted")

plt.title('Original and Flipped Labels')
plt.xlabel('Index')
plt.ylabel('Label')
plt.xticks(range(len(original_label)))
plt.yticks(range(10))
plt.legend()

# Display the plot
plt.grid(True)
plt.show()

        