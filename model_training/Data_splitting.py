#%%
import os
import shutil
import random

#%%
# Directories for original data
negative_dir = 'c:\\Users\\Shuab.Akintola\\Desktop\\Pytorch_Ultimate\\Negative'  # Path to your 'negative' folder
positive_dir = 'c:\\Users\\Shuab.Akintola\\Desktop\\Pytorch_Ultimate\\Positive'  # Path to your 'positive' folder
#%%
# Directories for output
train_dir = 'train'
test_dir = 'test'

#%%
os.getcwd()

#%%
# Create output subdirectories
for category in ['negative', 'positive']:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

def split_data(source_dir, train_dest, test_dest, split_ratio=0.8):
    # Get all file names
    all_files = os.listdir(source_dir)
    random.shuffle(all_files)  # Shuffle the data to ensure randomness

    split_index = int(len(all_files) * split_ratio)
    train_files = all_files[:split_index]
    test_files = all_files[split_index:]

    # Move the files
    for file_name in train_files:
        shutil.move(os.path.join(source_dir, file_name), os.path.join(train_dest, file_name))

    for file_name in test_files:
        shutil.move(os.path.join(source_dir, file_name), os.path.join(test_dest, file_name))

# Split negative images
split_data(negative_dir, os.path.join(train_dir, 'negative'), os.path.join(test_dir, 'negative'))

# Split positive images
split_data(positive_dir, os.path.join(train_dir, 'positive'), os.path.join(test_dir, 'positive'))

print("Data splitting complete!")

# %%
train.positive.size
# %%
def count_files(directory):
    total_files = sum([len(files) for _, _, files in os.walk(directory)])
    return total_files
# %%
train_size = count_files(train_dir)
test_size = count_files(test_dir)

print("Data splitting complete!")
print(f"Total training images: {train_size}")
print(f"Total testing images: {test_size}")
# %%
