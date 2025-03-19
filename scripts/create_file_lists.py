# scripts/create_file_lists.py
import glob

if __name__ == "__main__":
    train_lstmf_files = glob.glob('data/train_preprocessed/*.lstmf')
    with open('data/train_preprocessed/train_files.txt', 'w') as f:
        f.write('\n'.join(train_lstmf_files))
    
    val_lstmf_files = glob.glob('data/val_preprocessed/*.lstmf')
    with open('data/val_preprocessed/val_files.txt', 'w') as f:
        f.write('\n'.join(val_lstmf_files))
    
    print(f"Number of training .lstmf files: {len(train_lstmf_files)}")
    print(f"Number of validation .lstmf files: {len(val_lstmf_files)}")