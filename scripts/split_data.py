# scripts/split_data.py
import pickle
import random

if __name__ == "__main__":
    # Load cleaned_pairs
    with open('data/cleaned_pairs.pkl', 'rb') as f:
        cleaned_pairs = pickle.load(f)
    
    # Split the data
    random.shuffle(cleaned_pairs)
    total = len(cleaned_pairs)
    train = cleaned_pairs[:int(0.8 * total)]
    val = cleaned_pairs[int(0.8 * total):int(0.9 * total)]
    test = cleaned_pairs[int(0.9 * total):]
    
    # Save the splits
    splits = {
        'train': train,
        'val': val,
        'test': test
    }
    with open('data/splits.pkl', 'wb') as f:
        pickle.dump(splits, f)
    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")