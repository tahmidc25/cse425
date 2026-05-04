import os
import re

files_to_patch = [
    "src/training/train_task1.py",
    "src/training/train_task2.py",
    "src/training/train_task3.py",
    "src/training/train_task4.py"
]

for fpath in files_to_patch:
    if not os.path.exists(fpath):
        continue
    with open(fpath, "r") as f:
        content = f.read()
    
    # Replace NUM_EPOCHS to 50
    content = re.sub(r'NUM_EPOCHS\s*=\s*\d+', 'NUM_EPOCHS = 50', content)
    
    # Replace dataset creation logic
    if "train_task3.py" in fpath:
        # Task 3 is tokens
        replace_with = """    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.dataset_loader import create_real_dataset
    print("Creating real dataset...")
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    train_dataset, val_dataset = create_real_dataset(str(data_dir), batch_size=BATCH_SIZE, seq_len=512, is_tokens=True)
"""
        content = re.sub(r'    print\("Creating dataset..."\)\n    train_tensor, val_tensor = create_dummy_token_dataset\(num_samples=100\)\n    \n    train_dataset = TensorDataset\(train_tensor\)\n    val_dataset = TensorDataset\(val_tensor\)', replace_with, content)
    elif "train_task4.py" in fpath:
        replace_with = """    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.dataset_loader import create_real_dataset
    print("Creating real dataset...")
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    train_dataset, _ = create_real_dataset(str(data_dir), batch_size=BATCH_SIZE, seq_len=256, is_tokens=False)
"""
        content = re.sub(r'    print\("Creating dataset for reward model training..."\)\n    train_tensor = create_dummy_dataset\(num_samples=50\)\n    train_dataset = TensorDataset\(train_tensor\)', replace_with, content)
    else:
        replace_with = """    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.dataset_loader import create_real_dataset
    print("Creating real dataset...")
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    train_dataset, val_dataset = create_real_dataset(str(data_dir), batch_size=BATCH_SIZE, seq_len=256, is_tokens=False)
"""
        content = re.sub(r'    print\("Creating dataset..."\)\n    train_tensor, val_tensor = create_dummy_dataset\(num_samples=100\)\n    \n    train_dataset = TensorDataset\(train_tensor\)\n    val_dataset = TensorDataset\(val_tensor\)', replace_with, content)
    
    # Drop_last in dataloaders
    content = content.replace("shuffle=True)", "shuffle=True, drop_last=True)")
    
    with open(fpath, "w") as f:
        f.write(content)
