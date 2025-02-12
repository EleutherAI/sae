import os
import shutil
from pathlib import Path

def reorganize_directories(base_path):
    base_path = Path(base_path)
    # Track moves to avoid duplicates
    planned_moves = {}
    
    # First, find all directories and plan moves
    for layer_dir in base_path.glob("layers.*.mlp"):
        for step_dir in layer_dir.glob("*"):
            if not step_dir.is_dir() or not step_dir.name.isdigit():
                continue
                
            step_num = step_dir.name
            layer_name = layer_dir.name
            
            # New location
            new_parent = base_path / step_num
            new_location = new_parent / layer_name
            
            # Store the planned move
            if step_dir not in planned_moves:
                planned_moves[step_dir] = new_location

    # Execute moves
    for old_path, new_path in planned_moves.items():
        # Create parent directory if it doesn't exist
        new_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move the directory
        print(f"Moving {old_path} to {new_path}")
        shutil.move(str(old_path), str(new_path))
        
        # Remove empty source directories
        try:
            old_path.parent.rmdir()
        except OSError:
            # Directory not empty, skip
            pass

if __name__ == "__main__":
    # Example usage
    base_dir = "/mnt/ssd-1/lucia/sparsify/const-warmup"
    reorganize_directories(base_dir)
    