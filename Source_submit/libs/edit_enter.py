import os


def create_enter_file(fold_number, filter_train=True, use_val_is_train=False):
    """
    Create or update enter.py with validation settings.

    Args:
        fold_number (int): The fold number to use for validation
    """
    content = f'''import os

VAL_SELECTION = "val/fold_{fold_number}"
FILTER_VAL_IMAGES = {filter_train}
USE_VAL_IS_TRAIN = {use_val_is_train}
FINAL_TO_USE_DIR = "final_to_use"
'''

    try:
        with open('libs/enter.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print(
            f"[*] Successfully created/updated enter.py with fold {fold_number}, "
            f"\n- filter_train={filter_train}, "
            f"\n- use_val_is_train={use_val_is_train}"
        )
        return True
    except Exception as e:
        print(f"Error creating/updating enter.py: {str(e)}")
        return False


# Example usage:
if __name__ == "__main__":
    # Example 1: Replace content
    file_path = "enter.py"
    fold_number = 1  # Change this to desired fold number
    create_enter_file(fold_number)
