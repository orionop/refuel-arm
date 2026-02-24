"""
Cell 1: Install dependencies on Kaggle.
Run this FIRST in a Kaggle notebook cell:

    !pip install -q pytorch-lightning freia wandb tqdm tabulate
    !pip install -q git+https://github.com/jstmn/Jrl.git@2ba7c3995b36b32886a8aa021a00c73b2cd55b2c
    !git clone https://github.com/orionop/refuel-arm.git || true
    import sys; sys.path.insert(0, "refuel-arm/ikflow")
    import ikflow; print(f"✅ ikflow loaded from: {ikflow.__file__}")

Then run cells 2, 3, 4 as:

    %run refuel-arm/kaggle/cell_2_register_robot.py
    %run refuel-arm/kaggle/cell_3_generate_dataset.py
    %run refuel-arm/kaggle/cell_4_train.py
"""
