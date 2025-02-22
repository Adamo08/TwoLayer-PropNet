# Neural Network Classifier (N2N) 🐾✨

This project implements a **two-layer artificial neural network (ANN)** for binary classification using NumPy, with dazzling training visualizations powered by Matplotlib and Seaborn. Whether you prefer static 📊 or live 🎥 updates, this code tracks loss, accuracy, and even throws in a confusion matrix to show off its skills. The dataset—featuring adorable cats 🐱 and dogs 🐶—is loaded from HDF5 files via a handy function in `utilities.py`.

The neural network rocks:
- An **Input Layer** (sized by the dataset features)
- A **Hidden Layer** (you pick the neuron count! 🧠)
- An **Output Layer** (cat or dog? Binary magic! 🎯)

## Features 🌟
- Forward & backward propagation for training 🚀
- Log-loss cost function 📉
- Gradient descent optimization ⚙️
- Accuracy tracking + confusion matrix fun 🎨
- Loss & accuracy visuals over epochs (static or live!) 🌈
- Dataset preprocessing (reshaping & normalizing) 🔧

## Prerequisites ✅
You’ll need Python 3.6+ installed 🐍. Check `requirements.txt` for the full list of goodies.

### Installation 🛠️
1. Clone or grab this repo:
   ```bash
   git clone https://github.com/Adamo08/TwoLayer-PropNet.git
   cd TwoLayer-PropNet
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure `datasets/` has `trainset.hdf5` and `testset.hdf5` ready to roll! 🐾

### Requirements 📦
Here’s what’s in `requirements.txt`:
```
numpy
matplotlib
seaborn
scikit-learn
tqdm
h5py
```

Get them with:
```bash
pip install matplotlib seaborn scikit-learn tqdm numpy h5py
```

## File Structure 📂
```
├── datasets/
│   ├── trainset.hdf5    # Training dataset (cat & dog pix! 🐱🐶)
│   └── testset.hdf5     # Test dataset (more furry friends!)
├── utilities.py         # Helper functions (e.g., load_data) 🛠️
├── app.py               # Main script with ANN awesomeness 🌟
├── requirements.txt     # Required Python modules 📋
└── README.md            # You’re here! 👋
```

### Dataset 🐱🐶
The dataset is stored in HDF5 format and packed with images of **cats** and **dogs** for binary classification (cat = 0, dog = 1, or vice versa—your call!). It includes:
- **Training Set**: `X_train` (images) & `y_train` (labels)
- **Test Set**: `X_test` (images) & `y_test` (labels)

Loaded via `load_data()` in `utilities.py`. 😻

## Usage 🚀
1. Check that all files are in place.
2. Tweak hyperparameters in `app.py` if you’re feeling fancy:
   - `n1`: Hidden layer neurons (default: 3) 🧠
   - `alpha`: Learning rate (default: 0.01) ⚡
   - `epochs`: Training rounds (default: 100) ⏳
3. Fire it up:
   ```bash
   python app.py
   ```

### Training Options 🎛️
- **`N2N_NORMAL`**: Trains and shows static plots—loss, accuracy, and a confusion matrix—when done. 📈
- **`N2N_LIVE`**: Trains with live-updating loss & accuracy plots, plus a confusion matrix at the end. 🎬

Switch it up in `app.py`:
```python
# Static vibes
params = N2N_NORMAL(X_train_reshape, y_train, 3, 0.01, 100)

# Live action
params = N2N_LIVE(X_train_reshape, y_train, 3, 0.01, 100)
```

### Outputs 🎉
- **Console**: Final model accuracy 🏆
- **Plots**:
  - Loss & accuracy curves (saved during training) 📉📈
  - Confusion matrix (saved as `confusion_matrix.png`) 🎨
- **Sample Images**: A cute grid of cat & dog pics with labels before training starts! 🖼️

## Code Overview 🔍
### Key Functions
- `initialize`: Sets up weights & biases ⚖️
- `Forward_Propagation`: Runs data through the network ➡️
- `log_loss`: Measures the cost 📊
- `Back_Propagation`: Calculates gradients ⬅️
- `update`: Tweaks params with gradient descent 🔧
- `predict`: Makes cat-or-dog calls 🐾
- `Confusion_Matrix`: Plots the results 🎯
- `N2N_NORMAL` / `N2N_LIVE`: Trains with style! 🌟

### Preprocessing 🛠️
- **Reshaping**: Flattens images (e.g., `(samples, height, width)` → `(samples, features)`) 📏
- **Normalization**: Scales pixels from `[0, 255]` to `[0, 1]` 🌈

## Example 🌟
Say your dataset has 1000 cat & dog pics (64x64 grayscale):
1. `X_train` shape: `(1000, 64, 64)`
2. After reshaping: `(1000, 4096)`
3. After transposition: `(4096, 1000)`
4. `y_train` shape: `(1, 1000)`

Run:
```bash
python app.py
```
Trains with 3 hidden neurons, a 0.01 learning rate, and 100 epochs. 🐱🐶

## Notes 📝
- Built for binary classification (cat vs. dog, `n2 = 1`) 🐾
- For bigger datasets, bump up `n1`, `alpha`, or `epochs` 🔧
- Live plotting (`N2N_LIVE`) might lag a bit due to real-time updates ⏱️

## License 📜
Unlicensed—free to use or tweak for fun & learning! 🎓

## Acknowledgments 🙌
Inspired by classic neural network ideas and powered by Python’s awesome data science tools. 😎
