
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from utilities import *

# 🔹 Initialization function
def initialize(n0, n1, n2):

    W1 = np.random.randn(n1, n0)
    b1 = np.random.randn(n1, 1)

    W2 = np.random.randn(n2, n1)
    b2 = np.random.randn(n2, 1)

    params = {
        'W1' : W1,
        'b1' : b1,
        'W2' : W2,
        'b2' : b2
    }

    return params

# 🔹 Forward Propagation function
def Forward_Propagation(X, params):

    """
        Z1 = W1.X + b1
        A1 = 1 / (1+exp(-Z1))

        Z2 = W2.A1 + B2
        A2 =  1 / (1+exp(-Z2))
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    Z1 = W1.dot(X) + b1
    Z1 = np.clip(Z1, -500, 500)
    A1 = 1 / (1 + np.exp(-Z1))

    Z2 = W2.dot(A1) + b2
    Z2 = np.clip(Z2, -500, 500)
    A2 = 1 / (1 + np.exp(-Z2))

    activations = {
        'A1' : A1,
        'A2' : A2
    }

    return activations

# 🔹 Log-Loss function
def log_loss(A, y):
    epsilon = 1e-8  # Small value to avoid log(0)
    A = np.clip(A, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(A) + (1 - y) * np.log(1 - A))

# 🔹 Back-Propagation for Gradients calculation
def Back_Propagation(X, y, activations, params):
    """
        dZ2 = A2 - y
        dW2 = (1/m) * dZ2 . (A1)'
        db2 = (1/m) * sum(dZ2) --> mean(dZ2)

        dZ1 = (W2)'.dZ2 * A1(1- A1)
        dW1 = (1/m) * dZ1 . (X)'
        db1 = (1/m) * sum(dZ1) --> mean(dZ1)

    """

    A1 = activations['A1']
    A2 = activations['A2']
    W2 = params['W2']

    m = y.shape[1]

    # 1st Layer Gradients
    dZ2 = A2 - y
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = np.mean(dZ2, axis=1, keepdims=True)

    # 2nd Layer Gradients
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = np.mean(dZ1, axis=1, keepdims=True)


    gradients = {
        'dW2' : dW2,
        'db2' : db2,
        'dW1': dW1,
        'db1': db1
    }

    return gradients


# 🔹 Update function
def update(gradients, params, alpha):

    # Extracting params
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    # Extracting gradiends
    dW1 = gradients['dW1']
    db1 = gradients['db1']
    dW2 = gradients['dW2']
    db2 = gradients['db2']


    W1 -= alpha * dW1
    b1 -= alpha * db1

    W2 -= alpha * dW2
    b2 -= alpha * db2

    updated_params = {
        'W1' : W1,
        'b1' : b1,
        'W2' : W2,
        'b2' : b2
    }

    return updated_params

# 🔹 Prediction function
def predict(X, params):
    activations = Forward_Propagation(X, params)
    A2 = activations['A2']

    return np.round(A2)

# Confusion Matrix + Plot
def Confusion_Matrix(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", cbar=False, linewidths=1, linecolor='black')
    plt.title("Confusion Matrix", fontsize=14, fontweight='bold', color='#333333')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("confusion_matrix.png")
    plt.close()

# 🔹 Training function with Normal visualizations
def N2N_NORMAL(X, y, n1, alpha, epochs):

    """
        n0 : The number of entry parameters
        n1 : The Number of Neurones in the 1st layer
        n2 : The Number of Neurones in the 2nd layer (Output layer)
    """

    n0 = X.shape[0]
    n2 = y.shape[0]

    params = initialize(n0, n1, n2)

    Loss = []
    Acc = []

    for i in tqdm(range(epochs)):
        activations = Forward_Propagation(X, params)
        A2 = activations['A2']
        loss = log_loss(A2, y)

        if i % 10 == 0:
            Loss.append(loss)
            y_pred = predict(X, params)
            accuracy = accuracy_score(y.flatten(), y_pred.flatten())
            Acc.append(accuracy)

        gradients = Back_Propagation(X, y, activations, params)
        params = update(gradients, params, alpha)

    # 🎯 Predictions & Accuracy
    y_pred = predict(X, params)
    accuracy = accuracy_score(y, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # 📈 Stunning Loss + Accuracy Curve

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12, color='tab:red')
    ax1.plot(Loss, label='Loss', color='#FF6F61', linewidth=2, linestyle='dashed')
    ax1.scatter(range(0, len(Loss), len(Loss)//10), Loss[::len(Loss)//10],
                color='blue', edgecolors='black', zorder=3)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, linestyle='dotted', alpha=0.6)

    # Create a second y-axis for Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", fontsize=12, color='tab:blue')
    ax2.plot(Acc, label='Accuracy', color='tab:blue', linewidth=2)
    ax2.scatter(range(0, len(Acc), len(Acc)//10), Acc[::len(Acc)//10],
                color='red', edgecolors='black', zorder=3)
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.suptitle("Training Loss and Accuracy Over Epochs", fontsize=16, fontweight='bold', color='#333333')
    fig.tight_layout()
    plt.show()

    # 🔥 Confusion Matrix
    Confusion_Matrix(y.flatten(), y_pred.flatten())

    return params


# 🔹 Training function with Live visualizations
def N2N_LIVE(X, y, n1, alpha, epochs):
    """
        n0 : The number of entry parameters
        n1 : The Number of Neurones in the 1st layer
        n2 : The Number of Neurones in the 2nd layer (Output layer)
    """

    n0 = X.shape[0]
    n2 = y.shape[0]

    params = initialize(n0, n1, n2)

    Loss = []
    Acc = []

    # Enable interactive mode
    plt.ion()
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    for i in tqdm(range(epochs)):
        activations = Forward_Propagation(X, params)
        A2 = activations['A2']
        loss = log_loss(A2, y)

        if i % 10 == 0:
            Loss.append(loss)
            y_pred = predict(X, params)
            accuracy = accuracy_score(y.flatten(), y_pred.flatten())
            Acc.append(accuracy)

            # Live Plotting
            ax1.clear()
            ax2.clear()

            ax1.set_xlabel("Epochs", fontsize=12)
            ax1.set_ylabel("Loss", fontsize=12, color='tab:red')
            ax1.plot(Loss, label='Loss', color='#FF6F61', linewidth=2, linestyle='dashed')
            ax1.scatter(range(len(Loss)), Loss, color='blue', edgecolors='black', zorder=3)
            ax1.tick_params(axis='y', labelcolor='tab:red')
            ax1.grid(True, linestyle='dotted', alpha=0.6)

            ax2.set_ylabel("Accuracy", fontsize=12, color='tab:blue')
            ax2.plot(Acc, label='Accuracy', color='tab:blue', linewidth=2)
            ax2.scatter(range(len(Acc)), Acc, color='red', edgecolors='black', zorder=3)
            ax2.tick_params(axis='y', labelcolor='tab:blue')

            fig.suptitle("Training Loss and Accuracy Over Epochs", fontsize=16, fontweight='bold', color='#333333')
            fig.tight_layout()
            plt.pause(0.01)  # Pause for a short time to update the figure

        gradients = Back_Propagation(X, y, activations, params)
        params = update(gradients, params, alpha)

    plt.ioff()  # Disable interactive mode when training is done
    plt.show()

    # Final Confusion Matrix
    y_pred = predict(X, params)
    accuracy = accuracy_score(y.flatten(), y_pred.flatten())
    print(f"Model Accuracy: {accuracy:.4f}")

    # 🔥 Confusion Matrix
    Confusion_Matrix(y.flatten(), y_pred.flatten())

    return params




# 📥 Load data
X_train, y_train, X_test, y_test = load_data()

# 🔍 Show Sample Images
plt.figure(figsize=(16, 8))
for i in range(1, 10):
    plt.subplot(4, 5, i)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 🔄 Preprocessing: Reshape and Normalize
# Normalize pixel values to [0, 1]

"""
    Min-Max Normalisation
    X = (X - min(X)) / (max(X) - min(X))
    Black pixel: 0, White pixel: 255
    Min: 0, Max: 255
    X = X / 255.0

"""
X_train_reshape = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test_reshape = X_test.reshape(X_test.shape[0], -1) / 255.0

print("X_train_reshape shape:", X_train_reshape.shape)
print("X_test_reshape shape:", X_test_reshape.shape)


# 🔥 Train The ANN
"""
    Use N2N_NORMAL for Normal visualizations
    Use N2N_LIVE for Live visualizations
"""

# Transpose X_train_reshape to (features, samples): (4096, 1000)
X_train_reshape = X_train_reshape.T
y_train = y_train.reshape(1, -1)

params = N2N_NORMAL(X_train_reshape, y_train, 3, 0.01, 100)