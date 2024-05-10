import numpy as np

# Example data
phrases = [
    "hello",
    "hi",
    "how are you",
    "bye",
    "goodbye",
    "see you later"
]

# Corresponding intents (1-hot encoded)
intents = np.array([
    [1, 0],  # Greeting
    [1, 0],  # Greeting
    [1, 0],  # Greeting
    [0, 1],  # Farewell
    [0, 1],  # Farewell
    [0, 1]   # Farewell
])
