import numpy as np

# data.py
phrases = [
    "hello", "hi", "how are you", "hey",                        # Greetings
    "bye", "goodbye", "see you later",                          # Farewell
    "thanks", "thank you", "appreciate it",                     # Thanks
    "can I have information", "I need details", "tell me more"  # Request info
]

intents = np.array([
    [1, 0, 0, 0],  # Greeting
    [1, 0, 0, 0],  # Greeting
    [1, 0, 0, 0],  # Greeting
    [1, 0, 0, 0],  # Greeting
    [0, 1, 0, 0],  # Farewell
    [0, 1, 0, 0],  # Farewell
    [0, 1, 0, 0],  # Farewell
    [0, 0, 1, 0],  # Thank you
    [0, 0, 1, 0],  # Thank you
    [0, 0, 1, 0],  # Thank you
    [0, 0, 0, 1],  # Request info
    [0, 0, 0, 1],  # Request info
    [0, 0, 0, 1]   # Request info
])
