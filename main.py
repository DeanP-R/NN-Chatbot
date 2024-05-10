from chatbot_nn.neural_network import NeuralNetwork
from chatbot_nn.utils import encode_phrases
from chatbot_nn.data import phrases, intents


# Train the neural network
def main():
    # Encode phrases
    X_train, vectorizer = encode_phrases(phrases)

    # Neural network parameters
    input_size = X_train.shape[1]
    hidden_size = 5  # This can be adjusted based on the complexity of the problem
    output_size = 4  # Adjust based on number of classes/intents

    # Initialize and train the neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.train(X_train, intents, epochs=1000)

    # Interaction loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Exiting...")
            break

        # Encoding the user input using the same vectorized used during training
        test_encoded = vectorizer.transform([user_input]).toarray()
        prediction = nn.feedforward(test_encoded)

        print(f'Predicted Intent: {prediction}')


if __name__ == "__main__":
    main()
