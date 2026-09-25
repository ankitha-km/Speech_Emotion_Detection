from inference import predict_emotion


AUDIO_FILE = "temp_upload.wav"


print("\n==============================")
print(" Kannada Speech Emotion Test")
print("==============================\n")


emotion, probabilities = predict_emotion(
    AUDIO_FILE
)


print("Predicted Emotion:", emotion)

print("\nProbabilities:")

for i, probability in enumerate(probabilities):

    print(
        f"{i}: {probability:.4f}"
    )

print("\n==============================")