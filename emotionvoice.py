import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np
from keras.models import model_from_json

def predict_emotion(file_path, model):
    try:
        y, sr = librosa.load(file_path, sr=None, duration=3)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        input_data = np.expand_dims(np.expand_dims(mfccs, axis=0), axis=-1)
        prediction = model.predict(input_data)

        emotions = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Suprised", "Sad"]
        predicted_emotion = emotions[np.argmax(prediction)]

        result_label.config(text=f"Predicted Emotion: {predicted_emotion}")
        
    except:
        result_label.config(text="No Emotion Detected")
        

def predict_button():
    file_path = file_path_label.cget("text")
    if file_path:
        predict_emotion(file_path, voice_model)

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav;*.mp3")])
    if file_path:
        file_path_label.config(text=file_path)
        

app = tk.Tk()
app.title("Emotion Detection from Voice")
app.geometry('800x600')
app.configure(background='#CDCDCD')

with open("lstm_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
voice_model = model_from_json(loaded_model_json)
voice_model.load_weights("lstm_model_weights.h5")

heading = tk.Label(app, text='Emotion Detection from Voice', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

file_path_label = tk.Label(app, text="", font=('arial', 14))
file_path_label.pack(side='top', pady=10)


upload_button = tk.Button(app, text="Upload Voice Data", command=browse_file, padx=10, pady=5)
upload_button.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload_button.pack(side='top', pady=20)

result_label = tk.Label(app, text="Predicted Emotion: ", font=('arial', 20, 'bold'))
result_label.pack(side='top', pady=20)

predict_button = tk.Button(app, text="Predict Emotion", command=predict_button, padx=10, pady=5)
predict_button.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
predict_button.pack(side='top', pady=10)

app.mainloop()
