import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from PIL import Image

def main():
    st.title("Image Classifier")
    st.text("Upload an image to classify")

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'gif', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')  # Ensure the image is in RGB format
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")

        # Resize the image to 32x32 pixels
        resized_image = image.resize((32, 32))
        img_arr = np.array(resized_image) / 255.0

        # Ensure the image array has the shape (32, 32, 3)
        if img_arr.shape != (32, 32, 3):
            st.error("Image must have three channels (RGB).")
            return

        # Reshape the array to match the model's expected input shape
        img_arr = img_arr.reshape((1, 32, 32, 3))

        # Load the pre-trained model
        model = tf.keras.models.load_model('cifar10.model.keras')

        # Make a prediction
        prediction = model.predict(img_arr)
        cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # Plot the prediction probabilities
        fig, ax = plt.subplots()
        y_pos = np.arange(len(cifar10_classes))
        ax.barh(y_pos, prediction[0], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cifar10_classes)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Probability')
        ax.set_title('Prediction')
        st.pyplot(fig)

        # Display the top prediction
        st.write(f'The model predicts that this is a {cifar10_classes[np.argmax(prediction)]} with a probability of {np.max(prediction) * 100:.2f}%')
        st.write("")

    else:
        st.text('You Have Not Selected an Image Yet.')

if __name__ == '__main__':
    main()
