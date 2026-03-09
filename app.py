import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="AI Image Colorization App", layout="wide")
st.title("AI Image Colorization App 🎨")

st.write("Upload a black & white image and see it colorized using a pre-trained CNN model.")

# Upload image
uploaded_file = st.file_uploader("Upload Black & White Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    # Read uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        st.error("Error: Could not read image. Try another file.")
    else:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load pre-trained CNN model
        prototxt = "colorization_deploy_v2.prototxt"
        model = "colorization_release_v2.caffemodel"
        pts = "pts_in_hull.npy"

        try:
            net = cv2.dnn.readNetFromCaffe(prototxt, model)
            cluster_points = np.load(pts)

            class8 = net.getLayerId("class8_ab")
            conv8 = net.getLayerId("conv8_313_rh")

            cluster_points = cluster_points.transpose().reshape(2, 313, 1, 1)
            net.getLayer(class8).blobs = [cluster_points.astype(np.float32)]
            net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

            # Prepare image for CNN
            scaled = image.astype("float32") / 255.0
            lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
            L = lab[:, :, 0]

            L_resized = cv2.resize(L, (224, 224))
            L_resized -= 50

            net.setInput(cv2.dnn.blobFromImage(L_resized))
            ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
            ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

            colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
            colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
            colorized = np.clip(colorized, 0, 1)

            st.subheader("Colorized Image")
            st.image(colorized, use_column_width=True)

            # Download button
            colorized_uint8 = (colorized * 255).astype(np.uint8)
            colorized_pil = Image.fromarray(cv2.cvtColor(colorized_uint8, cv2.COLOR_BGR2RGB))
            st.download_button(
                label="Download Colorized Image",
                data=cv2.imencode(".jpg", colorized_uint8)[1].tobytes(),
                file_name="colorized_output.jpg",
                mime="image/jpeg"
            )

        except Exception as e:
            st.error(f"Error loading CNN model: {e}")