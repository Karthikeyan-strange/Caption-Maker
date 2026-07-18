import streamlit as st
from PIL import Image
import os

from Flourence_model import generate_caption

st.set_page_config(
    page_title="Offline Image Caption Generator",
    layout="wide"
)

st.title("📷 Offline Image Caption Generator")
st.write("Powered by Florence-2 Base")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    os.makedirs("uploads", exist_ok=True)

    image = Image.open(uploaded_file)

    image_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    image.save(image_path)

    st.image(image, use_container_width=True)

    if st.button("Generate Caption"):

        with st.spinner("Generating caption..."):

            caption = generate_caption(image_path)

        st.success("Done!")

        st.subheader("Generated Caption")

        st.write(caption)

        os.makedirs("outputs", exist_ok=True)

        with open("outputs/caption.txt", "w") as f:
            f.write(caption)

        st.download_button(
            "Download Caption",
            caption,
            file_name="caption.txt"
        )