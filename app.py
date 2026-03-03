import streamlit as st
import replicate
import os
import requests
from PIL import Image
import io
import time
import base64

st.set_page_config(page_title="Leather Pet Figurine", page_icon="🐾", layout="centered")
st.title("🐾 Leather Pet Figurine Generator")
st.markdown("Upload a photo of your dog and receive a beautiful handcrafted leather figurine!")

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")
if not REPLICATE_API_TOKEN:
    st.error("Replicate API token not found.")
    st.stop()

PROMPT = "Convert this dog photo into a Zuny-style leather craft figurine product. The entire animal body must be re-sculpted from flat panels of smooth matte synthetic leather stitched together at the seams. No fur, no hair, no skin texture anywhere. Every surface is leather. The body is plump, simplified and rounded like a stuffed leather cushion sewn into an animal shape. The color zones of the leather panels must match the color distribution of the original dog. Visible heavy white stitching runs along every panel edge and seam. The eyes are small circular metallic rivets. The nose is a flat leather oval. The legs are short solid leather tubes. The tail is a simple leather cone. Full body shown standing on all four legs in three-quarter view. Plain white background. Studio product photography lighting. The result must look exactly like a handmade leather craft toy sold in a design store, not a realistic dog or a plastic toy."

uploaded_file = st.file_uploader("Upload your dog's photo (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original photo", use_column_width=True)

    if st.button("Generate Leather Figurine"):
        with st.spinner("Crafting your leather figurine... please wait up to 3 minutes..."):
            try:
                image.thumbnail((768, 768))
                buf1 = io.BytesIO()
                image.save(buf1, format="PNG")
                b64 = base64.b64encode(buf1.getvalue()).decode("utf-8")
                img_url = "data:image/png;base64," + b64

                client = replicate.Client(api_token=REPLICATE_API_TOKEN)

                pred = client.models.predictions.create(
                    model="black-forest-labs/flux-kontext-pro",
                    input={
                        "prompt": PROMPT,
                        "input_image": img_url,
                        "output_format": "png",
                        "safety_tolerance": 5
                    }
                )

                max_wait = 180
                interval = 5
                elapsed = 0

                while elapsed < max_wait:
                    pred.reload()
                    if pred.status == "succeeded":
                        break
                    elif pred.status in ["failed", "canceled"]:
                        st.error("Generation failed: " + str(pred.error))
                        st.stop()
                    time.sleep(interval)
                    elapsed += interval

                if pred.status != "succeeded":
                    st.error("Timed out. Please try again.")
                    st.stop()

                output = pred.output
                if output:
                    out_url = output[0] if isinstance(output, list) else output
                    out_img = Image.open(io.BytesIO(requests.get(out_url).content))

                    st.success("Your leather figurine is ready!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Original", use_column_width=True)
                    with col2:
                        st.image(out_img, caption="Leather Figurine", use_column_width=True)

                    buf2 = io.BytesIO()
                    out_img.save(buf2, format="PNG")
                    st.download_button(
                        label="Download Figurine Image",
                        data=buf2.getvalue(),
                        file_name="leather_pet_figurine.png",
                        mime="image/png"
                    )

            except Exception as e:
                st.error("Something went wrong: " + str(e))

st.markdown("---")
st.markdown("Made with love for pet lovers | Powered by Replicate AI")
