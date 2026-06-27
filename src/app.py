import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pickle
import os
import base64
from pathlib import Path

from face_recognizer import recognize
from gallery_organizer import organize_gallery
from search_person import search_person
from analytics import gallery_statistics


from pathlib import Path

def set_background():
    image_path = Path(__file__).resolve().parent.parent / "assets" / "background.jpg"

    with open(image_path, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
    f"""
    <style>

    /* Background Image */
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* Main content */
    .main .block-container {{
        background: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 15px;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: rgba(20, 20, 20, 0.75);
        backdrop-filter: blur(10px);
    }}

    /* Sidebar text */
    section[data-testid="stSidebar"] * {{
        color: white !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)   

# --------------------------------------------------
# Page Configuration (MUST BE FIRST)
# --------------------------------------------------

st.set_page_config(
    page_title="AI Personal Gallery Organizer",
    page_icon="📸",
    layout="wide"
)

set_background()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("📸 AI Gallery Organizer")

st.sidebar.info("""
### Technologies Used

✅ TensorFlow

✅ FaceNet

✅ OpenCV

✅ MTCNN

✅ Streamlit
""")

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📸 AI Personal Gallery Organizer")

st.write(
    "Automatic Face Recognition and Gallery Organization "
    "using TensorFlow, FaceNet, OpenCV and MTCNN."
)

st.divider()

# --------------------------------------------------
# Dashboard
# --------------------------------------------------

with open("models/face_database.pkl", "rb") as f:
    database = pickle.load(f)

with open("models/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

total_people = len(database)
total_embeddings = len(embeddings)

total_images = 0

for person in os.listdir("dataset/original_images"):
    folder = os.path.join("dataset/original_images", person)

    if os.path.isdir(folder):
        total_images += len(os.listdir(folder))

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("👤 Registered Persons", total_people)

with col2:
    st.metric("🖼 Dataset Images", total_images)

with col3:
    st.metric("🧠 Embeddings", total_embeddings)

st.divider()

# --------------------------------------------------
# Tabs
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📷 Face Recognition",
    "📂 Gallery Organizer",
    "🔍 Search Person",
    "📊 Analytics",
    "📤 Upload Gallery"
])

# ==================================================
# TAB 1
# ==================================================

with tab1:

    st.subheader("Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results = recognize(image)

        if len(results) == 0:
            st.warning("⚠️ No face detected.")
            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )
            st.stop()

        # Draw rectangles

        for person in results:

            x, y, w, h = person["box"]

            name = person["name"]

            score = person["score"]

            color = (0,255,0)

            if name == "Unknown":
                color = (0,0,255)

            cv2.rectangle(
                image,
                (x,y),
                (x+w,y+h),
                color,
                2
            )

            cv2.putText(
                image,
                f"{name} ({score*100:.1f}%)",
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.success("✅ Recognition Completed!")

        st.image(
            image,
            caption="Recognition Result",
            use_container_width=True
        )

        st.subheader("Detected People")

        data = []

        for person in results:

            data.append({
                "Person": person["name"],
                "Confidence (%)": round(person["score"]*100,2)
            })

        st.dataframe(
            pd.DataFrame(data),
            hide_index=True,
            use_container_width=True
        )

# ==================================================
# TAB 2
# ==================================================

with tab2:

    st.subheader("📂 Automatic Gallery Organizer")

    st.write(
        "Place all your images inside the **new_gallery** folder "
        "and click the button below."
    )

    if st.button("🚀 Organize Gallery"):

        with st.spinner("Organizing Gallery..."):

            result = organize_gallery()

        st.success("Gallery Organized Successfully!")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Processed", result["processed"])

        with c2:
            st.metric("Organized", result["organized"])

        with c3:
            st.metric("Recognized", result["recognized"])

        with c4:
            st.metric("No Face", result["no_face"])


with tab3:

    st.subheader("🔍 Search Photos by Person")

    person = st.text_input(
        "Enter Person Name"
    )

    if st.button("Search"):

        images = search_person(person)

        if len(images) == 0:

            st.error("No images found.")

        else:

            st.success(
                f"{len(images)} images found."
            )

            cols = st.columns(3)

            for i, img in enumerate(images):

                cols[i % 3].image(
                    img,
                    use_container_width=True
                )


with tab4:

    st.subheader("📊 Gallery Analytics")

    stats, total = gallery_statistics()

    st.metric(
        "Total Organized Images",
        total
    )

    if len(stats) == 0:

        st.warning("No organized gallery found.")

    else:

        df = pd.DataFrame({

            "Person": stats.keys(),

            "Images": stats.values()

        })

        df = df.sort_values(
            "Images",
            ascending=False
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            df.set_index("Person")
        )


with tab5:

    st.subheader("📤 Upload Gallery")

    uploaded_files = st.file_uploader(
        "Select Multiple Images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    if uploaded_files:

        os.makedirs("new_gallery", exist_ok=True)

        if st.button("Save Images"):

            count = 0

            progress = st.progress(0)

            for i, file in enumerate(uploaded_files):

                save_path = os.path.join(
                    "new_gallery",
                    file.name
                )

                with open(save_path, "wb") as f:
                    f.write(file.read())

                count += 1

                progress.progress((i + 1) / len(uploaded_files))

            st.success(f"{count} images uploaded successfully!")
# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Developed by Nandini Guntupalli | "
    "AI Personal Gallery Organizer using TensorFlow, FaceNet and OpenCV"
)