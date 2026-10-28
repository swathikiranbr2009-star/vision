import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import moondream as md
from PIL import Image
import requests

# ---------------- PAGE SETUP ----------------

st.set_page_config(
    page_title="Style Vision AI",
    page_icon="👗",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background: radial-gradient(ellipse at 75% 5%, #254a36 0%, #0b271f 42%, #041713 100%);
    color: #f5f7f3;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4 {
    color: #f5f7f3 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 800 !important;
}

[data-testid="stSidebar"] {
    background: #0b211b;
    border-right: 1px solid #355748;
}

[data-testid="stSidebar"] * {
    color: #f5f7f3;
}

.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] {
    background: #102c24 !important;
    color: white !important;
    border: 1px solid #426354 !important;
    border-radius: 10px !important;
}

.stButton > button {
    background: #c9ff65;
    color: #102018 !important;
    border: 1px solid #c9ff65;
    border-radius: 30px;
    font-weight: 800;
    padding: 0.65rem 1.3rem;
}

.stButton > button:hover {
    background: #e0ffa3;
    border-color: #e0ffa3;
}

[data-testid="stImage"] img {
    border-radius: 14px;
    object-fit: cover;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(20, 51, 41, 0.85);
    border: 1px solid #355748;
    border-radius: 14px;
    padding: 12px;
}

[data-testid="stMetric"] {
    background: #102c24;
    border: 1px solid #355748;
    padding: 14px;
    border-radius: 12px;
}

[data-testid="stMetricValue"] {
    color: #c9ff65;
}

p, label, [data-testid="stCaptionContainer"] {
    color: #e0e9e2;
}

hr {
    border-color: #355748;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO BANNER ----------------

st.markdown("""
<style>
.hero-banner {
    background:
        linear-gradient(90deg, rgba(5,22,15,0.96), rgba(5,22,15,0.2)),
        url("https://cdn.dribbble.com/userupload/47787263/file/b0a8b0925f7d0b5b7162137d55cda5d9.jpg?resize=1024x768&vertical=center");
    background-size: cover;
    background-position: center;
    min-height: 440px;
    padding: 45px;
    border-radius: 22px;
}

.hero-banner h1 {
    color: white;
    font-size: clamp(44px, 7vw, 78px);
}

.hero-banner .accent {
    color: #c9ff65;
}
</style>

<div class="hero-banner">
    <h1>FIND YOUR<br><span class="accent">OWN STYLE.</span></h1>
    <p>Discover your vibe. Explore new looks. Find outfits that express who you are.</p>
    <div class="hero-button">FIND YOUR LOOK &nbsp; →</div>
    <div class="features">
        <span>✦ Personalised styles</span>
        <span>✦ Budget-friendly picks</span>
        <span>✦ Outfit inspiration</span>
    </div>
</div>
""", unsafe_allow_html=True)




# ---------------- LOAD PRODUCTS ----------------


# ---------------- REAL SHOPPING PRODUCTS ----------------

st.header("Find Your Style")
st.write("Discover real men's clothing with AI-powered similarity matching.")

preference = st.text_input(
    "Describe what you're looking for",
    placeholder="e.g. black oversized shirt for college"
)

col1, col2 = st.columns(2)

with col1:
    category = st.selectbox(
        "Category",
        ["Any", "Shirts", "T-shirts", "Jeans", "Trousers", "Jackets", "Hoodies"]
    )

    color = st.selectbox(
        "Preferred colour",
        ["Any", "Black", "White", "Blue", "Grey", "Green", "Brown", "Beige"]
    )

with col2:
    style = st.selectbox(
        "Fashion style",
        ["Any", "Casual", "Streetwear", "Minimalist", "Formal", "Sporty"]
    )

    occasion = st.selectbox(
        "Occasion",
        ["Any", "College", "Daily wear", "Party", "Formal", "Festival"]
    )

budget = st.slider(
    "Maximum budget (₹)",
    min_value=500,
    max_value=10000,
    value=2000,
    step=100
)

if st.button("Find My Style", type="primary"):
    api_key = st.secrets.get("SERPAPI_KEY", "")

    if not api_key:
        st.error("SerpApi key missing. Add SERPAPI_KEY in Streamlit Secrets.")

    elif not preference.strip() and all(
        x == "Any" for x in [category, color, style, occasion]
    ):
        st.warning("Enter a clothing preference or select a filter.")

    else:
        query_parts = [
            "men's clothing",
            preference.strip(),
            "" if category == "Any" else category,
            "" if color == "Any" else color,
            "" if style == "Any" else style,
            "" if occasion == "Any" else occasion
        ]

        shopping_query = " ".join(x for x in query_parts if x).strip()

        try:
            with st.spinner("Finding real products..."):
                response = requests.get(
                    "https://serpapi.com/search.json",
                    params={
                        "engine": "google_shopping",
                        "q": shopping_query,
                        "api_key": api_key,
                        "gl": "in",
                        "hl": "en",
                        "google_domain": "google.co.in",
                        "max_price": budget
                    },
                    timeout=30
                )

                response.raise_for_status()
                data = response.json()

            if data.get("error"):
                st.error(f"Shopping search error: {data['error']}")

            else:
                products = data.get("shopping_results", [])

                # Keep men's clothing results within the selected budget.
                valid_products = []

                for product in products:
                    title = product.get("title", "")
                    price = product.get("extracted_price")
                    image_url = product.get("thumbnail", "")
                    product_url = (
                        product.get("product_link")
                        or product.get("link", "")
                    )

                    if not title or price is None:
                        continue

                    try:
                        price = float(price)
                    except (ValueError, TypeError):
                        continue

                    if price > budget:
                        continue

                    valid_products.append({
                        "name": title,
                        "price": price,
                        "image": image_url,
                        "link": product_url,
                        "source": product.get("source", ""),
                        "description": product.get("snippet", "")
                    })

                if not valid_products:
                    st.warning(
                        "No suitable priced products found. "
                        "Try a different search or increase your budget."
                    )

                else:
                    # Rank the returned products by text similarity.
                    product_texts = [
                        p["name"] + " " + p["description"]
                        for p in valid_products
                    ]

                    search_text = " ".join([
                        preference,
                        category if category != "Any" else "",
                        color if color != "Any" else "",
                        style if style != "Any" else "",
                        occasion if occasion != "Any" else ""
                    ]).strip()

                    if search_text:
                        vectorizer = TfidfVectorizer(stop_words="english")
                        vectors = vectorizer.fit_transform(
                            [search_text] + product_texts
                        )

                        scores = cosine_similarity(
                            vectors[0:1], vectors[1:]
                        ).flatten()
                    else:
                        scores = [0.0] * len(valid_products)

                    ranked = sorted(
                        zip(valid_products, scores),
                        key=lambda item: item[1],
                        reverse=True
                    )

                    st.success(f"Found {len(ranked)} matching products!")
                    st.subheader("Recommended for You")

                    columns = st.columns(3)

                    for index, (product, score) in enumerate(ranked):
                        with columns[index % 3]:
                            with st.container(border=True):
                                if product["image"].startswith("https://") or (
                                    product["image"].startswith("http://")
                                ):
                                    st.image(
                                        product["image"],
                                        use_container_width=True
                                    )

                                st.markdown(f"### {product['name']}")
                                st.write(f"**Price:** ₹{product['price']:.0f}")

                                if product["source"]:
                                    st.caption(f"Store: {product['source']}")

                                st.progress(
                                    float(score),
                                    text=f"Text similarity: {score * 100:.0f}%"
                                )

                                if product["link"]:
                                    st.link_button(
                                        "View Product",
                                        product["link"],
                                        use_container_width=True
                                    )

        except requests.RequestException as error:
            st.error("Could not fetch shopping results. Check your connection.")
            st.caption(str(error))

# ---------------- END SHOPPING PRODUCTS ----------------


# ---------------- PROJECT INFORMATION ----------------

with st.expander("About this project"):
    st.write(
        "Style Vision AI is a fashion recommendation prototype. "
        "It uses TF-IDF and cosine similarity to compare a user's "
        "text preferences with product descriptions. Filters help "
        "limit results by budget, colour, style, category and occasion."
    )
    st.write(
        "This is a content-based recommendation demo, not a trained "
        "commercial AI system. Product results depend on the availability "
        "of Google Shopping results through SerpApi."
    )

st.caption("Style Vision AI | AI Project Prototype")


# ---------------- AI PHOTO STYLING ----------------

st.markdown("---")
st.header("📸 AI Photo Stylist")
st.write("Upload an outfit photo for personalised styling suggestions.")

uploaded_photo = st.file_uploader(
    "Upload your outfit photo",
    type=["jpg", "jpeg", "png", "webp"],
    key="styling_photo"
)

styling_occasion = st.selectbox(
    "Choose an occasion",
    ["Casual", "College", "Party", "Date or dinner",
     "Formal", "Festival"],
    key="styling_occasion"
)

styling_budget = st.number_input(
    "Budget for additional clothes (₹)",
    min_value=0,
    value=1500,
    step=100,
    key="styling_budget"
)

if uploaded_photo:
    image = Image.open(uploaded_photo).convert("RGB")
    st.image(image, caption="Your outfit", use_container_width=True)

    if st.button("✨ Suggest My Style"):
        api_key = st.secrets.get("MOONDREAM_API_KEY", "")

        if not api_key:
            st.error("Moondream API key is not configured.")
        else:
            try:
                with st.spinner("Analysing your outfit..."):
                    model = md.vl(api_key=api_key)

                    result = model.query(
                        image,
                        f"Analyze this clothing photo and suggest three "
                        f"ways to style the outfit for {styling_occasion}. "
                        f"My budget for additional clothes is "
                        f"₹{styling_budget}. Suggest color combinations, "
                        f"matching men's clothing, and practical tips. "
                        f"Only describe details visible in the image."
                    )

                st.subheader("Your Styling Suggestions")
                st.markdown(result.get("answer", "No suggestions returned."))

            except Exception as e:
                st.error("Photo analysis failed. Check your API configuration.")
                st.caption(str(e))
