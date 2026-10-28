
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        linear-gradient(90deg,
            rgba(5, 22, 15, 0.96) 0%,
            rgba(5, 22, 15, 0.78) 45%,
            rgba(5, 22, 15, 0.20) 100%),
        url("https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?auto=format&fit=crop&w=1800&q=85");

    background-size: cover;
    background-position: center 35%;
    background-repeat: no-repeat;

    min-height: 440px;
    padding: 45px;
    border-radius: 22px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    overflow: hidden;
    box-sizing: border-box;
}

.hero-banner h1 {
    color: #ffffff !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: clamp(44px, 7vw, 78px);
    font-weight: 800;
    line-height: 0.98;
    margin: 0;
    max-width: 590px;
}

.hero-banner .accent {
    color: #c9ff65 !important;
}

.hero-banner p {
    color: #e2e9e4 !important;
    font-size: 15px;
    line-height: 1.8;
    max-width: 360px;
    margin: 20px 0;
}

.hero-banner .hero-button {
    display: inline-block;
    width: fit-content;
    color: #102018 !important;
    background: #c9ff65;
    border-radius: 30px;
    padding: 13px 24px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
}

.hero-banner .features {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    margin-top: 28px;
    color: #e2e9e4;
    font-size: 11px;
}

@media (max-width: 600px) {
    .hero-banner {
        padding: 28px 22px;
        min-height: 400px;
        background-position: center;
    }
}
</style>

<div class="hero-banner">
    <h1>
        FIND YOUR<br>
        <span class="accent">OWN STYLE.</span>
    </h1>

    <p>
        Discover your vibe. Explore new looks.
        Find outfits that express who you are.
    </p>

    <div class="hero-button">
        FIND YOUR LOOK &nbsp; →
    </div>

    <div class="features">
        <span>✦ Personalised styles</span>
        <span>✦ Budget-friendly picks</span>
        <span>✦ Outfit inspiration</span>
    </div>
</div>
""", unsafe_allow_html=True)




# ---------------- LOAD PRODUCTS ----------------

def load_products():
    data = pd.read_csv("clothes.csv")

    # Convert prices into numbers
    data["price"] = pd.to_numeric(
        data["price"]
        .astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce"
    )

    # Remove products with invalid prices
    data = data.dropna(subset=["price"]).copy()
    data["price"] = data["price"].astype(float)

    return data




try:
    df = load_products()
except Exception as error:
    st.error(f"Could not load clothes.csv: {error}")
    st.info(
        "Check that clothes.csv is in the same folder as app.py "
        "and contains the required column names."
    )
    st.stop()

if df.empty:
    st.warning("No products with valid prices were found in clothes.csv.")
    st.stop()

# ---------------- PREPARE AI MODEL ----------------

df["search_text"] = (
    df["name"] + " " +
    df["category"] + " " +
    df["color"] + " " +
    df["style"] + " " +
    df["occasion"] + " " +
    df["description"]
)

try:
    vectorizer = TfidfVectorizer(stop_words="english")
    product_vectors = vectorizer.fit_transform(df["search_text"])
except ValueError:
    st.error("Your product descriptions are empty. Add product text to clothes.csv.")
    st.stop()

# ---------------- SIDEBAR FILTERS ----------------

st.sidebar.header("Your Preferences")

max_price = int(df["price"].max())
min_price = int(df["price"].min())

budget = st.sidebar.slider(
    "Maximum budget (₹)",
    min_value=0,
    max_value=max(1000, max_price),
    value=max(1000, min_price),
    step=100
)

def filter_options(column):
    values = sorted(
        value for value in df[column].unique()
        if value.strip()
    )
    return ["Any"] + values

category = st.sidebar.selectbox(
    "Category", filter_options("category")
)

color = st.sidebar.selectbox(
    "Preferred colour", filter_options("color")
)

style = st.sidebar.selectbox(
    "Fashion style", filter_options("style")
)

occasion = st.sidebar.selectbox(
    "Occasion", filter_options("occasion")
)

# ---------------- FASHION SEARCH ----------------

st.header("Find Your Style")

preference = st.text_input(
    "Describe what you're looking for",
    placeholder="e.g. black casual clothes for college"
)

if st.button("Find My Style", type="primary"):
    st.session_state["search_clicked"] = True

if "search_clicked" not in st.session_state:
    st.session_state["search_clicked"] = False

# ---------------- FILTER PRODUCTS ----------------

filtered = df[df["price"] <= budget].copy()

if category != "Any":
    filtered = filtered[filtered["category"] == category]

if color != "Any":
    filtered = filtered[filtered["color"] == color]

if style != "Any":
    filtered = filtered[filtered["style"] == style]

if occasion != "Any":
    filtered = filtered[filtered["occasion"] == occasion]

# ---------------- AI RECOMMENDATIONS ----------------

if st.session_state["search_clicked"]:
    query_parts = [
        preference,
        "" if category == "Any" else category,
        "" if color == "Any" else color,
        "" if style == "Any" else style,
        "" if occasion == "Any" else occasion
    ]

    query = " ".join(query_parts).strip()

    if filtered.empty:
        st.warning(
            "No products match these filters. "
            "Try increasing your budget or changing your preferences."
        )
    else:
        if query:
            query_vector = vectorizer.transform([query])
            scores = cosine_similarity(
                query_vector, product_vectors
            ).flatten()

            filtered["AI Match Score"] = scores[filtered.index]
            filtered = filtered.sort_values(
                "AI Match Score", ascending=False
            )
        else:
            filtered["AI Match Score"] = 0.0
            filtered = filtered.sort_values("price")

        st.success(
            f"Found {len(filtered)} products within your preferences!"
        )

        st.subheader("Recommended for You")
        columns = st.columns(3)

        for position, (_, product) in enumerate(filtered.iterrows()):
            with columns[position % 3]:
                st.markdown("---")

                if product["image"].startswith(("https://", "http://")):
                    st.image(
                        product["image"],
                        use_container_width=True
                    )

                st.markdown(f"### {product['name']}")
                st.write(f"**Price:** ₹{product['price']:.0f}")
                st.write(f"**Category:** {product['category']}")
                st.write(f"**Colour:** {product['color']}")
                st.write(f"**Style:** {product['style']}")
                st.write(f"**Occasion:** {product['occasion']}")
                st.caption(product["description"])

                score = float(product["AI Match Score"]) * 100
                st.progress(
                    min(1.0, max(0.0, score / 100)),
                    text=f"Text similarity: {score:.0f}%"
                )

                feedback = st.selectbox(
                    "Your feedback",
                    ["Choose feedback", "👍 Like", "👎 Dislike"],
                    key=f"feedback_{product.name}"
                )

                if feedback in ["👍 Like", "👎 Dislike"]:
                    st.caption(
                        "Feedback selected for this demo. "
                        "It does not retrain the model."
                    )

        # ---------------- OUTFIT SUGGESTIONS ----------------

        st.subheader("Outfit Suggestions")

        tops = filtered[
            filtered["category"].str.lower().isin(
                ["tops", "top", "shirt", "t-shirt"]
            )
        ]

        bottoms = filtered[
            filtered["category"].str.lower().isin(
                ["bottoms", "bottom", "pants", "jeans"]
            )
        ]

        shoes = filtered[
            filtered["category"].str.lower().isin(
                ["shoes", "footwear"]
            )
        ]

        if not tops.empty and not bottoms.empty:
            top_product = tops.iloc[0]
            bottom_product = bottoms.iloc[0]

            outfit_total = (
                float(top_product["price"]) +
                float(bottom_product["price"])
            )

            st.write(
                f"**Suggested combination:** "
                f"{top_product['name']} + {bottom_product['name']}"
            )
            st.write(f"Combined price: ₹{outfit_total:.0f}")

            if not shoes.empty:
                shoe_product = shoes.iloc[0]
                full_total = outfit_total + float(shoe_product["price"])

                st.write(
                    f"Add {shoe_product['name']} "
                    f"for a total of ₹{full_total:.0f}."
                )
        else:
            st.info(
                "Add products from different categories to your "
                "catalogue to generate more outfit combinations."
            )
else:
    st.info(
        "Choose your filters, describe your style, "
        "and click Find My Style to see recommendations."
    )

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
        "commercial AI system. Recommendations depend on the products "
        "available in clothes.csv."
    )

st.caption("Style Vision AI | AI Project Prototype")


