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
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800;900&family=DM+Sans:wght@400;500;600;700&display=swap');
:root { --lime:#d7ff3f; --line:rgba(215,255,63,.25); }
.stApp { color:#f4f7f2; font-family:'DM Sans',sans-serif; background:radial-gradient(ellipse at 88% 0%,rgba(115,255,168,.13),transparent 28%),radial-gradient(ellipse at 4% 20%,rgba(215,255,63,.08),transparent 26%),linear-gradient(145deg,#050807 0%,#08110d 48%,#030504 100%); }
.stApp:before { content:""; position:fixed; inset:0; pointer-events:none; opacity:.12; z-index:0; background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px); background-size:42px 42px; }
.block-container { max-width:1500px; padding-top:1.3rem; padding-bottom:3rem; }
h1,h2,h3,h4 { font-family:'Barlow Condensed',sans-serif!important; font-weight:900!important; letter-spacing:-.035em; color:#f7faf5!important; }
h2 { font-size:clamp(2rem,4vw,3.2rem)!important; } h3 { font-size:1.7rem!important; }
p,label,[data-testid="stCaptionContainer"] { color:#d2ded5; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0b1510,#050807)!important; border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:#f4f7f2; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2 { color:var(--lime)!important; }
.stTextInput input,.stTextArea textarea,[data-baseweb="select"]>div,[data-testid="stNumberInput"] input { background:#0b1510!important; color:#f7faf5!important; border:1px solid #31483a!important; border-radius:12px!important; min-height:46px; }
.stTextInput input:focus,.stTextArea textarea:focus,[data-baseweb="select"]>div:focus-within,[data-testid="stNumberInput"] input:focus { border-color:var(--lime)!important; box-shadow:0 0 0 3px rgba(215,255,63,.12)!important; }
.stButton>button,[data-testid="stLinkButton"] a { background:var(--lime)!important; color:#071008!important; border:1px solid var(--lime)!important; border-radius:999px!important; font-weight:900!important; letter-spacing:.07em; text-transform:uppercase; min-height:48px; transition:transform .2s ease,box-shadow .2s ease!important; }
.stButton>button:hover,[data-testid="stLinkButton"] a:hover { transform:translateY(-3px); box-shadow:0 10px 34px rgba(215,255,63,.22)!important; }
[data-testid="stImage"] img { border-radius:14px; border:1px solid rgba(215,255,63,.15); }
[data-testid="stVerticalBlockBorderWrapper"] { background:linear-gradient(145deg,rgba(18,31,24,.97),rgba(7,13,10,.98))!important; border:1px solid #2b4434!important; border-radius:20px!important; padding:14px; box-shadow:0 14px 38px rgba(0,0,0,.22); transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease; }
[data-testid="stVerticalBlockBorderWrapper"]:hover { transform:translateY(-4px); border-color:rgba(215,255,63,.58)!important; box-shadow:0 18px 45px rgba(0,0,0,.36),0 0 24px rgba(215,255,63,.06); }
[data-testid="stMetric"] { background:linear-gradient(135deg,#111e16,#08100b); border:1px solid var(--line); padding:16px; border-radius:16px; }
[data-testid="stMetricValue"] { color:var(--lime)!important; font-family:'Barlow Condensed',sans-serif; }
[data-testid="stProgressBar"]>div>div { background:var(--lime)!important; }
[data-testid="stExpander"] { background:rgba(11,21,16,.78); border:1px solid #2b4434; border-radius:16px; }
hr { border-color:rgba(215,255,63,.2)!important; margin:2rem 0!important; }
.stMarkdown a { color:var(--lime); }
.hero-banner { position:relative; overflow:hidden; isolation:isolate; background:linear-gradient(90deg,rgba(3,12,8,.97) 0%,rgba(3,12,8,.82) 35%,rgba(3,12,8,.12) 100%),url("https://cdn.dribbble.com/userupload/47787263/file/b0a8b0925f7d0b5b7162137d55cda5d9.jpg?resize=1024x768&vertical=center"); background-size:cover; background-position:center; min-height:490px; padding:clamp(28px,5vw,70px); border-radius:28px; border:1px solid rgba(215,255,63,.32); box-shadow:0 24px 90px rgba(0,0,0,.42),inset 0 0 70px rgba(0,0,0,.28); animation:heroEnter .7s ease-out both; }
.hero-banner:before { content:"SV / 01"; position:absolute; top:25px; right:30px; color:var(--lime); font:700 12px 'DM Sans',sans-serif; letter-spacing:.24em; border:1px solid rgba(215,255,63,.5); padding:9px 13px; border-radius:99px; background:rgba(3,12,8,.48); }
.hero-banner:after { content:""; position:absolute; width:340px; height:340px; right:-100px; bottom:-180px; border:1px solid rgba(215,255,63,.42); border-radius:50%; box-shadow:0 0 0 28px rgba(215,255,63,.035),0 0 0 58px rgba(215,255,63,.025); z-index:-1; }
.hero-banner h1 { margin-top:28px; line-height:.82; font-size:clamp(58px,9vw,112px); letter-spacing:-.055em; max-width:720px; text-shadow:0 5px 35px rgba(0,0,0,.55); }
.hero-banner .accent { color:var(--lime); text-shadow:0 0 34px rgba(215,255,63,.2); }
.hero-banner p { font-size:clamp(14px,1.6vw,18px); max-width:480px; line-height:1.7; color:#e0e9e1; }
.hero-button { display:inline-block; margin-top:16px; background:var(--lime); color:#071008; padding:14px 23px; border-radius:999px; font-size:12px; font-weight:900; letter-spacing:.12em; box-shadow:0 0 35px rgba(215,255,63,.2); }
.features { display:flex; flex-wrap:wrap; gap:10px; margin-top:32px; }
.features span { border:1px solid rgba(255,255,255,.18); background:rgba(5,12,8,.58); color:#f4f7f2; border-radius:99px; padding:9px 13px; font-size:12px; backdrop-filter:blur(10px); }
@keyframes heroEnter { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
@media(max-width:640px) { .hero-banner { min-height:430px; padding:24px; border-radius:20px; } .hero-banner:before { top:15px; right:15px; } .hero-banner h1 { margin-top:55px; } }
</style>
""", unsafe_allow_html=True)

# ---------------- HERO BANNER ----------------

st.markdown("""
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







