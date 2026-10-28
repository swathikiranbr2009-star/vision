
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
