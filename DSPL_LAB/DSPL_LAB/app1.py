import streamlit as st
import joblib
import re
from scipy.sparse import hstack, csr_matrix


# ==========================================
# 1. Load model and vectorizer
# ==========================================

model = joblib.load(
    "models/logistic_regression_comparison.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer_text_rating.pkl"
)


# ==========================================
# 2. Text cleaning
# ==========================================

def clean_text(text):
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove newline characters
    text = re.sub(r"\n+", " ", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# 3. Page configuration
# ==========================================

st.set_page_config(
    page_title="Fake Product Review Detector",
    page_icon="🔍",
    layout="centered"
)


# ==========================================
# 4. Custom CSS
# ==========================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #666666;
        margin-bottom: 30px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f5f7fa;
        border: 1px solid #e2e6ea;
        margin-bottom: 20px;
    }

    .result-title {
        text-align: center;
        font-size: 30px;
        font-weight: 700;
    }

    .confidence {
        text-align: center;
        font-size: 20px;
        margin-bottom: 10px;
    }

    .feature-box {
        padding: 10px;
        border-radius: 8px;
        background-color: #f7f7f7;
        margin-bottom: 6px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# 5. Header
# ==========================================

st.markdown(
    '<div class="main-title">🔍 Fake Product Review Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'NLP and Machine Learning Based Review Analysis'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# 6. Project information
# ==========================================

with st.expander("ℹ️ About this project"):

    st.write(
        "This system analyzes product review text and star rating "
        "using a machine-learning model to classify a review as "
        "likely computer-generated or organic."
    )

    st.write(
        "**Dataset:** 40,412 reviews after duplicate removal"
    )

    st.write(
        "**Final model:** Logistic Regression"
    )

    st.write(
        "**Features:** TF-IDF text features + star rating"
    )

    st.write(
        "**Test accuracy:** 93.44%"
    )


# ==========================================
# 7. Review input
# ==========================================

st.subheader("📝 Enter Product Review")

review = st.text_area(
    "Write or paste the review below:",
    placeholder=(
        "Example: The product is very comfortable and "
        "works exactly as expected..."
    ),
    height=200
)


# ==========================================
# 8. Rating input
# ==========================================

st.subheader("⭐ Select Star Rating")

rating = st.radio(
    "Choose the rating given to the product:",
    [1, 2, 3, 4, 5],
    format_func=lambda x: f"{'⭐' * x}  ({x})",
    horizontal=True
)


st.write("")


# ==========================================
# 9. Analyze button
# ==========================================

analyze = st.button(
    "🔎 Analyze Review",
    use_container_width=True
)


# ==========================================
# 10. Prediction
# ==========================================

if analyze:

    if review.strip() == "":
        st.warning("Please enter a product review before analyzing.")

    else:

        # --------------------------------------
        # Clean review
        # --------------------------------------

        cleaned_review = clean_text(review)

        # --------------------------------------
        # Convert text to TF-IDF
        # --------------------------------------

        review_tfidf = vectorizer.transform(
            [cleaned_review]
        )

        # --------------------------------------
        # Convert rating to sparse matrix
        # --------------------------------------

        rating_feature = csr_matrix(
            [[rating]]
        )

        # --------------------------------------
        # Combine text + rating
        # --------------------------------------

        final_features = hstack([
            review_tfidf,
            rating_feature
        ])

        # --------------------------------------
        # Prediction
        # --------------------------------------

        prediction = model.predict(
            final_features
        )[0]

        probabilities = model.predict_proba(
            final_features
        )[0]

        genuine_probability = probabilities[0]
        fake_probability = probabilities[1]


        # ======================================
        # Display result
        # ======================================

        st.divider()

        st.subheader("📊 Result")

        if prediction == 1:

            result = "FAKE / SUSPICIOUS"
            confidence = fake_probability * 100

            st.error(
                f"### ⚠️ {result}"
            )

        else:

            result = "GENUINE"
            confidence = genuine_probability * 100

            st.success(
                f"### ✅ {result}"
            )


        st.markdown(
            f'<div class="confidence">'
            f'Confidence: <b>{confidence:.2f}%</b>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.progress(
            confidence / 100
        )


        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Rating",
                f"{rating} / 5"
            )

        with col2:
            st.metric(
                "Fake Probability",
                f"{fake_probability * 100:.2f}%"
            )


        # ======================================
        # Explanation
        # ======================================

        st.divider()

        st.subheader("💡 Why this prediction?")

        if prediction == 1:

            st.write(
                "The model classified this review as "
                "**FAKE / SUSPICIOUS** because its learned "
                "text patterns and rating contributed more "
                "strongly toward the computer-generated class."
            )

        else:

            st.write(
                "The model classified this review as "
                "**GENUINE** because its learned text patterns "
                "and rating contributed more strongly toward "
                "the organic class."
            )


        # ======================================
        # Model feature explanation
        # ======================================

        coefficients = model.coef_[0]

        feature_names = vectorizer.get_feature_names_out()

        tfidf_values = review_tfidf.toarray()[0]

        text_coefficients = coefficients[
            :len(feature_names)
        ]

        text_contributions = (
            tfidf_values * text_coefficients
        )


        # --------------------------------------
        # Ignore common words
        # --------------------------------------

        stop_words = {
            "the", "and", "for", "with", "this", "that",
            "from", "have", "has", "had", "was", "were",
            "are", "you", "your", "they", "their", "its",
            "but", "not", "very", "can", "just", "into",
            "what", "all", "one", "our", "out", "too",
            "a", "an", "of", "to", "in", "is", "it"
        }


        feature_contributions = []

        for i, contribution in enumerate(
            text_contributions
        ):

            if contribution == 0:
                continue

            feature = feature_names[i]

            words = feature.split()

            # Ignore features made only of common words
            if all(
                word in stop_words or len(word) <= 2
                for word in words
            ):
                continue

            feature_contributions.append(
                (feature, contribution)
            )


        # ======================================
        # Show relevant features
        # ======================================

        if prediction == 1:

            important_features = [
                item
                for item in feature_contributions
                if item[1] > 0
            ]

            important_features.sort(
                key=lambda x: x[1],
                reverse=True
            )

        else:

            important_features = [
                item
                for item in feature_contributions
                if item[1] < 0
            ]

            important_features.sort(
                key=lambda x: x[1]
            )


        important_features = important_features[:5]


        if important_features:

            st.write("**Important model features:**")

            for feature, contribution in important_features:

                st.markdown(
                    f'<div class="feature-box">🔹 '
                    f'<b>{feature}</b>'
                    f'</div>',
                    unsafe_allow_html=True
                )


        # ======================================
        # Rating contribution
        # ======================================

        rating_coefficient = coefficients[-1]

        rating_contribution = (
            rating * rating_coefficient
        )

        if rating_contribution > 0:
            rating_direction = "toward FAKE"
        else:
            rating_direction = "toward GENUINE"


        st.write(
            f"**Rating contribution:** "
            f"Rating {rating} contributed {rating_direction}."
        )


        # ======================================
        # Probabilities
        # ======================================

        with st.expander("View prediction probabilities"):

            st.write(
                f"**Genuine (OR):** "
                f"{genuine_probability * 100:.2f}%"
            )

            st.write(
                f"**Fake (CG):** "
                f"{fake_probability * 100:.2f}%"
            )


        # ======================================
        # Important note
        # ======================================

        st.caption(
            "Important: The model identifies patterns learned "
            "from the training dataset. A prediction does not "
            "prove that a review is actually fake or genuine."
        )


# ==========================================
# 11. Footer
# ==========================================

st.divider()

st.caption(
    "Fake Product Review Detector | "
    "NLP + Machine Learning | DSPL Project"
)