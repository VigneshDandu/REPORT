import streamlit as st
import joblib
import re


# -----------------------------------
# 1. Load trained model and vectorizer
# -----------------------------------
model = joblib.load("models/logistic_regression.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# -----------------------------------
# 2. Text cleaning function
# -----------------------------------
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


# -----------------------------------
# 3. Page configuration
# -----------------------------------
st.set_page_config(
    page_title="Fake Product Review Detector",
    page_icon="🔍",
    layout="centered"
)


# -----------------------------------
# 4. Title
# -----------------------------------
st.title("🔍 Fake Product Review Detector")

st.write(
    "Enter a product review and select its star rating "
    "to analyze whether the review is likely to be "
    "computer-generated or organic."
)


# -----------------------------------
# 5. Review input
# -----------------------------------
review = st.text_area(
    "Enter Product Review",
    placeholder="Example: Amazing product! Very good quality. I highly recommend it.",
    height=150
)


# -----------------------------------
# 6. Rating input
# -----------------------------------
rating = st.selectbox(
    "Select Star Rating",
    [1, 2, 3, 4, 5],
    index=4
)


# -----------------------------------
# 7. Analyze button
# -----------------------------------
if st.button("🔎 Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a product review.")

    else:
        # Clean review
        cleaned_review = clean_text(review)

        # Convert review into TF-IDF features
        review_vector = vectorizer.transform([cleaned_review])

        # Prediction
        prediction = model.predict(review_vector)[0]

        # Probability
        probabilities = model.predict_proba(review_vector)[0]

        genuine_probability = probabilities[0]
        fake_probability = probabilities[1]

        # -----------------------------------
        # 8. Display result
        # -----------------------------------
        if prediction == 1:
            result = "FAKE / SUSPICIOUS"
            confidence = fake_probability * 100
        else:
            result = "GENUINE"
            confidence = genuine_probability * 100

        st.subheader("Result")

        if prediction == 1:
            st.error(f"Prediction: {result}")
        else:
            st.success(f"Prediction: {result}")

        st.write(f"Confidence: **{confidence:.2f}%**")

        st.write(f"Selected Rating: **{'⭐' * rating}**")


        # -----------------------------------
        # 9. Explanation
        # -----------------------------------
        st.subheader("Explanation")

        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]

        feature_values = review_vector.toarray()[0]

        contributions = feature_values * coefficients

        # Get features actually present in the review
        non_zero_indices = review_vector.nonzero()[1]

        feature_contributions = []

        for index in non_zero_indices:
            feature_contributions.append(
                (
                    feature_names[index],
                    contributions[index]
                )
            )

        # Sort by absolute influence
        feature_contributions.sort(
            key=lambda x: abs(x[1]),
            reverse=True
        )

        top_features = feature_contributions[:5]

        if top_features:

            if prediction == 1:
                st.write(
                    "The model classified this review as suspicious "
                    "because some of its words/patterns are associated "
                    "more strongly with computer-generated reviews "
                    "in the training data."
                )
            else:
                st.write(
                    "The model classified this review as genuine "
                    "because some of its words/patterns are associated "
                    "more strongly with organic reviews in the "
                    "training data."
                )

            st.write("Important words/patterns detected:")

            for word, contribution in top_features:
                st.write(f"- **{word}**")

        else:
            st.write(
                "The model could not identify strong text patterns "
                "for an explanation."
            )


# -----------------------------------
# 10. Important note
# -----------------------------------
st.divider()

st.caption(
    "Note: The current ML model makes its prediction using the "
    "review text. The selected star rating is displayed but is "
    "not yet used by the trained model."
)