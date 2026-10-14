from collections import Counter
import streamlit as st
from transformers.pipelines import pipeline

MODEL = "dslim/bert-base-NER"
st.title("Sarib Javed | B040041SJ | Named Entity Recognition")
st.caption(f"Model: {MODEL}. Counts represent mentions, not unique people or places.")
st.info("Use public or fictional text. Entity predictions can be incorrect.")

@st.cache_resource
def load_model():
    return pipeline("token-classification", model=MODEL,
                    aggregation_strategy="simple", device=-1)

text = st.text_area("Enter English text", "John lives in London.", max_chars=10000)
if st.button("Detect entities"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            with st.spinner("Loading model and detecting entities..."):
                ner = load_model()
                tokens = ner.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
                if len(tokens) > ner.model.config.max_position_embeddings:
                    st.warning("Text is too long. Shorten it to fit the model's 512-token limit.")
                    st.stop()
                entities = ner(text)
            st.success(f"Model loaded: {MODEL}")
            if not entities:
                st.info("No named entities detected.")
            else:
                rows = [{"Entity": text[e["start"]:e["end"]],
                         "Type": e["entity_group"],
                         "Confidence": round(float(e["score"]), 4)} for e in entities]
                st.dataframe(rows, hide_index=True)
            counts = Counter(e["entity_group"] for e in entities)
            st.subheader("Entity counts by type")
            st.dataframe([{"Type": label, "Count": counts[label]}
                          for label in ("PER", "ORG", "LOC", "MISC")], hide_index=True)
        except Exception as exc:
            st.error(f"Unable to complete inference: {type(exc).__name__}. Check installation and model download access.")
