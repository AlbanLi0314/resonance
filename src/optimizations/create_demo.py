"""
Create Streamlit Web Demo
=========================
Generates a Streamlit app for interactive researcher search.
"""

from pathlib import Path


def create_streamlit_demo(output_file: Path = None) -> dict:
    """
    Create a Streamlit web demo for the academic matcher.

    Args:
        output_file: Path to save the demo app

    Returns:
        Result dict
    """

    demo_code = '''#!/usr/bin/env python3
"""
Academic Matcher - Web Demo
===========================
Interactive web interface for searching Cornell researchers.

Run with:
    streamlit run demo_app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.matcher import AcademicMatcher
from src.config import GEMINI_API_KEY

# Page config
st.set_page_config(
    page_title="Cornell Academic Matcher",
    page_icon="🎓",
    layout="wide"
)

# Title
st.title("🎓 Cornell Academic Matcher")
st.markdown("Find Cornell researchers whose work matches your interests")

# Sidebar
with st.sidebar:
    st.header("Settings")
    use_rerank = st.checkbox("Use LLM Reranking", value=bool(GEMINI_API_KEY))
    if use_rerank and not GEMINI_API_KEY:
        st.warning("GEMINI_API_KEY not set. Reranking disabled.")
        use_rerank = False

    num_results = st.slider("Number of results", 3, 10, 5)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool uses:
    - **SPECTER2** embeddings for semantic search
    - **FAISS** for fast vector similarity
    - **Gemini** for intelligent reranking (optional)
    """)

# Initialize matcher (cached)
@st.cache_resource
def load_matcher(skip_rerank: bool):
    matcher = AcademicMatcher()
    try:
        matcher.initialize(skip_rerank=skip_rerank)
        return matcher
    except FileNotFoundError as e:
        st.error(f"Error: {e}")
        st.info("Please run `python scripts/step1_build_index.py` first to build the index.")
        return None

# Load matcher
with st.spinner("Loading matcher..."):
    matcher = load_matcher(skip_rerank=not use_rerank)

if matcher is None:
    st.stop()

# Search interface
st.markdown("---")
query = st.text_area(
    "Describe your research interests or the type of researcher you're looking for:",
    placeholder="e.g., DNA origami self-assembly, machine learning for materials discovery, polymer synthesis...",
    height=100
)

col1, col2 = st.columns([1, 4])
with col1:
    search_clicked = st.button("🔍 Search", type="primary")

# Run search
if search_clicked and query:
    with st.spinner("Searching..."):
        results = matcher.search(query, final_k=num_results, skip_rerank=not use_rerank)

    if results:
        st.markdown(f"### Found {len(results)} researchers")

        for r in results:
            with st.container():
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**#{r['rank']} {r['name']}**")
                    st.markdown(f"*{r['position']}* | {r['department']}")
                    if r.get('lab'):
                        st.markdown(f"🔬 {r['lab']}")

                with col2:
                    if r.get('email'):
                        st.markdown(f"📧 [{r['email']}](mailto:{r['email']})")
                    if r.get('personal_website'):
                        st.markdown(f"🌐 [Website]({r['personal_website']})")
                    if r.get('embedding_score'):
                        st.metric("Score", f"{r['embedding_score']:.3f}")

                st.markdown(f"💡 **Why:** {r['reason']}")
                st.markdown("---")
    else:
        st.warning("No results found. Try a different query.")

elif search_clicked:
    st.warning("Please enter a search query.")

# Example queries
st.markdown("### Try these example queries:")
examples = [
    "DNA origami self-assembly and nanomaterials",
    "Machine learning for battery materials discovery",
    "Polymer synthesis and controlled polymerization",
    "CO2 reduction and electrocatalysis",
    "2D materials graphene synthesis",
    "Bioprinting and tissue engineering",
]

cols = st.columns(3)
for i, example in enumerate(examples):
    with cols[i % 3]:
        if st.button(example, key=f"example_{i}"):
            st.session_state['query'] = example
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "*Built with SPECTER2, FAISS, and Gemini | "
    "[View Source](https://github.com/your-repo)*"
)
'''

    output_file = output_file or Path(__file__).parent.parent.parent / "demo_app.py"
    output_file = Path(output_file)

    with open(output_file, 'w') as f:
        f.write(demo_code)

    print(f"Streamlit demo created: {output_file}")
    print(f"Run with: streamlit run {output_file}")

    return {
        "output_file": str(output_file),
        "command": f"streamlit run {output_file}"
    }


if __name__ == "__main__":
    result = create_streamlit_demo()
    print(f"Result: {result}")
