"""
Gemini API Client for Academic Matcher
Handles all LLM interactions for match explanation and conversation.
"""
import os
from typing import List, Dict, Optional
from pathlib import Path

# Load .env file
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

import google.generativeai as genai


class GeminiClient:
    """Client for Gemini API interactions"""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3-flash-preview"):
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
            model_name: Gemini model to use (default: gemini-3-flash-preview)
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set. Please set it in .env file or pass directly.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name

    def explain_match(
        self,
        query: str,
        researcher: Dict,
        user_question: Optional[str] = None
    ) -> str:
        """
        Explain why a researcher matches the search query.

        Args:
            query: Original search query
            researcher: Researcher dict with name, department, interests, etc.
            user_question: Optional follow-up question

        Returns:
            Explanation text
        """
        context = self._build_context(researcher)
        question = user_question or "Why is this researcher a good match for my search query?"

        prompt = f"""<system>
You are an academic research advisor at Cornell University.
Your role is to explain why researchers match search queries and suggest collaboration opportunities.
Be specific, professional, and helpful.
</system>

<context>
Search Query: "{query}"

Researcher Profile:
{context}
</context>

<task>
{question}
</task>

<instructions>
- Explain the specific research overlap between the query and the researcher's work
- Mention actual research areas, techniques, or topics from the profile
- If relevant, suggest potential collaboration angles
- Keep response concise (under 150 words)
- Use bullet points for clarity when listing multiple points
- Be professional but approachable
</instructions>"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def chat(
        self,
        message: str,
        researcher: Dict,
        history: List[Dict],
        original_query: str = ""
    ) -> str:
        """
        Continue conversation about a researcher.

        Args:
            message: User's new message
            researcher: Current researcher context
            history: Previous messages [{"role": "user/assistant", "content": "..."}]
            original_query: Original search query

        Returns:
            Assistant response
        """
        context = self._build_context(researcher)

        # Build conversation history string (keep last 6 messages)
        history_str = ""
        for msg in history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n\n"

        prompt = f"""<system>
You are an academic research advisor at Cornell University.
Help users learn about researchers and discover collaboration opportunities.
Answer questions based only on the provided context.
</system>

<researcher_profile>
{context}
</researcher_profile>

<original_search>
{original_query}
</original_search>

<conversation_history>
{history_str}
</conversation_history>

<current_question>
{message}
</current_question>

<instructions>
- Answer based on the researcher profile provided
- If information is not available in the context, say so honestly
- Be concise (under 100 words unless more detail is needed)
- Maintain a professional, helpful tone
- If asked about collaboration, suggest specific research angles based on the profile
</instructions>"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _build_context(self, researcher: Dict) -> str:
        """Build researcher context string for prompts"""
        parts = []

        # Basic info
        parts.append(f"Name: {researcher.get('name', 'Unknown')}")
        parts.append(f"Department: {researcher.get('department', 'N/A')}")

        if researcher.get('position'):
            parts.append(f"Position: {researcher['position']}")

        # Research interests (critical for matching)
        if researcher.get('research_interests'):
            parts.append(f"Research Interests: {researcher['research_interests']}")

        # Publications
        if researcher.get('papers'):
            papers_list = []
            for p in researcher['papers'][:5]:  # Limit to 5 papers
                title = p.get('title', 'Untitled')
                year = p.get('year', '')
                papers_list.append(f"  - {title}" + (f" ({year})" if year else ""))
            if papers_list:
                parts.append(f"Recent Publications:\n" + "\n".join(papers_list))

        # Additional raw text (truncated)
        if researcher.get('raw_text'):
            raw = researcher['raw_text']
            if len(raw) > 1000:
                raw = raw[:1000] + "..."
            parts.append(f"Additional Information: {raw}")

        return "\n".join(parts)

    def test_connection(self) -> bool:
        """Test if the API connection works"""
        try:
            response = self.model.generate_content("Say 'Connection successful!' in exactly those words.")
            return "successful" in response.text.lower()
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False


# Quick test
if __name__ == "__main__":
    print("Testing Gemini Client...")
    print("=" * 50)

    try:
        client = GeminiClient()
        print(f"Model: {client.model_name}")

        # Test connection
        print("\n1. Testing connection...")
        if client.test_connection():
            print("   ✓ Connection successful!")
        else:
            print("   ✗ Connection failed!")
            exit(1)

        # Test explain_match
        print("\n2. Testing explain_match...")
        test_researcher = {
            "name": "Dr. Jane Smith",
            "department": "Materials Science & Engineering",
            "research_interests": "Quantum materials, superconductivity, thin film growth, MBE synthesis",
            "papers": [
                {"title": "Topological superconductivity in quantum heterostructures", "year": "2024"},
                {"title": "MBE growth of high-Tc superconducting films", "year": "2023"}
            ]
        }

        result = client.explain_match(
            query="quantum computing materials",
            researcher=test_researcher
        )
        print(f"   Query: 'quantum computing materials'")
        print(f"   Response:\n{result[:500]}...")

        # Test chat
        print("\n3. Testing chat...")
        chat_result = client.chat(
            message="What specific techniques does this researcher use?",
            researcher=test_researcher,
            history=[],
            original_query="quantum computing materials"
        )
        print(f"   Question: 'What specific techniques does this researcher use?'")
        print(f"   Response:\n{chat_result[:500]}...")

        print("\n" + "=" * 50)
        print("All tests passed! Gemini client is ready.")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)
