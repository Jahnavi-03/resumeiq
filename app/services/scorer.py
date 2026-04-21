import json

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.models.schemas import CandidateAnalysis, CandidateRanking

# ---------------------------------------------------------------------------
# Why temperature=0.1?
#
# Temperature controls how "creative" the LLM is.
#   temperature=0.0  → completely deterministic, same output every time
#   temperature=1.0  → highly creative, unpredictable outputs
#   temperature=0.1  → nearly deterministic but with tiny flexibility
#
# We need structured JSON back from the LLM. If temperature is high,
# the LLM might format the response differently each time, add random
# commentary, or change field names. At 0.1 it stays focused and consistent.
#
# Why not 0.0? A tiny bit of flexibility helps the LLM reason about
# edge cases in resumes without getting stuck in rigid patterns.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Why "no markdown, no code blocks" in every prompt?
#
# LLMs are trained to be helpful and readable. By default they wrap JSON
# in markdown code blocks like:
#
#   ```json
#   { "ats_score": 75 }
#   ```
#
# json.loads() cannot parse that — it only handles raw JSON strings.
# We explicitly tell the LLM not to do this so we get clean parseable JSON.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# What does RAG context add?
#
# RAG = Retrieval Augmented Generation. Instead of relying only on what
# the LLM learned during training, we retrieve relevant chunks from our
# knowledge base (ATS rules, FAANG guide, scoring rubric etc.) and inject
# them into the prompt as expert context.
#
# Without RAG: LLM uses general knowledge → generic advice
# With RAG:    LLM uses our curated rules → specific, accurate advice
#              e.g. "According to ATS rules: two-column layouts fail parsing"
# ---------------------------------------------------------------------------


class ScoringService:
    """
    Calls the Groq LLM to analyze resumes and return structured Pydantic objects.

    Two modes:
      score_candidate() → analyzes one resume against one job description
      score_recruiter() → ranks one candidate relative to a job description
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

        # Initialize the Groq LLM client once.
        # model_name and api_key come from settings → read from .env
        self._llm = ChatGroq(
            model=self._settings.chat_model,
            api_key=self._settings.groq_api_key,
            temperature=0.1,   # see note at top of file
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """
        Send a prompt to the Groq LLM and return the raw text response.

        HumanMessage wraps the prompt string in the format LangChain expects.
        .invoke() sends it to Groq and returns an AIMessage object.
        .content extracts the text string from that object.
        """
        response = self._llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def _parse_json(self, raw: str) -> dict:
        """
        Parse the LLM's raw text response into a Python dictionary.

        Why strip()? The LLM sometimes adds a leading newline or trailing space.
        Why check for ```? Despite our instructions, some LLMs still wrap JSON
        in markdown code blocks. We strip those out as a safety net.
        """
        text = raw.strip()

        # Safety net — remove markdown code block wrappers if LLM added them
        if text.startswith("```"):
            # Remove opening ```json or ``` line
            text = text.split("\n", 1)[-1]
            # Remove closing ``` line
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]

        return json.loads(text.strip())

    # -----------------------------------------------------------------------
    # Candidate mode
    # -----------------------------------------------------------------------

    def score_candidate(
        self,
        resume_text: str,
        job_description: str,
        context: str,
    ) -> CandidateAnalysis:
        """
        Analyze a single resume against a job description.

        Args:
            resume_text:     Full text extracted from the candidate's PDF.
            job_description: The job posting the candidate is applying for.
            context:         RAG chunks joined into one string — expert knowledge
                             retrieved from our knowledge base relevant to this resume.

        Returns:
            A validated CandidateAnalysis Pydantic object.

        Raises:
            ValueError: If the LLM returns invalid JSON after one retry,
                        or if the JSON doesn't match the expected schema.
        """
        prompt = f"""You are an expert ATS resume analyzer with deep knowledge of
hiring practices at top tech companies.

=== EXPERT KNOWLEDGE (use this to inform your analysis) ===
{context}

=== RESUME ===
{resume_text}

=== JOB DESCRIPTION ===
{job_description}

Analyze the resume against the job description using the expert knowledge above.
Return ONLY valid JSON, no markdown, no code blocks, no explanation.

The JSON must match this exact structure:
{{
  "ats_score": <integer 0-100>,
  "keyword_match": <integer 0-100>,
  "format_score": <integer 0-10>,
  "missing_required_skills": ["skill1", "skill2"],
  "missing_preferred_skills": ["skill1", "skill2"],
  "formatting_issues": ["issue1", "issue2"],
  "bullet_analyses": [
    {{
      "original_bullet": "Led a team",
      "original_score": <integer 1-10>,
      "rewritten_bullet": "Led cross-functional team of 8 engineers...",
      "rewritten_score": <integer 1-10>,
      "issue": "No metric, weak verb",
      "improvement_notes": "Added team size and quantified outcome",
      "priority": "High"
    }}
  ],
  "suggestions": ["suggestion1", "suggestion2"],
  "projected_score": <integer 0-100>
}}

Rules:
- priority must be exactly "High", "Medium", or "Low"
- rewritten_score must be >= original_score
- projected_score must be > ats_score
- Analyze at least 3 bullet points from the resume
- Return ONLY the JSON object, nothing else"""

        # First attempt
        raw = self._call_llm(prompt)

        try:
            data = self._parse_json(raw)
        except json.JSONDecodeError:
            # If JSON is invalid, retry once with a stricter instruction.
            # One retry is enough — if it fails twice the prompt has a deeper problem.
            retry_prompt = (
                f"{prompt}\n\nYour previous response was not valid JSON. "
                "Return ONLY a raw JSON object. No text before or after it."
            )
            raw = self._call_llm(retry_prompt)
            try:
                data = self._parse_json(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"LLM returned invalid JSON after retry: {exc}\n"
                    f"Raw response was:\n{raw}"
                ) from exc

        # Validate the parsed dict against our Pydantic schema.
        # If any field is missing, the wrong type, or out of range,
        # Pydantic raises ValidationError with a clear description.
        try:
            return CandidateAnalysis(**data)
        except ValidationError as exc:
            raise ValueError(
                f"LLM response did not match CandidateAnalysis schema:\n{exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Recruiter mode
    # -----------------------------------------------------------------------

    def score_recruiter(
        self,
        resume_text: str,
        job_description: str,
        context: str,
        candidate_name: str,
    ) -> CandidateRanking:
        """
        Rank a single candidate against a job description.

        Args:
            resume_text:     Full text extracted from the candidate's PDF.
            job_description: The job posting being hired for.
            context:         RAG chunks — expert knowledge for this analysis.
            candidate_name:  Candidate's name extracted from the resume.

        Returns:
            A validated CandidateRanking Pydantic object.

        Raises:
            ValueError: If the LLM returns invalid JSON after one retry,
                        or if the JSON doesn't match the expected schema.
        """
        prompt = f"""You are an expert technical recruiter evaluating candidates
for a role at a top tech company.

=== EXPERT KNOWLEDGE (use this to inform your evaluation) ===
{context}

=== CANDIDATE NAME ===
{candidate_name}

=== RESUME ===
{resume_text}

=== JOB DESCRIPTION ===
{job_description}

Evaluate how well this candidate matches the job description.
Return ONLY valid JSON, no markdown, no code blocks, no explanation.

The JSON must match this exact structure:
{{
  "name": "{candidate_name}",
  "match_score": <integer 0-100>,
  "strengths": ["strength1", "strength2", "strength3"],
  "skill_gaps": ["gap1", "gap2"],
  "recommendation": "Strong Yes"
}}

Rules:
- recommendation must be exactly one of: "Strong Yes", "Yes", "Maybe", "No"
- match_score 80-100 → "Strong Yes", 60-79 → "Yes", 40-59 → "Maybe", 0-39 → "No"
- strengths must have at least 2 items
- Return ONLY the JSON object, nothing else"""

        # First attempt
        raw = self._call_llm(prompt)

        try:
            data = self._parse_json(raw)
        except json.JSONDecodeError:
            # Retry once with a stricter instruction
            retry_prompt = (
                f"{prompt}\n\nYour previous response was not valid JSON. "
                "Return ONLY a raw JSON object. No text before or after it."
            )
            raw = self._call_llm(retry_prompt)
            try:
                data = self._parse_json(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"LLM returned invalid JSON after retry: {exc}\n"
                    f"Raw response was:\n{raw}"
                ) from exc

        try:
            return CandidateRanking(**data)
        except ValidationError as exc:
            raise ValueError(
                f"LLM response did not match CandidateRanking schema:\n{exc}"
            ) from exc
