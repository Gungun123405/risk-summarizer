# prompts.py
# Centralized prompt templates for the Client Risk Summarizer.
# Each category gets a framing suited to what "risk" actually means for that
# kind of content, instead of forcing the same "financial risk" lens onto
# every article regardless of topic.

BUSINESS_PROMPT = """Summarize this business article in 2 sentences for a consultant assessing financial and operational risk.
Focus on: financial figures, regulatory exposure, debt/liquidity concerns, and any signals that could affect profitability or investor confidence.

Article:
{article_text}"""

POLITICS_PROMPT = """Summarize this political article in 2 sentences for a consultant assessing political and regulatory risk.
Focus on: policy changes, regulatory or legal exposure for businesses, political stability signals, and any implications for organizations operating in this environment.

Article:
{article_text}"""

SPORT_PROMPT = """Summarize this sports article in 2 sentences for a consultant assessing brand and sponsorship exposure.
Focus on: outcomes that affect sponsorship value, athlete/team reputational standing, and audience or commercial implications, if any are genuinely present. If there is no meaningful commercial angle, say so plainly rather than inventing one.

Article:
{article_text}"""

TECH_PROMPT = """Summarize this technology article in 2 sentences for a consultant assessing technology and market risk.
Focus on: product/market shifts, competitive implications, data/security concerns, and regulatory exposure relevant to tech companies.

Article:
{article_text}"""

ENTERTAINMENT_PROMPT = """Summarize this entertainment article in 2 sentences for a consultant assessing brand and reputational exposure.
Focus on: reputational implications for named individuals or companies, and any commercial impact, if genuinely present. If there is no meaningful risk or commercial angle, say so plainly rather than inventing one.

Article:
{article_text}"""

UNCERTAIN_PROMPT = """Summarize this article in 2 neutral sentences, focusing only on what actually happened.
Then add one short sentence stating whether this article has any clear financial, political, or reputational risk relevance -- and if it's unclear or not applicable, say so directly instead of speculating.

Article:
{article_text}"""

CATEGORY_PROMPTS = {
    "business": BUSINESS_PROMPT,
    "politics": POLITICS_PROMPT,
    "sport": SPORT_PROMPT,
    "tech": TECH_PROMPT,
    "entertainment": ENTERTAINMENT_PROMPT,
    "uncertain / out-of-domain": UNCERTAIN_PROMPT,
}


def build_prompt(category, article_text):
    """
    Returns the fully-formatted prompt string for a given classifier
    category and article text. Falls back to UNCERTAIN_PROMPT for any
    category not explicitly mapped.
    """
    key = (category or "").strip().lower()
    template = CATEGORY_PROMPTS.get(key, UNCERTAIN_PROMPT)
    return template.format(article_text=article_text)