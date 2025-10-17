"""
Transcription Error Correction Module

This module uses an LLM to identify and fix common transcription errors
in Portuguese business/technical meetings before fact extraction.

Common issues:
- Phonetic errors: "sapo" → "SAP", "arrepê" → "ERP"
- Missing spaces: "ex-cel" → "Excel"
- Domain term confusion: "esquela" → "schema"
- Name variations: "pórtaga" → "Portuga"
"""

from __future__ import annotations
import os
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI


# Common domain-specific mappings for Brazilian Portuguese business context
DOMAIN_TERM_MAP = {
    # ERP/Software Systems
    "sapo": "SAP",
    "sábio": "SAP",
    "arrepê": "ERP",
    "erp": "ERP",
    "ex-cel": "Excel",
    "excel": "Excel",
    "esquela": "schema",
    "esquelas": "schemas",
    "esquelão": "schema",
    "midler": "middleware",
    "midlayer": "middleware",
    
    # Business/Compliance Terms
    "compliesce": "compliance",
    "compliesse": "compliance",
    "complice": "compliance",
    "colímpio": "compliance",
    
    # Technical Terms
    "darmazinha": "armazém",
    "darmanzém": "armazém",
    "api": "API",
    "apis": "APIs",
    "pi": "PI",
    "rpa": "RPA",
    "ia": "IA",
    "ai": "AI",
    "blrm": "PLM",
    "plnn": "PLM",
    "pni": "PNI",
    "ngpa": "GPA",
    "capice": "CapEx",
    
    # Common Mishearings
    "pabolo": "Pablo",
    "arturco": "Arturo",
    "pórtaga": "Portuga",
    "caioinha": "Caio",
    "albalbalbal": "seu bolso",
    "pelinha": "telinha",
    "lavatório": "relatório",
    "exameira": "à toa",
    "afinaramento": "refinamento",
    "incendiferra": "me ferrar",
    "vogado": "advogado",
    "mumpadora": "montadora",
    "seperem": "se pedem",
    "joga a onda": "jogar",
    
    # Time/Dates
    "teo": "TO",
}



def build_correction_prompt(text: str, domain_hints: Optional[List[str]] = None) -> str:
    """
    Build a prompt for the LLM to correct transcription errors.
    
    Args:
        text: Raw transcript text with potential errors
        domain_hints: Optional list of domain-specific terms to watch for
    
    Returns:
        Formatted prompt for error correction
    """
    domain_context = ""
    if domain_hints:
        domain_context = f"\n\nDomain context: This is a meeting about {', '.join(domain_hints)}."
    
    prompt = f"""You are an expert at correcting Portuguese transcription errors from business meetings.

Your task is to fix transcription mistakes AND remove hallucinations using ONLY these strategies:

CORRECTION STRATEGY - Apply in order:
1. Remove obvious hallucinations (music/noise artifacts, excessive repetition)
2. Check if word exists in Portuguese dictionary → If YES, keep it unchanged
3. Check if it's a common phonetic error → Fix only if you're 100% certain
4. Check if it's a split/merged word → Fix only if obvious
5. If NONE of the above → LEAVE UNCHANGED (don't guess)

HALLUCINATION REMOVAL (remove these patterns):
- Music/noise markers: "♪", "♫", "🎵", "[Music]", sequences of "..."
- Excessive repetition: Same word/phrase repeated 5+ times consecutively
  Example: "e aí e aí e aí e aí e aí" → Remove entirely (not real speech)
- Meaningless loops: Very long segments where the same 2-3 words keep repeating
  Example: "era tão raro era tão raro era tão raro..." → Remove entirely
- Background noise transcribed as words: Unintelligible sequences that don't form coherent speech
- IMPORTANT: Only remove OBVIOUS hallucinations. Natural repetition in speech is OK!
  Example: "Tá, tá, entendi" → KEEP (natural affirmation)
  Example: "Não, não, não quero" → KEEP (natural emphasis)
  Example: "e aí e aí e aí e aí e aí e aí" (10+ times) → REMOVE (hallucination)

PHONETIC ERROR PATTERNS (fix only these clear patterns):
- Phonetic spelling of acronyms: "sapo"/"sapô" → "SAP", "arrepê"/"erre-pê" → "ERP"
- Phonetic spelling of software: "ex-cel"/"éxcel" → "Excel"
- Common misspellings with clear alternatives: "planilas" → "planilhas", "devogado" → "advogado"

WORD SPLITTING/MERGING (fix only obvious cases):
- Merged words that should be split: "doarmazém" → "do armazém"
- Split words that should be merged: "dar mazinha" → "armazém" (ONLY if context makes it clear)

WHAT TO KEEP UNCHANGED:
- Informal speech: "né", "cara", "tá", "pra", "pro" (these are valid colloquial Portuguese)
- Natural repetition (2-3 times for emphasis)
- Names and proper nouns (even if they sound unusual)
- Words you're not 100% certain about
- Technical jargon that might be correct
- Any word that could be a valid Portuguese word
{domain_context}

CRITICAL RULES:
- CONSERVATIVE APPROACH: When in doubt, keep original
- Only remove/fix OBVIOUS hallucinations and errors
- Don't "improve" the text, just fix clear transcription errors and remove noise
- Preserve the speaker's voice and informal style
- If a segment is entirely hallucination (no real speech), output an empty line for that segment
- Return ONLY the corrected text, no explanations

VALIDATION BEFORE OUTPUT:
For each word you change or remove, ask yourself:
1. Is this obviously not real speech (music, excessive repetition)? → Remove
2. Does the original word NOT exist in Portuguese? (If it exists, don't change)
3. Am I 100% certain about the correction? (If not, don't change)
4. Is this a clear phonetic/technical term error? (If not clear, don't change)
- NEVER create hybrid/fake words by combining parts of different words
- When in doubt, CHECK: Does this word actually exist? If not sure, keep original
- Examples of FORBIDDEN outputs: "complice" (not a word), "afinaramento" (not a word), "planilas" (should be "planilhas")

Original transcript:
{text}

Corrected transcript (with hallucinations removed):"""
    
    return prompt


def apply_quick_fixes(text: str, custom_map: Optional[Dict[str, str]] = None) -> str:
    """
    Apply quick regex-based fixes for common known errors.
    This is faster than LLM but less intelligent.
    
    Args:
        text: Input text
        custom_map: Optional custom term mapping to use
    
    Returns:
        Text with quick fixes applied
    """
    term_map = DOMAIN_TERM_MAP.copy()
    if custom_map:
        term_map.update(custom_map)
    
    result = text
    
    # Apply word boundary replacements (case-insensitive)
    for wrong, correct in term_map.items():
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(wrong) + r'\b'
        result = re.sub(pattern, correct, result, flags=re.IGNORECASE)
    
    # Fix common spacing issues
    result = re.sub(r'\bex-cel\b', 'Excel', result, flags=re.IGNORECASE)
    result = re.sub(r'\be x c e l\b', 'Excel', result, flags=re.IGNORECASE)
    
    return result


def detect_suspicious_words(text: str, original_text: str) -> List[str]:
    """
    Detect potentially hallucinated words by comparing corrected text with original.
    Returns list of suspicious words that might be LLM hallucinations.
    
    Strategy: Flag words that:
    1. Are new (not in original)
    2. Are very similar to original words but not exact matches
    3. Match common hallucination patterns
    
    This is a safety check, not a validator.
    """
    suspicious = []
    
    # Get words from both texts
    original_words = set(re.findall(r'\b\w+\b', original_text.lower()))
    corrected_words = re.findall(r'\b\w+\b', text.lower())
    
    # Common hallucination patterns (words that don't exist in Portuguese)
    hallucination_patterns = [
        r'^complice$',  # not a word (cúmplice is)
        r'^planilas$',  # not a word (planilhas is)
        r'^afinaramento$',  # not a word
        r'.*ramento$',  # suspicious -ramento endings (often hallucinations)
        r'^.*lice$',  # suspicious -lice endings
    ]
    
    for word in corrected_words:
        # Skip very short words and numbers
        if len(word) <= 2 or word.isdigit():
            continue
            
        # Check if word matches hallucination patterns
        for pattern in hallucination_patterns:
            if re.match(pattern, word, re.IGNORECASE):
                suspicious.append(word)
                break
    
    return suspicious


def correct_transcription_llm(
    text: str,
    *,
    api_key: str,
    base_url: Optional[str] = None,
    model: str = "gpt-5-mini",
    domain_hints: Optional[List[str]] = None,
    num_passes: int = 2,
    validate: bool = True,
) -> str:
    """
    Use LLM to intelligently correct transcription errors with multiple passes.
    
    Args:
        text: Raw transcript text
        api_key: OpenAI API key
        base_url: Optional OpenAI-compatible base URL
        model: Model to use (default: gpt-5-mini for better quality than nano)
        domain_hints: Optional domain context hints
        num_passes: Number of correction passes (default: 2 for thorough correction)
        validate: Whether to check for suspicious hallucinations (default: True)
    
    Returns:
        Corrected transcript text
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    current_text = text
    
    # Run multiple correction passes to catch more errors
    for pass_num in range(num_passes):
        prompt = build_correction_prompt(current_text, domain_hints)
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at correcting Portuguese transcription errors AND removing hallucinations from business meetings. You MUST: 1) Remove obvious hallucinations (music markers, excessive repetition of 5+ times, noise artifacts), 2) Only output real Portuguese words or known technical terms, 3) Never invent nonsense words, 4) If unsure about a correction, leave the word unchanged. Return only the corrected text with hallucinations removed, no explanations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Very low temperature to minimize hallucinations and creative word invention
                max_completion_tokens=len(current_text.split()) * 2,
            )
            
            corrected = response.choices[0].message.content.strip()
            
            # Validate for suspicious hallucinations if requested
            if validate:
                suspicious = detect_suspicious_words(corrected, text)
                if suspicious:
                    print(f"[TranscriptionCorrection] WARNING: Detected suspicious words (possible hallucinations): {suspicious}")
                    print(f"[TranscriptionCorrection] If these persist, consider reducing temperature or number of passes")
            
            # If text didn't change, no need for more passes
            if corrected == current_text:
                print(f"[TranscriptionCorrection] Converged after {pass_num + 1} pass(es)")
                break
            
            current_text = corrected
            print(f"[TranscriptionCorrection] Completed pass {pass_num + 1}/{num_passes}")
            
        except Exception as e:
            print(f"[TranscriptionCorrection] LLM correction failed on pass {pass_num + 1}: {e}")
            # If first pass fails, fallback to quick fixes
            if pass_num == 0:
                return apply_quick_fixes(text)
            # If later pass fails, return what we have so far
            break
    
    return current_text


def correct_transcription(
    text: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini",
    mode: str = "llm",  # Changed default from "hybrid" to "llm" - pure LLM approach
    domain_hints: Optional[List[str]] = None,
    custom_term_map: Optional[Dict[str, str]] = None,
    num_passes: int = 2,
) -> str:
    """
    Correct transcription errors using specified mode.
    
    Args:
        text: Raw transcript text
        api_key: OpenAI API key (required for LLM modes)
        base_url: Optional OpenAI-compatible base URL
        model: Model to use for LLM correction (default: gpt-5-mini)
        mode: Correction mode - "quick", "llm" (default), or "hybrid"
        domain_hints: Optional domain context hints
        custom_term_map: Optional custom term replacements (for hybrid/quick modes)
        num_passes: Number of LLM correction passes (default: 2)
    
    Returns:
        Corrected transcript text
    
    Modes:
        - llm: Pure LLM-based correction (RECOMMENDED - smart, context-aware)
        - hybrid: Quick regex fixes first, then LLM (legacy, uses term map)
        - quick: Fast regex-based replacements only (not recommended)
    
    Note: 
        The term map is now optional. Pure LLM mode with GPT-5 Mini + 2 passes
        is smart enough to handle most corrections contextually without hardcoded rules.
    """
    if not text or not text.strip():
        return text
    
    # Quick mode: regex only
    if mode == "quick":
        return apply_quick_fixes(text, custom_term_map)
    
    # LLM mode: intelligent correction
    if mode == "llm":
        if not api_key:
            print("[TranscriptionCorrection] No API key provided, falling back to quick mode")
            return apply_quick_fixes(text, custom_term_map)
        return correct_transcription_llm(
            text,
            api_key=api_key,
            base_url=base_url,
            model=model,
            domain_hints=domain_hints,
            num_passes=num_passes
        )
    
    # Hybrid mode (default): quick fixes first, then LLM for context
    if mode == "hybrid":
        # Apply quick fixes first
        quick_fixed = apply_quick_fixes(text, custom_term_map)
        
        # If no API key, return quick fixes only
        if not api_key:
            print("[TranscriptionCorrection] No API key provided, using quick fixes only")
            return quick_fixed
        
        # Then apply LLM for more nuanced corrections
        return correct_transcription_llm(
            quick_fixed,
            api_key=api_key,
            base_url=base_url,
            model=model,
            domain_hints=domain_hints,
            num_passes=num_passes
        )
    
    # Unknown mode, default to quick
    print(f"[TranscriptionCorrection] Unknown mode '{mode}', using quick mode")
    return apply_quick_fixes(text, custom_term_map)


def correct_transcript_from_env(
    text: str,
    *,
    mode: Optional[str] = None,
    domain_hints: Optional[List[str]] = None,
    custom_term_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Convenience function that reads configuration from environment variables.
    
    Environment variables:
        - OPENAI_API_KEY: OpenAI API key
        - OPENAI_BASE_URL: Optional base URL
        - OPENAI_MODEL: Optional model override (default: gpt-5-mini)
        - SPINE_CORRECTION_MODE: "llm" (default, pure AI), "hybrid" (term map + AI), or "quick" (term map only)
        - SPINE_CORRECTION_ENABLED: Set to "false" to disable (default: true)
    
    Args:
        text: Raw transcript text
        mode: Optional mode override
        domain_hints: Optional domain context hints
        custom_term_map: Optional custom term replacements
    
    Returns:
        Corrected transcript text
        
    Note:
        Default mode is now "llm" (pure LLM) instead of "hybrid" (term map + LLM).
        This leverages GPT-5 Mini's intelligence for context-aware corrections without
        maintaining a static term map. Use "hybrid" if you want term map as a first pass.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    correction_mode = mode or os.getenv("SPINE_CORRECTION_MODE", "llm")  # Changed default to "llm"
    
    return correct_transcription(
        text,
        api_key=api_key,
        base_url=base_url,
        model=model,
        mode=correction_mode,
        domain_hints=domain_hints,
        custom_term_map=custom_term_map,
    )


# CLI for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python transcription_correction.py <input_file> [mode]")
        print("Modes: quick, llm, hybrid (default)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "hybrid"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print("=== Original Text ===")
    print(text[:500] + "..." if len(text) > 500 else text)
    print()
    
    corrected = correct_transcript_from_env(text, mode=mode)
    
    print("=== Corrected Text ===")
    print(corrected[:500] + "..." if len(corrected) > 500 else corrected)
    print()
    
    # Write to output file
    output_file = input_file.replace('.txt', '_corrected.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(corrected)
    
    print(f"✅ Corrected text written to: {output_file}")
