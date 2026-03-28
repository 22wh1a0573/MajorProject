def generate_intent(product_name, category, description):
    prompt = f"""
Write ONE short sentence describing ONLY the functional use.

Rules:
- Start with body region (upper-body / lower-body / full-body)
- Mention activity or use-case
- No brand names
- No marketing language
- Under 12 words

Product name: {product_name}
Category: {category}
Description: {description}

Functional intent:
"""

    try:
        result = subprocess.run(
            [
                r"C:\Users\prath\AppData\Local\Programs\Ollama\ollama.exe",
                "run",
                "phi3"
            ],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=30
        )

        output = result.stdout.strip().split("\n")[0]
        return output[:120]

    except Exception as e:
        return ""
