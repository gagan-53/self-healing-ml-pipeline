"""SH-MLP Prompt Library — Master Catalogue"""
PROMPT_CATALOGUE = []

def register(prompt):
    PROMPT_CATALOGUE.append(prompt)
    return prompt
