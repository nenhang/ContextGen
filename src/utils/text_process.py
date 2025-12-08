def get_kontext_edit_template(instruction: str) -> str:
    prefixes = ["a photo of ", "an image of ", "a picture of "]
    for prefix in prefixes:
        if instruction.lower().startswith(prefix):
            instruction = instruction[len(prefix) :].strip()
            break

    return (
        f"Modify the image to depict '{instruction}'. "
        "While preserving the original layout, key object features and human identities "
        "(including facial details), adjust element poses for better composition "
        "and naturally fill in the background. "
        "Ensure the result appears natural and visually harmonious."
    )
