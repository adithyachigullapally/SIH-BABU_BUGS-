"""The five tools the agent brain may call (build report Section 7).

OpenAI function-calling shape, because every configured provider speaks it.
Keep the descriptions short and sharply distinguishable — the free tiers'
real ceiling is tokens per minute, and tool choice is where small backup
models (ministral-8b) get sloppy.
"""

TOOL_SCHEMAS = [
    {
        "name": "run_vqa",
        "description": "Answer a DESCRIPTIVE question about one remote-sensing image - what is present, what type, what condition. Returns prose, never a verified number: it cannot count and cannot measure, and any figure it states is invented rather than computed. For 'how many X' use run_grounding, which counts real detections. For area, coverage or percentage use run_land_cover. Use this tool only for questions that contain no quantity.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string", "description": "'image1' or 'image2'"},
                "question": {"type": "string"},
            },
            "required": ["image_id", "question"],
        },
    },
    {
        "name": "run_caption",
        "description": "Describe a whole scene freely. Use ONLY when the user asks for a description or caption with no specific question - 'describe this', 'caption this image', 'what does this show'. If the user asks anything answerable with a fact, use run_vqa instead.",
        "parameters": {
            "type": "object",
            "properties": {"image_id": {"type": "string", "description": "'image1' or 'image2'"}},
            "required": ["image_id"],
        },
    },
    {
        "name": "run_land_cover",
        "description": "MEASURE how much of one image is vegetation, water, or built-up/bare ground, as a percentage and (when the ground resolution is known) in hectares. Use whenever the user asks how much, how many hectares, what area, what percentage, or how large something is. run_vqa only describes; this one returns numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string", "description": "'image1' or 'image2'"},
            },
            "required": ["image_id"],
        },
    },
    {
        "name": "run_grounding",
        "description": "Locate every object matching a description and return one bounding box per match plus their count. Use when the user says locate, find, highlight or show where - and also for 'how many X are there', because the count comes from actual detections rather than from the model's prose. Detection saturates at 50 objects, so a count of 50 means 'at least 50'.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string", "description": "'image1' or 'image2'"},
                "referring_expression": {"type": "string", "description": "a SINGULAR object noun - 'house', not 'houses'"},
            },
            "required": ["image_id", "referring_expression"],
        },
    },
    {
        "name": "run_change_analysis",
        "description": "Compare two images of the same place at different times and report what changed. Use only when pair_type is bi_temporal.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_id_t1": {"type": "string"},
                "image_id_t2": {"type": "string"},
                "question": {"type": "string"},
            },
            "required": ["image_id_t1", "image_id_t2", "question"],
        },
    },
    {
        "name": "run_fusion_analysis",
        "description": "Jointly analyze a co-registered optical and SAR pair. Use only when pair_type is cross_modal.",
        "parameters": {
            "type": "object",
            "properties": {
                "optical_id": {"type": "string"},
                "sar_id": {"type": "string"},
                "question": {"type": "string"},
            },
            "required": ["optical_id", "sar_id", "question"],
        },
    },
]

# What the OpenAI-compatible endpoints actually want on the wire.
OPENAI_TOOLS = [{"type": "function", "function": schema} for schema in TOOL_SCHEMAS]

TOOL_NAMES = [schema["name"] for schema in TOOL_SCHEMAS]
