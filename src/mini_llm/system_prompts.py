import random
from datetime import date

from mini_llm.prompt import DEFAULT_SYSTEM_PROMPT, build_system_prompt

MATH_SYSTEM_PROMPT = "Solve the problem step by step. End with the numeric answer."

SYSTEM_PARAPHRASES = [
    "You are Petal, a compact AI assistant. Keep answers short and direct. "
    "Use a short markdown list when it helps. You can do simple arithmetic.",
    "You are Petal, a small AI assistant. Be brief and to the point. "
    "When a list makes things clearer, write a short markdown list. "
    "Basic arithmetic is something you can do.",
    "You are Petal, a small assistant built to answer concisely. "
    "When several items are involved, list them as a short markdown list. "
    "You can handle basic arithmetic.",
]

DATE_QUESTIONS = [
    "What is today's date?",
    "What's the date today?",
    "What day is it today?",
    "What is the current date?",
    "Do you know today's date?",
    "Tell me today's date.",
    "what is the date?",
    "What date is it?",
]
DATE_START = date(2024, 1, 1).toordinal()
DATE_END = date(2027, 12, 31).toordinal()


def sample_date(rng):
    return date.fromordinal(rng.randint(DATE_START, DATE_END)).isoformat()


def date_pairs(count, seed=3):
    rng = random.Random(seed)
    pairs = []
    for _ in range(count):
        day = sample_date(rng)
        system = build_system_prompt(DEFAULT_SYSTEM_PROMPT, date=day)
        pairs.append((system, rng.choice(DATE_QUESTIONS), f"Today is {day}."))
    return pairs


VARIATIONS = [
    (
        "Answer in one sentence.", "Explain what a linked list is used for.",
        "A linked list is a structure that stores elements in sequence with pointers, useful when data changes size often.",
    ),
    (
        "Answer with a numbered list.", "Give five tips to sleep better. Skip the intro.",
        "1. Keep a fixed bedtime.\n2. Avoid screens before bed.\n3. Keep the room cool.\n4. Skip caffeine in the evening.\n5. Get daylight during the day.",
    ),
    (
        "Answer with a short markdown list of four items.", "How do I start learning guitar?",
        "- Practice 20 minutes daily.\n- Learn 4 basic chords first.\n- Play along with songs you like.\n- Pick songs you actually enjoy.",
    ),
    (
        "Reply in one word.", "What is 12 times 3?",
        "36",
    ),
    (
        "Explain it to a 10-year-old.", "What is gravity?",
        "Gravity is like an invisible hand that pulls everything toward the ground. It is why a dropped ball falls down and never up.",
    ),
    (
        "Be formal. No contractions.", "What's the capital of Canada?",
        "The capital of Canada is Ottawa.",
    ),
    (
        "You are a pirate. Stay in character.", "What time is it?",
        "Arr, I be a small land-locked model with no clock, matey. Check any reasonable timepiece!",
    ),
    (
        "Answer with two bullet points only.", "Soup or salad for a cold day?",
        "- Soup: warm and easy to eat.\n- Salad: fresh but cold on a cold day.",
    ),
    (
        "End your answer with a question.", "What is the largest prime under 20?",
        "The largest prime under 20 is 19. Would you like the next one?",
    ),
    (
        "Answer in exactly three words.", "How do I get better at chess?",
        "Study tactics daily.",
    ),
    (
        "Answer with a code block only. No words outside it.", "Show a python one-liner that squares numbers 0 to 9.",
        "```python\n[n * n for n in range(10)]\n```",
    ),
    (
        "Answer with a markdown list of four items.", "How do I keep houseplants alive?",
        "- Water when the top soil is dry.\n- Give them bright indirect light.\n- Repot once a year.\n- Wipe the leaves now and then.",
    ),
    (
        "Give four numbered steps and nothing else.", "How to write email a whole team will read?",
        "1. Open with the purpose in one sentence.\n2. List the actions with owners.\n3. Keep it under ten lines.\n4. Send it before the deadline.",
    ),
    (
        "Answer with a short intro line, then a three-item list.", "What should I pack for a beach day?",
        "Keep it light:\n- Sunscreen.\n- Water and a snack.\n- A hat.",
    ),
    (
        "Answer with a two-item list comparing the options.", "Morning study or evening study?",
        "- Morning: quiet mind, fewer distractions.\n- Evening: more time, tired attention.",
    ),
    (
        "Reply with a single three-item list.", "How to practice piano?",
        "- Scales every day.\n- Hands slowly, separately.\n- Full tempo only when clean.",
    ),
    (
        "Give a numbered list of five quick wins.", "How to keep a desk tidy?",
        "1. Bin everything you do not need.\n2. Keep one inbox tray.\n3. Files go in folders, not piles.\n4. Charge the phone away from the desk.\n5. Reset the desk before leaving.",
    ),
]

UNCERTAINTY_PAIRS = [
    (
        DEFAULT_SYSTEM_PROMPT, "What is the population of Liechtenstein in 2041?",
        "I don't know a reliable answer for a future figure, and I won't guess specific numbers.",
    ),
    (
        DEFAULT_SYSTEM_PROMPT, "Who won the 1984 World Chess Championship?",
        "I don't have a confident answer for that. I'd rather say I don't know than get it wrong.",
    ),
    (
        DEFAULT_SYSTEM_PROMPT, "What is the boiling point of aerogel?",
        "I'm not sure. Aerogel is not a liquid, so that question doesn't have a standard answer I can trust.",
    ),
    (
        DEFAULT_SYSTEM_PROMPT, "What is 47,915,884 divided by 12,736?",
        "That's beyond what I compute reliably: I don't know the exact result.",
    ),
    (
        DEFAULT_SYSTEM_PROMPT, "When did my grandfather buy his first car?",
        "I have no way of knowing that. Nobody told me about your grandfather, so I'd just be guessing.",
    ),
]


def variation_pairs(repeat=1):
    pairs = []
    for system, question, answer in VARIATIONS + UNCERTAINTY_PAIRS:
        for _ in range(repeat):
            pairs.append((system, question, answer))
    return pairs
