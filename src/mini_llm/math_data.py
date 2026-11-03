import random

ADD = ["What is {a} + {b}?", "what is {a} + {b}", "what's {a}+{b}", "{a} + {b}", "How much is {a} plus {b}?"]
SUB = ["What is {a} - {b}?", "what is {a} - {b}", "what's {a}-{b}", "{a} - {b}", "How much is {a} minus {b}?"]
MUL = ["What is {a} * {b}?", "what is {a} x {b}", "what's {a}x{b}", "{a} x {b}", "How much is {a} times {b}?"]
DIV = [
    "What is {a} / {b}?", "what is {a} / {b}", "what's {a}/{b}", "{a} / {b}", "How much is {a} divided by {b}?",
]

TABLE_MAX = 12
WORDS = [
    "sun", "apple", "dance", "river", "cloud", "train", "night", "green", "smile", "bread",
    "ocean", "music", "dream", "horse", "stone", "light", "chair", "fruit", "quiet", "shore",
]

OP_WEIGHTS = {"add": 4, "sub": 4, "mul": 3, "div": 3}
ADD_SUB_WIDTH_WEIGHTS = {1: 25, 2: 25, 3: 18, 4: 14, 5: 10, 6: 8}
MUL_WIDTH_WEIGHTS = {(2, 1): 22, (2, 2): 20, (3, 1): 18, (3, 2): 16, (4, 1): 10, (4, 2): 8, (5, 1): 6}
DIV_QUOTIENT_WEIGHTS = {2: 40, 3: 35, 4: 25}
NEGATIVE_SHARE = 0.1
CARRY_SHARE = 0.3
SPLIT_SHARE = 0.5

SIGNS = {"add": "+", "sub": "-", "mul": "x", "div": "/"}


def number(rng, digits):
    low = 0 if digits == 1 else 10 ** (digits - 1)
    return rng.randint(low, 10 ** digits - 1)


def choose_weighted(rng, weights):
    keys = sorted(weights)
    r = rng.randint(1, sum(weights.values()))
    for key in keys:
        r -= weights[key]
        if r <= 0:
            return key
    return keys[-1]


def template(rng, base):
    return rng.choice([base[0], base[2], base[1], base[3], base[4]])


def plain_answer(kind, a, b, result):
    return f"{a} {SIGNS[kind]} {b} = {result}."


def carried_answer(kind, a, b, result):
    step = (b // 100) * 100
    if step <= 0 or step >= b:
        return plain_answer(kind, a, b, result)
    sign = SIGNS[kind]
    mid = a + step if kind == "add" else a - step
    return f"{a} {sign} {b} = {a} {sign} {step} = {mid}, then {mid} {sign} {b - step} = {result}."


def split_answer(a, b):
    tens, ones = divmod(b, 10)
    if ones == 0:
        return plain_answer("mul", a, b, a * b)
    high = a * tens * 10
    low = a * ones
    return f"{a} x {b} = {a} x {tens * 10} + {a} x {ones} = {high} + {low} = {a * b}."


def add_sub_pair(rng, kind):
    width = choose_weighted(rng, ADD_SUB_WIDTH_WEIGHTS)
    a, b = number(rng, width), number(rng, width)
    negative = kind == "sub" and rng.random() < NEGATIVE_SHARE
    if not negative and b > a:
        a, b = b, a
    result = a + b if kind == "add" else a - b
    if width >= 4 and rng.random() < CARRY_SHARE:
        answer = carried_answer(kind, a, b, result)
    else:
        answer = plain_answer(kind, a, b, result)
    return template(rng, ADD if kind == "add" else SUB).format(a=a, b=b), answer


def mul_pair(rng):
    a_width, b_width = choose_weighted(rng, MUL_WIDTH_WEIGHTS)
    a = number(rng, a_width)
    b = rng.randint(1, 9) if b_width == 1 else number(rng, b_width)
    if b >= 10 and rng.random() < SPLIT_SHARE:
        answer = split_answer(a, b)
    else:
        answer = plain_answer("mul", a, b, a * b)
    return template(rng, MUL).format(a=a, b=b), answer


def div_pair(rng):
    quotient = number(rng, choose_weighted(rng, DIV_QUOTIENT_WEIGHTS))
    divisor = rng.randint(2, 99)
    a = divisor * quotient
    return template(rng, DIV).format(a=a, b=divisor), plain_answer("div", a, divisor, quotient)


def op_pair(rng):
    kind = choose_weighted(rng, OP_WEIGHTS)
    if kind == "add" or kind == "sub":
        return add_sub_pair(rng, kind)
    if kind == "mul":
        return mul_pair(rng)
    return div_pair(rng)


def table_pairs(seed=0):
    rng = random.Random(seed)
    pairs = []
    for a in range(1, TABLE_MAX + 1):
        for b in range(1, TABLE_MAX + 1):
            pairs.append((template(rng, MUL).format(a=a, b=b), plain_answer("mul", a, b, a * b)))
    for p in range(2, TABLE_MAX + 1):
        for q in range(1, TABLE_MAX + 1):
            pairs.append((template(rng, DIV).format(a=p * q, b=p), plain_answer("div", p * q, p, q)))
    return pairs


def synthetic_pairs(count, seed=1, table_share=0.1):
    rng = random.Random(seed + 1)
    tables = table_pairs(seed)
    n_tables = min(count, int(count * table_share))
    start = rng.randrange(len(tables))
    pool = [tables[(start + i) % len(tables)] for i in range(n_tables)]
    pool += [op_pair(rng) for _ in range(count - n_tables)]
    rng.shuffle(pool)
    return pool


def count_pair(rng):
    n = rng.randint(3, 8)
    words = rng.sample(WORDS, n)
    return f'How many words are in this sentence? "{" ".join(words)}"', f"{n} words."


def sort_pair(rng):
    k = rng.randint(3, 5)
    nums = rng.sample(range(0, 10000), k)
    text = ", ".join(str(x) for x in nums)
    if rng.choices(("up", "down"), weights=(3, 1))[0] == "up":
        out = sorted(nums)
        suffix = "ascending"
    else:
        out = sorted(nums, reverse=True)
        suffix = "descending"
    return (
        f"Sort these numbers: {text}",
        ", ".join(str(x) for x in out) + f" ({suffix}).",
    )


def reverse_pair(rng):
    if rng.random() < 0.5:
        word = rng.choice(WORDS)
        return f'Say "{word}" backwards', f"{word[::-1]}."
    digits = str(number(rng, rng.randint(2, 5)))
    return f'Read "{digits}" backwards', f"{digits[::-1]}."


SKILLS = [(count_pair, 5), (sort_pair, 3), (reverse_pair, 2)]


def skill_pairs(count, seed=2):
    rng = random.Random(seed + 2)
    names = [f for f, w in SKILLS for _ in range(w)]
    return [rng.choice(names)(rng) for _ in range(count)]


STORY_NAMES = ["Ana", "Ben", "Maya", "Leo", "Sara", "Tom", "Nina", "Omar", "Lily", "Raj"]
STORY_ITEMS = [
    ("apple", "apples"), ("pencil", "pencils"), ("cookie", "cookies"), ("marble", "marbles"),
    ("book", "books"), ("sticker", "stickers"), ("coin", "coins"), ("egg", "eggs"),
    ("orange", "oranges"), ("card", "cards"), ("ticket", "tickets"), ("shell", "shells"),
]
STORY_KINDS = ["add", "sub", "mul", "div", "two"]
STORY_WEIGHTS = [3, 3, 2, 2, 2]


def story_add(rng):
    name = rng.choice(STORY_NAMES)
    many = rng.choice(STORY_ITEMS)[1]
    a, b = rng.randint(2, 40), rng.randint(2, 30)
    question = f"{name} has {a} {many} and buys {b} more. How many {many} does {name} have now?"
    return question, f"{a} + {b} = {a + b}. The answer is {a + b}."


def story_sub(rng):
    name, other = rng.sample(STORY_NAMES, 2)
    many = rng.choice(STORY_ITEMS)[1]
    a = rng.randint(5, 60)
    b = rng.randint(2, a - 1)
    question = f"{name} has {a} {many} and gives {b} to {other}. How many {many} are left?"
    return question, f"{a} - {b} = {a - b}. The answer is {a - b}."


def story_mul(rng):
    name = rng.choice(STORY_NAMES)
    many = rng.choice(STORY_ITEMS)[1]
    boxes, per = rng.randint(2, 12), rng.randint(2, 12)
    question = f"{name} has {boxes} boxes with {per} {many} in each box. How many {many} are there in total?"
    return question, f"{boxes} x {per} = {boxes * per}. The answer is {boxes * per}."


def story_div(rng):
    name = rng.choice(STORY_NAMES)
    many = rng.choice(STORY_ITEMS)[1]
    per, friends = rng.randint(2, 9), rng.randint(2, 9)
    total = per * friends
    question = f"{name} shares {total} {many} equally among {friends} friends. How many {many} does each friend get?"
    return question, f"{total} / {friends} = {per}. The answer is {per}."


def story_two_step(rng):
    name, other = rng.sample(STORY_NAMES, 2)
    many = rng.choice(STORY_ITEMS)[1]
    a, b = rng.randint(3, 30), rng.randint(3, 25)
    give = rng.randint(1, a + b - 2)
    total, left = a + b, a + b - give
    question = (
        f"{name} picks {a} {many} on Monday and {b} {many} on Tuesday. "
        f"Then {name} gives {give} {many} to {other}. How many {many} does {name} have left?"
    )
    return question, f"{a} + {b} = {total}, then {total} - {give} = {left}. The answer is {left}."


STORY_MAKERS = {
    "add": story_add, "sub": story_sub, "mul": story_mul, "div": story_div, "two": story_two_step,
}


def story_pair(rng):
    return STORY_MAKERS[rng.choices(STORY_KINDS, weights=STORY_WEIGHTS)[0]](rng)


def story_pairs(count, seed=4):
    rng = random.Random(seed + 4)
    return [story_pair(rng) for _ in range(count)]

