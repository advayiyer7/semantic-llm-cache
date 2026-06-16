"""Generate a documented, reproducible request workload.

Mix (default 40/30/30):
  - exact:      verbatim repeat of a previously-seen prompt
  - paraphrase: same meaning, different surface form (tests *semantic* hits)
  - unique:     a brand-new prompt (should always miss)

Reporting savings against a *stated* mix is what keeps the headline number
honest — a workload of all-repeats would inflate it; all-unique would zero it.
"""

from __future__ import annotations

import random

_BASE_PROMPTS = [
    "What is the capital of France?",
    "How does a hash map work?",
    "What causes the seasons on Earth?",
    "Explain the difference between TCP and UDP.",
    "What is the time complexity of binary search?",
    "How do vaccines train the immune system?",
    "What is a closure in JavaScript?",
    "Why is the sky blue?",
    "What does the GIL do in Python?",
    "How does HTTPS keep traffic private?",
    "What is the purpose of an index in a database?",
    "Explain what a deadlock is.",
    "What is the difference between a process and a thread?",
    "How does garbage collection work?",
    "What is a vector embedding?",
    "Explain the CAP theorem.",
    "What is idempotency in REST APIs?",
    "How does DNS resolve a domain name?",
    "What is the difference between SQL and NoSQL?",
    "What is a race condition?",
    "How does public-key cryptography work?",
    "What is backpropagation?",
    "Explain eventual consistency.",
    "What is a memory leak?",
]

_PARAPHRASE_TEMPLATES = [
    "Can you tell me: {q}",
    "I'd like to understand — {q}",
    "Quick question: {q}",
    "Could you explain, {q}",
    "In simple terms, {q}",
]

# Diverse subjects + templates so generated "unique" prompts are *semantically*
# distinct (otherwise near-identical fillers false-hit each other and inflate the
# hit rate). 6 templates x ~90 subjects = ~540 low-similarity combinations.
_UNIQUE_TEMPLATES = [
    "What is {x}?",
    "How does {x} work?",
    "Why does {x} matter?",
    "Explain {x} simply.",
    "What causes {x}?",
    "Give an overview of {x}.",
]
_SUBJECTS = [
    "photosynthesis", "inflation", "the French Revolution", "quicksort", "mitochondria",
    "blockchain", "plate tectonics", "the Doppler effect", "compound interest", "osmosis",
    "the water cycle", "machine learning", "the stock market", "antibiotics", "gravity",
    "the immune system", "supply and demand", "quantum entanglement", "the Cold War",
    "natural selection", "the greenhouse effect", "encryption", "the nervous system",
    "tectonic earthquakes", "the electoral college", "carbon dating", "neural networks",
    "the Krebs cycle", "monetary policy", "black holes", "the Renaissance", "vaccination",
    "tidal forces", "recursion", "the Magna Carta", "genetic mutation", "the ozone layer",
    "interest rates", "sound waves", "the printing press", "cellular respiration",
    "the Roman Empire", "fermentation", "the placebo effect", "binary numbers",
    "the industrial revolution", "DNA replication", "the speed of light", "tariffs",
    "erosion", "the scientific method", "the Big Bang", "protein folding", "deforestation",
    "the gold standard", "convection", "the Silk Road", "antibodies", "tessellation",
    "the Pythagorean theorem", "evaporation", "the assembly line", "homeostasis",
    "the stock exchange", "radioactivity", "the Enlightenment", "photosynthetic pigments",
    "market equilibrium", "wave interference", "the feudal system", "enzyme catalysis",
    "the Bill of Rights", "thermodynamics", "the nitrogen cycle", "gerrymandering",
    "capillary action", "the Trojan War", "mitosis", "exchange rates", "magnetism",
    "the Berlin Wall", "lactic acid", "hash functions", "the Louisiana Purchase",
    "diffusion", "the Bronze Age", "synaptic transmission", "quantitative easing",
    "refraction", "the Great Depression", "ribosomes",
]


def _paraphrase(prompt: str, rng: random.Random) -> str:
    core = prompt[0].lower() + prompt[1:]
    return rng.choice(_PARAPHRASE_TEMPLATES).format(q=core)


def build_workload(
    n: int,
    seed: int = 7,
    mix: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> list[dict]:
    """Return n request items, each {"prompt": str, "kind": exact|paraphrase|unique}."""
    rng = random.Random(seed)
    exact_p, para_p, _ = mix
    pool = list(_BASE_PROMPTS)
    rng.shuffle(pool)
    used_unique: set[str] = set()

    def _next_unique() -> str:
        # Real questions first, then diverse generated ones; both kept distinct.
        for base in pool:
            if base not in used_unique:
                used_unique.add(base)
                return base
        for _ in range(60):
            candidate = rng.choice(_UNIQUE_TEMPLATES).format(x=rng.choice(_SUBJECTS))
            if candidate not in used_unique:
                used_unique.add(candidate)
                return candidate
        return f"Tell me one fact about {rng.choice(_SUBJECTS)} (#{len(used_unique)})."

    seen: list[str] = []
    items: list[dict] = []
    for _ in range(n):
        roll = rng.random()
        if seen and roll < exact_p:
            items.append({"prompt": rng.choice(seen), "kind": "exact"})
        elif seen and roll < exact_p + para_p:
            items.append({"prompt": _paraphrase(rng.choice(seen), rng), "kind": "paraphrase"})
        else:
            base = _next_unique()
            seen.append(base)
            items.append({"prompt": base, "kind": "unique"})
    return items


def mix_counts(items: list[dict]) -> dict[str, int]:
    counts = {"exact": 0, "paraphrase": 0, "unique": 0}
    for it in items:
        counts[it["kind"]] += 1
    return counts
