#!/usr/bin/env python3
"""Generate synthetic burst workload data for prefix cache benchmarking.

Simulates hot-prefix + cold-prompt bursty arrival patterns:
- Hot prefixes: fixed system prompts reused across many requests
- Cold prompts: unique one-off requests with no prefix sharing
- Burst windows: groups of requests arriving together

Output: vLLM custom JSONL format
  {"prompt": "...", "output_tokens": N, "prefix_group": "hot_0"|"cold"}
"""

import argparse
import json
import random
import string
from pathlib import Path

# A pool of realistic system prompts used as hot prefixes
SYSTEM_PROMPTS = [
    (
        "You are a helpful AI assistant specialized in software engineering. "
        "You help users write clean, efficient, and well-documented code. "
        "You follow best practices for code review, testing, and deployment. "
        "When asked about code, you provide detailed explanations with examples. "
        "You are familiar with Python, JavaScript, Go, Rust, and Java. "
        "Always consider edge cases, error handling, and performance implications."
    ),
    (
        "You are a medical information assistant. You provide accurate, "
        "evidence-based health information while always reminding users to "
        "consult qualified healthcare professionals for medical advice. "
        "You explain medical concepts in clear, accessible language. "
        "You reference current medical guidelines and research when possible. "
        "You never diagnose conditions or prescribe treatments."
    ),
    (
        "You are a financial advisor assistant. You help users understand "
        "investment concepts, portfolio management, tax implications, and "
        "retirement planning. You explain complex financial instruments in "
        "simple terms. You always remind users that past performance does "
        "not guarantee future results and recommend consulting with a "
        "certified financial planner for personalized advice."
    ),
    (
        "You are an educational tutor specializing in mathematics and science. "
        "You break down complex problems into manageable steps and guide "
        "students through solutions rather than providing direct answers. "
        "You use analogies and real-world examples to explain abstract concepts. "
        "You adapt your explanations to the student's level of understanding "
        "and encourage critical thinking and problem-solving skills."
    ),
    (
        "You are a creative writing assistant. You help users with story "
        "development, character creation, dialogue writing, and narrative "
        "structure. You provide constructive feedback on writing samples "
        "and suggest improvements while respecting the author's voice and "
        "style. You are knowledgeable about various literary genres, "
        "techniques, and publishing industry practices."
    ),
    (
        "You are a legal information assistant. You provide general legal "
        "information and help users understand legal concepts, terminology, "
        "and procedures. You explain different areas of law including "
        "contract law, intellectual property, employment law, and civil "
        "rights. You always emphasize that your information is educational "
        "and not a substitute for professional legal counsel."
    ),
    (
        "You are a data science and machine learning expert assistant. "
        "You help users with data analysis, statistical modeling, feature "
        "engineering, and model selection. You explain algorithms, evaluation "
        "metrics, and best practices for ML pipelines. You are proficient "
        "with pandas, scikit-learn, PyTorch, TensorFlow, and common data "
        "visualization libraries. You emphasize reproducibility and rigor."
    ),
    (
        "You are a travel planning assistant. You help users plan trips "
        "by providing information about destinations, accommodations, "
        "transportation options, local customs, and travel logistics. "
        "You create detailed itineraries tailored to user preferences "
        "and budgets. You provide tips for safety, cultural etiquette, "
        "and making the most of travel experiences around the world."
    ),
]

# User question templates for hot-prefix requests
USER_QUESTIONS = [
    "Can you explain how {topic} works?",
    "What are the best practices for {topic}?",
    "Help me understand the difference between {topic_a} and {topic_b}.",
    "What should I consider when choosing {topic}?",
    "Can you give me an example of {topic}?",
    "What are the common mistakes people make with {topic}?",
    "How do I get started with {topic}?",
    "What are the pros and cons of {topic}?",
    "Can you summarize the key points about {topic}?",
    "What is the current state of {topic}?",
    "How does {topic} compare to alternatives?",
    "What resources do you recommend for learning {topic}?",
]

TOPICS = [
    "machine learning", "distributed systems", "database indexing",
    "API design", "microservices", "container orchestration",
    "CI/CD pipelines", "load balancing", "caching strategies",
    "message queues", "data pipelines", "monitoring and observability",
    "authentication", "encryption", "cloud architecture",
    "serverless computing", "graph databases", "search engines",
    "recommendation systems", "natural language processing",
    "computer vision", "reinforcement learning", "transfer learning",
    "time series analysis", "A/B testing", "feature stores",
    "model serving", "data governance", "stream processing",
    "edge computing", "WebAssembly", "quantum computing basics",
]


def generate_user_question(rng: random.Random) -> str:
    template = rng.choice(USER_QUESTIONS)
    topic = rng.choice(TOPICS)
    topic_a = rng.choice(TOPICS)
    topic_b = rng.choice([t for t in TOPICS if t != topic_a])
    return template.format(topic=topic, topic_a=topic_a, topic_b=topic_b)


def generate_cold_prompt(rng: random.Random, min_len: int, max_len: int) -> str:
    """Generate a unique random prompt with no shared prefix."""
    length = rng.randint(min_len, max_len)
    words = []
    for _ in range(length):
        word_len = rng.randint(3, 10)
        words.append("".join(rng.choices(string.ascii_lowercase, k=word_len)))
    return " ".join(words)


def generate_dataset(args) -> list[dict]:
    rng = random.Random(args.seed)
    prompts = SYSTEM_PROMPTS[:args.num_hot_prefixes]
    num_hot = int(args.num_requests * args.hot_ratio)
    num_cold = args.num_requests - num_hot

    records = []

    # Generate hot-prefix requests
    for _ in range(num_hot):
        prefix_idx = rng.randrange(len(prompts))
        system = prompts[prefix_idx]
        question = generate_user_question(rng)
        prompt = f"[SYSTEM] {system} [/SYSTEM]\n\nUser: {question}\nAssistant:"
        records.append({
            "prompt": prompt,
            "output_tokens": args.output_tokens,
            "prefix_group": f"hot_{prefix_idx}",
        })

    # Generate cold (unique) requests
    for _ in range(num_cold):
        cold_text = generate_cold_prompt(rng, 30, 80)
        prompt = f"User: {cold_text}\nAssistant:"
        records.append({
            "prompt": prompt,
            "output_tokens": args.output_tokens,
            "prefix_group": "cold",
        })

    # Arrange in burst windows
    if args.burst_size > 0:
        rng.shuffle(records)
        bursts = []
        for i in range(0, len(records), args.burst_size):
            burst = records[i : i + args.burst_size]
            # Within each burst, cluster hot prefixes together to simulate
            # realistic arrival patterns (users in same session)
            burst.sort(key=lambda r: r["prefix_group"])
            bursts.extend(burst)
        records = bursts
    else:
        rng.shuffle(records)

    return records


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic burst data")
    parser.add_argument("--num-hot-prefixes", type=int, default=5,
                        help="Number of hot system prompt prefixes")
    parser.add_argument("--num-requests", type=int, default=500,
                        help="Total number of requests to generate")
    parser.add_argument("--hot-ratio", type=float, default=0.7,
                        help="Fraction of requests using hot prefixes")
    parser.add_argument("--burst-size", type=int, default=20,
                        help="Requests per burst window (0=fully shuffled)")
    parser.add_argument("--output-tokens", type=int, default=128,
                        help="Output tokens per request")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL path")
    args = parser.parse_args()

    records = generate_dataset(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print stats
    hot_count = sum(1 for r in records if r["prefix_group"] != "cold")
    cold_count = len(records) - hot_count
    prefixes_used = set(r["prefix_group"] for r in records if r["prefix_group"] != "cold")
    print(f"Generated {len(records)} requests → {output_path}")
    print(f"  Hot: {hot_count} ({hot_count/len(records)*100:.0f}%), "
          f"Cold: {cold_count} ({cold_count/len(records)*100:.0f}%)")
    print(f"  Hot prefixes used: {len(prefixes_used)}")
    print(f"  Burst window size: {args.burst_size}")


if __name__ == "__main__":
    main()
