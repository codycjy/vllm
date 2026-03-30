#!/usr/bin/env python3
"""Generate synthetic RAG workload data for prefix cache benchmarking.

Simulates retrieval-augmented generation with Zipfian document popularity:
- Document pool with skewed access frequency (few hot docs, many cold docs)
- Each request: "Based on this document: [doc]\n\nQuestion: [query]"
- Hot documents create shared prefixes → high cache reuse

Output: vLLM custom JSONL format
"""

import argparse
import json
import math
import random
import string
from pathlib import Path

# Paragraph templates for generating synthetic documents
DOCUMENT_TOPICS = [
    ("Machine Learning Fundamentals",
     "Neural networks consist of layers of interconnected nodes that process "
     "information using weighted connections. Training involves adjusting these "
     "weights through backpropagation to minimize a loss function. Key concepts "
     "include gradient descent, regularization, and hyperparameter tuning. "
     "Common architectures include feedforward networks, convolutional neural "
     "networks for image processing, and recurrent neural networks for "
     "sequential data. Modern approaches leverage attention mechanisms and "
     "transformer architectures for improved performance on various tasks."),

    ("Cloud Computing Architecture",
     "Cloud computing provides on-demand access to computing resources including "
     "servers, storage, databases, networking, and software. The three main "
     "service models are Infrastructure as a Service (IaaS), Platform as a "
     "Service (PaaS), and Software as a Service (SaaS). Key architectural "
     "patterns include microservices, serverless computing, and containerization. "
     "Organizations must consider scalability, reliability, security, and cost "
     "optimization when designing cloud-native applications."),

    ("Database Systems and Optimization",
     "Modern database systems range from traditional relational databases to "
     "NoSQL solutions including document stores, key-value stores, column-family "
     "stores, and graph databases. Query optimization involves index selection, "
     "join ordering, and execution plan analysis. Advanced topics include "
     "distributed transactions, eventual consistency, sharding strategies, "
     "and replication protocols. Performance tuning requires understanding "
     "of storage engines, buffer pool management, and query execution plans."),

    ("Cybersecurity Best Practices",
     "Information security encompasses confidentiality, integrity, and "
     "availability of data and systems. Key areas include network security, "
     "application security, identity and access management, and incident "
     "response. Common threats include phishing, ransomware, supply chain "
     "attacks, and zero-day exploits. Defense strategies involve layered "
     "security, zero trust architecture, encryption at rest and in transit, "
     "and continuous monitoring with SIEM solutions."),

    ("Distributed Systems Design",
     "Distributed systems coordinate multiple computers to achieve a common "
     "goal. Key challenges include network partitions, clock synchronization, "
     "consensus, and fault tolerance. Important algorithms include Paxos, Raft "
     "for consensus, vector clocks for ordering, and consistent hashing for "
     "load distribution. The CAP theorem states that a distributed system can "
     "provide at most two of consistency, availability, and partition tolerance "
     "simultaneously."),

    ("Operating Systems Internals",
     "Operating systems manage hardware resources and provide abstractions "
     "for application programs. Core components include process management, "
     "memory management, file systems, and I/O systems. Process scheduling "
     "algorithms balance throughput, latency, and fairness. Virtual memory "
     "uses paging and segmentation to provide each process with an isolated "
     "address space. Modern OS features include namespaces, cgroups for "
     "containerization, and eBPF for programmable kernel extensions."),

    ("Software Engineering Practices",
     "Software engineering encompasses methodologies, tools, and practices "
     "for building reliable software systems. Agile methodologies including "
     "Scrum and Kanban emphasize iterative development and continuous feedback. "
     "Version control with Git enables collaboration and code review workflows. "
     "Continuous integration and deployment automate testing and delivery. "
     "Code quality practices include testing at multiple levels, static "
     "analysis, and architectural decision records."),

    ("Natural Language Processing",
     "NLP enables computers to understand, interpret, and generate human "
     "language. Key tasks include tokenization, part-of-speech tagging, named "
     "entity recognition, sentiment analysis, and machine translation. Modern "
     "approaches use transformer-based models like BERT, GPT, and T5 which "
     "are pretrained on large corpora and fine-tuned for specific tasks. "
     "Challenges include handling ambiguity, context, multilingual processing, "
     "and ensuring fairness and safety in language generation."),

    ("Computer Networks and Protocols",
     "Computer networks enable communication between devices through layered "
     "protocol stacks. The TCP/IP model includes link, internet, transport, "
     "and application layers. HTTP/HTTPS powers the web, while DNS provides "
     "name resolution. Modern networking includes software-defined networking, "
     "content delivery networks, and service meshes. Performance optimization "
     "involves techniques like connection pooling, compression, caching, "
     "and load balancing across multiple servers."),

    ("Data Engineering and Pipelines",
     "Data engineering focuses on building systems to collect, store, and "
     "analyze data at scale. ETL pipelines extract data from sources, "
     "transform it for analysis, and load it into data warehouses. Modern "
     "architectures use data lakes, lakehouse patterns, and streaming "
     "platforms like Kafka. Data quality involves validation, deduplication, "
     "schema evolution, and lineage tracking. Orchestration tools like "
     "Airflow manage complex workflow dependencies."),

    ("Compiler Design and Programming Languages",
     "Compilers translate source code into machine code through phases "
     "including lexical analysis, parsing, semantic analysis, optimization, "
     "and code generation. Intermediate representations enable portable "
     "optimization passes. Just-in-time compilation combines interpretation "
     "flexibility with compiled performance. Modern language features include "
     "type inference, pattern matching, algebraic data types, and ownership "
     "systems for memory safety without garbage collection."),

    ("Quantum Computing Concepts",
     "Quantum computing leverages quantum mechanical phenomena including "
     "superposition and entanglement to perform computations. Qubits can "
     "exist in superpositions of 0 and 1 states simultaneously. Quantum "
     "gates manipulate qubits to implement algorithms. Notable algorithms "
     "include Shor's algorithm for factoring and Grover's algorithm for "
     "search. Current challenges include decoherence, error correction, "
     "and scaling up the number of reliable qubits."),

    ("Robotics and Autonomous Systems",
     "Robotics integrates mechanical engineering, electrical engineering, "
     "and computer science to create autonomous machines. Key components "
     "include sensors for perception, actuators for movement, and control "
     "systems for decision making. SLAM algorithms enable simultaneous "
     "localization and mapping. Motion planning algorithms find collision-free "
     "paths through configuration space. Reinforcement learning enables "
     "robots to learn complex behaviors through trial and error."),

    ("Blockchain and Distributed Ledgers",
     "Blockchain technology provides a decentralized, immutable ledger for "
     "recording transactions. Consensus mechanisms including Proof of Work "
     "and Proof of Stake ensure agreement among distributed participants. "
     "Smart contracts enable programmable business logic on the blockchain. "
     "Key challenges include scalability, energy consumption, privacy, and "
     "regulatory compliance. Layer 2 solutions and sharding aim to improve "
     "transaction throughput while maintaining security guarantees."),

    ("Human-Computer Interaction",
     "HCI studies the design and use of computer technology at the interface "
     "between people and computers. Key principles include visibility, "
     "feedback, affordance, and consistency. User research methods include "
     "interviews, surveys, usability testing, and A/B testing. Accessibility "
     "ensures technology is usable by people with diverse abilities. Modern "
     "trends include voice interfaces, augmented reality, and adaptive "
     "interfaces that personalize to individual users."),

    ("Computer Graphics and Visualization",
     "Computer graphics generates images using computational techniques. "
     "The rendering pipeline transforms 3D geometry through model, view, "
     "and projection transformations before rasterization. Ray tracing "
     "simulates light transport for photorealistic rendering. GPU computing "
     "with shaders enables real-time graphics. Scientific visualization "
     "maps data to visual representations to reveal patterns. Modern "
     "techniques include neural radiance fields and differentiable rendering."),

    ("Bioinformatics and Computational Biology",
     "Bioinformatics applies computational methods to biological data. "
     "Sequence alignment algorithms like BLAST compare DNA and protein "
     "sequences. Genome assembly reconstructs complete genomes from short "
     "reads. Phylogenetic analysis infers evolutionary relationships. "
     "Machine learning predicts protein structures, drug interactions, "
     "and gene expression patterns. Single-cell analysis reveals cellular "
     "heterogeneity at unprecedented resolution."),

    ("Embedded Systems and IoT",
     "Embedded systems are specialized computers within larger devices. "
     "Real-time operating systems guarantee timing constraints for critical "
     "applications. IoT connects physical devices to the internet for "
     "monitoring and control. Communication protocols include MQTT, CoAP, "
     "and LoRaWAN for different range and power requirements. Edge computing "
     "processes data near the source to reduce latency and bandwidth. "
     "Security is critical as IoT devices often have limited resources."),

    ("Information Retrieval and Search",
     "Information retrieval systems find relevant documents from large "
     "collections. Inverted indexes map terms to document lists for efficient "
     "lookup. TF-IDF and BM25 rank documents by relevance. Modern systems "
     "use learned dense representations for semantic search. Query processing "
     "includes expansion, spell correction, and intent classification. "
     "Evaluation metrics include precision, recall, NDCG, and MAP. "
     "Personalization adapts results to individual user preferences."),

    ("Parallel and High-Performance Computing",
     "HPC enables solving large computational problems through parallelism. "
     "Shared memory programming uses threads and synchronization primitives. "
     "Distributed memory programming uses message passing with MPI. GPU "
     "computing with CUDA enables massive parallelism for data-parallel "
     "workloads. Performance analysis identifies bottlenecks in computation, "
     "communication, and memory access. Scalability is measured by speedup "
     "and efficiency as processor count increases."),
]

QUESTION_TEMPLATES = [
    "What are the key concepts discussed in this document?",
    "Summarize the main points in 3-4 sentences.",
    "What challenges are mentioned in this document?",
    "How do the concepts described here relate to modern software development?",
    "What are the practical applications of the topics discussed?",
    "Explain the trade-offs mentioned in this document.",
    "What future directions or trends are implied by this content?",
    "Compare and contrast two main concepts from this document.",
    "What prerequisites would someone need to understand this content?",
    "How would you explain the core idea of this document to a beginner?",
    "What are the most important terms and their definitions?",
    "Identify potential limitations of the approaches described.",
]


def zipfian_distribution(n: int, s: float = 1.0) -> list[float]:
    """Return Zipfian probabilities for n items with exponent s."""
    weights = [1.0 / (k ** s) for k in range(1, n + 1)]
    total = sum(weights)
    return [w / total for w in weights]


def expand_document(title: str, base_text: str, rng: random.Random,
                    target_words: int = 300) -> str:
    """Expand a short document template to target length."""
    doc = f"# {title}\n\n{base_text}"
    current_words = len(doc.split())

    # Pad with elaboration paragraphs
    elaborations = [
        f"\nFurthermore, {title.lower()} involves considerations of scalability, "
        f"maintainability, and performance optimization. Practitioners must "
        f"balance theoretical foundations with practical constraints.",
        f"\nRecent advances in {title.lower()} have been driven by improvements "
        f"in hardware capabilities, algorithm design, and the availability of "
        f"large-scale datasets for training and evaluation.",
        f"\nThe field of {title.lower()} continues to evolve rapidly, with new "
        f"tools, frameworks, and methodologies emerging regularly. Staying "
        f"current requires continuous learning and adaptation.",
    ]

    for elab in elaborations:
        if current_words >= target_words:
            break
        doc += elab
        current_words = len(doc.split())

    return doc


def generate_dataset(args) -> list[dict]:
    rng = random.Random(args.seed)
    num_docs = min(args.num_documents, len(DOCUMENT_TOPICS))
    topics = DOCUMENT_TOPICS[:num_docs]

    # Build document pool
    documents = []
    for title, text in topics:
        doc = expand_document(title, text, rng, target_words=args.doc_words)
        documents.append((title, doc))

    # Zipfian popularity: first few docs are much more popular
    probs = zipfian_distribution(num_docs, s=args.zipf_exponent)
    doc_indices = list(range(num_docs))

    records = []
    for _ in range(args.num_requests):
        # Select document by Zipfian popularity
        doc_idx = rng.choices(doc_indices, weights=probs, k=1)[0]
        title, doc_text = documents[doc_idx]
        question = rng.choice(QUESTION_TEMPLATES)

        prompt = (
            f"Based on the following document:\n\n"
            f"{doc_text}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        records.append({
            "prompt": prompt,
            "output_tokens": args.output_tokens,
            "prefix_group": f"doc_{doc_idx}",
            "doc_title": title,
        })

    rng.shuffle(records)
    return records


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic RAG data")
    parser.add_argument("--num-documents", type=int, default=20,
                        help="Document pool size (max 20)")
    parser.add_argument("--doc-words", type=int, default=300,
                        help="Target words per document")
    parser.add_argument("--zipf-exponent", type=float, default=1.0,
                        help="Zipfian exponent (higher = more skewed)")
    parser.add_argument("--num-requests", type=int, default=500,
                        help="Total requests to generate")
    parser.add_argument("--output-tokens", type=int, default=128,
                        help="Expected output tokens per request")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    records = generate_dataset(args)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print stats
    doc_counts = {}
    for r in records:
        doc_counts[r["doc_title"]] = doc_counts.get(r["doc_title"], 0) + 1

    print(f"Generated {len(records)} RAG requests → {args.output}")
    print(f"  Documents: {len(doc_counts)}, Zipf exponent: {args.zipf_exponent}")
    print(f"  Document access distribution:")
    for title, cnt in sorted(doc_counts.items(), key=lambda x: -x[1]):
        bar = "#" * (cnt * 40 // max(doc_counts.values()))
        print(f"    {title[:35]:35s} {cnt:4d} {bar}")


if __name__ == "__main__":
    main()
