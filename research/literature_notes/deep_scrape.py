"""
Deep scrape of HippoRAG, RepoCoder, SWE-agent, Agentless, and Haveliwala-Kamvar.
Extract exact formulas, algorithms, parameters, metrics, and experimental methodology.
Writes directly to tests/extracted_paper_insights.md with UTF-8 encoding.
"""

from __future__ import annotations

import re
from pathlib import Path
from scrapling import Fetcher

papers = [
    ("HippoRAG (NeurIPS 2024)", "https://arxiv.org/html/2405.14831v1"),
    ("RepoCoder (EMNLP 2023)", "https://arxiv.org/html/2303.12570v1"),
    ("SWE-agent (NeurIPS 2024)", "https://arxiv.org/html/2405.15793v1"),
    ("Agentless (FSE 2025)", "https://arxiv.org/html/2407.01489v1"),
]

def run_deep_extraction():
    out_file = Path("tests/extracted_paper_insights.md")
    lines = []
    lines.append("# Deep Extraction of Reference Papers: Methodology, Metrics, and Proven Facts\n")

    for title, url in papers:
        lines.append(f"## {title}")
        lines.append(f"**Source URL**: {url}\n")
        try:
            page = Fetcher.get(url)
            lines.append(f"- **HTTP Status**: {page.status}")
            actual_title = page.css("h1.title::text, title::text").get()
            lines.append(f"- **Document Title**: {actual_title.strip() if actual_title else 'N/A'}")

            # Extract abstract
            paras = page.css("p.ltx_p").getall()
            lines.append("### Key Methodological & Experimental Findings")

            matched_snippets = []
            for p in paras:
                clean_p = re.sub(r"<[^>]+>", " ", p).strip()
                clean_p = re.sub(r"\s+", " ", clean_p)
                # Match keywords: pagerank, diffusion, damping, recall, mrr, context, sliding window, tokens, ACI, localization, budget
                keywords = ["pagerank", "damping", "ppr", "hipporag", "repocoder", "swe-agent", "agentless", "recall@", "mrr", "context window", "token", "budget", "localization", "aci", "search-and-replace", "indentation", "syntax error"]
                if any(k in clean_p.lower() for k in keywords) and len(clean_p) > 80:
                    matched_snippets.append(clean_p)

            lines.append(f"- Total matched relevant technical paragraphs: {len(matched_snippets)}\n")
            for idx, snip in enumerate(matched_snippets[:12]):
                lines.append(f"#### Finding {idx+1}")
                lines.append(f"> {snip}\n")

        except Exception as e:
            lines.append(f"- **Error extracting {title}**: {e}\n")

    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Extraction complete. Written to {out_file}")

if __name__ == "__main__":
    run_deep_extraction()
