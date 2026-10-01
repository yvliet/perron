"""
Scrape and extract technical contents, formulas, and experimental methodology
from reference papers on arXiv using Scrapling.
"""

from __future__ import annotations

import re
from scrapling import Fetcher

papers = {
    "hipporag": "https://arxiv.org/html/2405.14831v1",
    "repocoder": "https://arxiv.org/html/2303.12570v1",
    "sweagent": "https://arxiv.org/html/2405.15793v1",
    "agentless": "https://arxiv.org/html/2407.01489v1",
}

def extract_paper_insights():
    extracted_data = {}
    for name, url in papers.items():
        print(f"\n==========================================")
        print(f"SCRAPING PAPER: {name.upper()} ({url})")
        print(f"==========================================")
        try:
            page = Fetcher.get(url)
            print(f"Status: {page.status}")
            title = page.css("h1.title::text, title::text").get()
            print(f"Title: {title.strip() if title else 'Unknown'}")

            # Extract abstract
            abstract = page.css("div.abstract::text, blockquote.abstract::text, p.ltx_p::text").getall()
            abstract_text = " ".join([a.strip() for a in abstract if a.strip()])[:600]
            print(f"Abstract snippet:\n{abstract_text}\n")

            # Look for sections related to PageRank, algorithms, retrieval, experiments, equations
            headings = page.css("h2.ltx_title, h3.ltx_title, h2::text, h3::text").getall()
            print(f"Headings found ({len(headings)}):", [h.strip() for h in headings[:8]])

            # Extract math formulas / equations
            math_blocks = page.css("span.ltx_Math, table.ltx_equation, div.ltx_equation").getall()
            print(f"Math blocks found: {len(math_blocks)}")

            # Extract text matching damping, pagerank, retrieval, metrics, recall, mrr
            text_blocks = page.css("p.ltx_p::text, div.ltx_para::text").getall()
            relevant_snippets = []
            for t in text_blocks:
                t_clean = t.strip()
                if any(w in t_clean.lower() for w in ["damping", "pagerank", "personalized", "damping factor", "beta", "recall@", "mrr", "teleport"]):
                    relevant_snippets.append(t_clean)
                    if len(relevant_snippets) >= 6:
                        break

            print(f"Relevant algorithm/metric snippets ({len(relevant_snippets)}):")
            for i, snip in enumerate(relevant_snippets):
                print(f"  [{i+1}] {snip[:250]}...")

            extracted_data[name] = {
                "title": title,
                "snippets": relevant_snippets,
            }
        except Exception as e:
            print(f"Error scraping {name}: {e}")

    return extracted_data

if __name__ == "__main__":
    extract_paper_insights()
