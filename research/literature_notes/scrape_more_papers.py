import json
import re
from scrapling import Fetcher

targets = {
    "SWE-bench": "https://arxiv.org/html/2310.06770v2",
    "AutoCodeRover": "https://arxiv.org/html/2404.05427v1",
}

results = {}

for name, url in targets.items():
    print(f"Fetching {name} from {url}...")
    try:
        page = Fetcher.get(url)
        print(f"Status: {page.status}")
        
        # Extract title
        title = page.css("title::text").get() or name
        
        # Extract paragraphs containing technical formulas, tests, datasets, metrics
        paragraphs = page.css("p, .ltx_para, .ltx_theorem, .ltx_equation").getall()
        
        relevant_text = []
        keywords = [
            "experiment", "benchmark", "dataset", "metric", "recall", "pass@1", 
            "localization", "retrieval", "context", "token", "ast", "spectrum",
            "evaluat", "table", "baseline", "accuracy", "graph", "call"
        ]
        
        for p in paragraphs:
            # clean tags
            clean = re.sub(r"<[^>]+>", " ", p)
            clean = re.sub(r"\s+", " ", clean).strip()
            if len(clean) > 80:
                low = clean.lower()
                matches = sum(1 for kw in keywords if kw in low)
                if matches >= 2:
                    relevant_text.append(clean)
                    
        results[name] = {
            "title": title,
            "url": url,
            "status": page.status,
            "total_paras": len(relevant_text),
            "paragraphs": relevant_text[:25]
        }
    except Exception as e:
        print(f"Error fetching {name}: {e}")
        results[name] = {"error": str(e)}

with open("tests/extracted_swebench_autocoderover.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Saved to tests/extracted_swebench_autocoderover.json")
