"""snippets_deploy.py: Runnable illustrations for the deploying-and-versioning section.

Two mechanisms the chapter describes in prose but did not show as code:

1. A version manifest that pins model ID + prompts + tool code + configuration together
   and derives one fingerprint from the whole bundle; so "what version ran?" has a
   single answer, and a one-word prompt edit visibly produces a NEW version.
2. The shadow comparison of Figure 12.19: compare the candidate version's verdict
   distribution against the current version's on the same inputs and flag a material
   shift for human sign-off.

Offline and deterministic. 

Run it:  .venv/Scripts/python.exe snippets_deploy.py
"""

from __future__ import annotations

import hashlib
import json

# --- 1. The version manifest: pin everything, fingerprint the bundle -----------------------
def build_manifest(*, model: str, prompts: dict, tool_code_commit: str, config: dict) -> dict:
    return {"model": model,
            "prompt_sha": {name: hashlib.sha256(text.encode()).hexdigest()[:12]
                           for name, text in prompts.items()},
            "tool_code_commit": tool_code_commit,
            "config": config}


def version_fingerprint(manifest: dict) -> str:
    """One reproducible ID for the whole pinned bundle.

    The manifest carries the exact model ID (never a floating alias), the exact prompt
    texts, the tool-code commit, and the configuration. Hashing the canonicalised JSON
    means a change to ANY component -- one word of a prompt included -- yields a new
    version ID, which is precisely the discipline 'a prompt change is a code change'."""
    canonical = json.dumps(manifest, sort_keys=True)
    return "v-" + hashlib.sha256(canonical.encode()).hexdigest()[:12]


# --- 2. Shadow comparison: is the candidate's behaviour shift material? ---------------------
def shadow_compare(current: list[str], candidate: list[str],
                   material_flip_share: float = 0.10) -> dict:
    """Compare per-name verdicts from two versions run on the SAME shadow inputs.

    The offline golden set answers 'is the candidate still good?'; this answers a
    different question the golden set cannot: 'how much does it CHANGE the fleet's
    behaviour on live data?'. Above the threshold, the shift is material; not
    necessarily wrong, but a decision a human must sign off, not a silent cutover."""
    flips = sum(1 for cur, cand in zip(current, candidate) if cur != cand)
    flip_share = flips / len(current)
    return {"names": len(current), "flips": flips,
            "flip_share": round(flip_share, 3),
            "material": flip_share >= material_flip_share,
            "action": ("route to human sign-off before cutover"
                       if flip_share >= material_flip_share else "safe to canary")}


# --- Demo -------------------------------------------------------------------------------------
def main() -> None:
    print("1) a one-word prompt edit is a new version")
    base = dict(model="gpt-4o-mini-2024-07-18",            # exact ID, never a floating alias such as "gpt-4o-mini"
                tool_code_commit="9f31c2ab",
                config={"max_turns": 8, "temperature": 0.2,
                        "tools": ["get_key_ratios", "get_recent_news"]})
    prompt_v1 = "Synthesise a verdict from the ratios and the news sentiment."
    prompt_v2 = ("Synthesise a verdict from the ratios and the news sentiment, "
                 "weighting the profit-margin trend more heavily.")
    m1 = build_manifest(prompts={"synthesis": prompt_v1}, **base)
    m2 = build_manifest(prompts={"synthesis": prompt_v2}, **base)
    print(f"    current   : {version_fingerprint(m1)}")
    print(f"    candidate : {version_fingerprint(m2)}   (only the prompt changed)")
    print(json.dumps(m2, indent=6)[:220] + " ...\n")

    print("2) shadow comparison over 50 live names (the Figure 12.19 scenario)")
    current = ["FAIRLY_VALUED"] * 40 + ["OVERVALUED"] * 10
    candidate = list(current)
    for i in range(9):                                     # the edit flips 9/50 = 18% of names
        candidate[i] = "OVERVALUED"
    report = shadow_compare(current, candidate)
    print(f"    {report['flips']}/{report['names']} verdicts flip "
          f"({report['flip_share']:.0%})  material={report['material']}")
    print(f"    -> {report['action']}")


if __name__ == "__main__":
    main()
