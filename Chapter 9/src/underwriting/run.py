"""
run.py
======
Simplified underwriting pipeline with parallel data enrichment.

Usage:
    python -m src.underwriting.run

Demonstrates a sequential workflow whose enrichment stage applies
the parallelization style (Ch 7, Section 2.5).
"""

import asyncio


async def check_credit(applicant_id: str) -> dict:
    """Simulate credit check (parallel enrichment agent 1)."""
    await asyncio.sleep(1)  # Simulate API latency
    return {"applicant_id": applicant_id, "credit_score": 742,
            "credit_rating": "good"}


async def check_driving_record(license_number: str) -> dict:
    """Simulate driving record check (parallel enrichment agent 2)."""
    await asyncio.sleep(1.2)
    return {"license": license_number, "violations": 0,
            "accidents_3yr": 1, "risk_tier": "standard"}


async def check_property_value(address: str) -> dict:
    """Simulate property valuation (parallel enrichment agent 3)."""
    await asyncio.sleep(0.8)
    return {"address": address, "estimated_value": 425000,
            "risk_zone": "moderate", "year_built": 2005}


async def enrich_application(application: dict) -> dict:
    """Run three enrichment agents concurrently."""
    print("Running parallel data enrichment...")

    credit, driving, prop = await asyncio.gather(
        check_credit(application["applicant_id"]),
        check_driving_record(application["license_number"]),
        check_property_value(application["address"]),
    )

    print(f"  Credit score: {credit['credit_score']}")
    print(f"  Driving: {driving['violations']} violations, "
          f"{driving['accidents_3yr']} accidents")
    print(f"  Property value: ${prop['estimated_value']:,}")

    return {
        "credit": credit,
        "driving": driving,
        "property": prop,
    }


def assess_risk(enrichment: dict) -> dict:
    """Synthesize enrichment data into risk score and premium."""
    credit = enrichment["credit"]["credit_score"]
    accidents = enrichment["driving"]["accidents_3yr"]
    prop_value = enrichment["property"]["estimated_value"]

    # Simplified risk scoring
    base_score = 100
    if credit >= 750:
        base_score -= 10
    elif credit < 650:
        base_score += 20
    base_score += accidents * 15

    # Premium calculation (simplified)
    base_premium = prop_value * 0.004  # 0.4% of property value
    risk_multiplier = 1.0 + (base_score - 80) / 100

    return {
        "risk_score": base_score,
        "risk_tier": ("preferred" if base_score < 70
                      else "standard" if base_score < 100
                      else "high_risk"),
        "annual_premium": round(base_premium * risk_multiplier, 2),
        "decision": ("accept" if base_score < 120
                     else "refer_to_senior"),
    }


async def run_underwriting():
    """Run the complete underwriting pipeline."""
    application = {
        "applicant_id": "APP-2026-001",
        "name": "James Rodriguez",
        "license_number": "DL-NY-8834521",
        "address": "42 Oak Street, Westchester, NY 10601",
        "coverage_requested": "homeowners + auto bundle",
    }

    print(f"Underwriting application: {application['name']}\n")

    # Step 1: Parallel enrichment
    enrichment = await enrich_application(application)

    # Step 2: Risk assessment
    print("\nAssessing risk...")
    risk = assess_risk(enrichment)

    print(f"\n{'=' * 50}")
    print(f"  UNDERWRITING DECISION")
    print(f"{'=' * 50}")
    print(f"  Risk score: {risk['risk_score']}")
    print(f"  Risk tier: {risk['risk_tier']}")
    print(f"  Annual premium: ${risk['annual_premium']:,.2f}")
    print(f"  Decision: {risk['decision']}")

    return risk


if __name__ == "__main__":
    asyncio.run(run_underwriting())
