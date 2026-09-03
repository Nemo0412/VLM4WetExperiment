"""FineBio-style protocol cards with hazard lines."""

from __future__ import annotations

PROTOCOL_NAMES = {
    1: "Cell lysate collection (single PBS wash)",
    2: "Cell lysate collection (double PBS wash)",
    3: "Magnetic-bead DNA extraction (single ethanol wash)",
    4: "Magnetic-bead DNA extraction (double ethanol wash)",
    5: "PCR reaction setup with 8-tube strips",
    6: "Spin-column DNA extraction (two wash steps)",
    7: "Spin-column DNA extraction (three wash steps)",
}

HAZARDS = {
    1: ["Pipette", "Culture plate", "Cell lysate reagent"],
    2: ["Pipette", "Culture plate", "Cell lysate reagent"],
    3: ["Pipette", "Centrifuge", "Magnetic rack", "70% ethanol (flammable)"],
    4: ["Pipette", "Centrifuge", "Magnetic rack", "70% ethanol (flammable)"],
    5: ["Pipette", "PCR machine (heat)", "8-tube strips"],
    6: ["Pipette", "Centrifuge", "Spin column", "Wash buffer"],
    7: ["Pipette", "Centrifuge", "Spin column", "Wash buffer"],
}

STEPS = {
    1: [
        "Remove culture medium",
        "Add PBS",
        "Shake plate",
        "Aspirate PBS",
        "Add cell lysate",
        "Shake plate",
        "Transfer cell lysate to tube",
        "Spindown",
        "Aspirate supernatant",
    ],
    2: [
        "Remove culture medium",
        "Add PBS",
        "Shake plate",
        "Aspirate PBS",
        "Add PBS again (second wash)",
        "Shake plate",
        "Aspirate PBS",
        "Add cell lysate",
        "Shake plate",
        "Transfer cell lysate to tube",
        "Spindown",
        "Aspirate supernatant",
    ],
    3: [
        "Add magnetic beads to the sample tube",
        "Pipette to mix",
        "Vortex",
        "Spin down in centrifuge",
        "Place tube in magnetic rack",
        "Aspirate supernatant",
        "Add wash buffer",
        "Mix, vortex, spin down, rack, aspirate",
        "Add 70% ethanol",
        "Mix, vortex, spin down, rack, aspirate",
        "Add sterile water",
        "Mix, vortex, spin down, rack",
        "Transfer supernatant to a new empty tube",
    ],
    4: [
        "Add magnetic beads",
        "Mix / vortex / spin / magnetic rack / aspirate",
        "Add wash buffer (first)",
        "Mix / vortex / spin / rack / aspirate",
        "Add wash buffer (second)",
        "Mix / vortex / spin / rack / aspirate",
        "Add 70% ethanol",
        "Mix / vortex / spin / rack / aspirate",
        "Add sterile water",
        "Transfer supernatant to a new tube",
    ],
    5: [
        "Transfer sample to 8-tube strips",
        "Transfer PCR mix",
        "Transfer forward primer",
        "Transfer reverse primer",
        "Transfer template DNA",
        "Transfer water",
        "Close 8-tube strip lids",
        "Vortex strips",
        "Spindown strips",
        "Load strips into PCR machine",
    ],
    6: [
        "Add binding buffer",
        "Transfer sample to spin-column tube",
        "Spindown",
        "Add wash buffer",
        "Spindown",
        "Add wash buffer (second)",
        "Spindown",
        "Detach spin column and insert into a new tube",
        "Add extract / dispense through column",
        "Spindown and collect",
    ],
    7: [
        "Add binding buffer",
        "Transfer sample to spin-column tube",
        "Spindown",
        "Add wash buffer (three washes)",
        "Detach spin column and insert into a new tube",
        "Add extract / dispense through column",
        "Spindown and collect",
    ],
}


def format_protocol(protocol_id: int) -> str:
    name = PROTOCOL_NAMES[protocol_id]
    hazards = ", ".join(HAZARDS[protocol_id])
    body = "\n".join(
        f"  P{i}. {step}" for i, step in enumerate(STEPS[protocol_id], 1)
    )
    return (
        f"Protocol {protocol_id}: {name}\n"
        f"Hazard instruments: {hazards}\n"
        f"Steps (must be followed in order):\n{body}"
    )
