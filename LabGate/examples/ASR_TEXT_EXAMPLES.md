# LabGate examples (ASR = text proxy)

FineBio has no microphone track. These strings are treated as if speech
recognition already transcribed them.

## NONE — correct work, silence

- Video: `P25_01_01.mp4`
- ASR: *(empty)*
- Expected: Judger `NO`, no 32B call, output `NONE`

## ASSISTANT — user asks for help

- ASR: `What should I do next according to the protocol?`
- Expected: Judger `YES (user_query)`, expert `ASSISTANT`

Other examples:

- `Which buffer do I add in this step?`
- `How long should I vortex?`
- `Do I need sterile water after ethanol?`
- `Where should I put the supernatant?`

## SAFETY — spoken hazard

- ASR: `The centrifuge is spinning with the lid open.`
- Expected: Judger `YES (safety)`, expert `SAFETY`

Other examples:

- `I just spilled 70 percent ethanol next to a hot plate.`
- `I got lysate in my eye, it burns.`
- `Someone turned on a Bunsen burner near the ethanol wash.`
- `The centrifuge is making a loud grinding noise.`

## ACTION_ERROR — spoken self-report

- Video: correct `P25_03_01.mp4`
- ASR: `I think I skipped the sterile water wash.`
- Expected: expert `ACTION_ERROR`

## ACTION_ERROR — silent visual mistake

- Video: `P06_03_02.mp4`, missing sterile-water wash
- ASR: *(empty)*
- Expected: expert `ACTION_ERROR`
