# 逐决策原始指导与参考答案

作业 `16997938`。下表为模型原始输出，尚未由领域专家审核；门控正确不等于指导正确。

完整输入提示词和历史保存在 scratch 的 `*.predictions.jsonl`，汇总 CSV 为 [decisions.csv](decisions.csv)。

## gold

| 决策 | 时间(s) | 阶段 | GT | 小模型 | 大模型原始指导 | 参考指导 |
|---|---|---|---|---|---|---|
| d01 | 3.000 | protocol | interrupt | $silent$ | （未调用） | $interrupt$After sanitizing your hands, take the cell flask from the incubator. |
| d02 | 15.000 | protocol | silent | $interrupt$ | Disinfect the exterior of the cell flask using a disinfectant spray and allow it to sit for a moment before placing it in the biosafety cabinet. Ensure the UV light is off and the sash is at the recommended working height. | $silent$ |
| d03 | 18.000 | error | interrupt | $interrupt$ | Disinfect the exterior of the cell flask and place it in the biosafety cabinet. Ensure the UV light is off and the sash is at the recommended working height. | $interrupt$Stop—do not open or use the biosafety cabinet while the UV light is on. Turn off the UV light before proceeding. |
| d04 | 33.533 | recovery | silent | $interrupt$ | Disinfect the exterior of the cell flask using the spray bottle, ensuring thorough coverage. | $silent$ |
| d05 | 41.567 | error | interrupt | $interrupt$ | Turn off the UV light on the biosafety cabinet before proceeding. | $interrupt$Do not reach into the sterile cabinet with bare hands. Put on gloves and disinfect them with alcohol before continuing. |
| d06 | 66.097 | recovery | silent | $interrupt$ | Sanitize the exterior of the cell flask using 70% ethanol before placing it in the biosafety cabinet. Ensure the UV light is off and the sash is at the recommended working height. | $silent$ |
| d07 | 71.100 | error | interrupt | $interrupt$ | Sanitize the exterior of the cell flask using a disinfectant spray or wipe, ensuring thorough coverage before placing it inside the biosafety cabinet. | $interrupt$The cabinet sash is raised too high, which may compromise aseptic conditions. Lower it to the recommended working height before continuing. |
| d08 | 87.197 | recovery | silent | $interrupt$ | Sanitize the exterior of the cell flask using a disinfectant solution before placing it in the biosafety cabinet. | $silent$ |
| d09 | 110.200 | protocol | interrupt | $interrupt$ | Gently rock or tap the flask to distribute the trypsin evenly, avoiding vigorous shaking. | $interrupt$Now add PBS to rinse the cells, then discard the rinse. |
| d10 | 132.233 | error | interrupt | $interrupt$ | Add the equivalent of 2 volumes of complete growth medium to neutralize trypsin. Gently pipette to disperse the cells. | $interrupt$Do not leave the flask dry and exposed. Add PBS immediately to rinse the cells. |
| d11 | 166.433 | recovery | silent | $interrupt$ | Add the equivalent of 2 volumes of complete growth medium to neutralize trypsin. Gently pipette to disperse the cells. | $silent$ |
| d12 | 188.833 | protocol | interrupt | $interrupt$ | Add the equivalent of 2 volumes of complete growth medium to neutralize trypsin. Gently pipette to disperse the cells. | $interrupt$Add trypsin to cover the cell layer. |
| d13 | 217.833 | protocol | silent | $interrupt$ | Transfer the appropriate volume of the cell suspension to a new flask, add fresh culture medium, and label the new flask. Return it to the incubator. | $silent$ |
| d14 | 253.867 | error | interrupt | $interrupt$ | Gently pipette the cell suspension to disperse the cells evenly, ensuring they are well mixed. | $interrupt$Avoid vigorous shaking. Gently rock or tap the flask so the trypsin is distributed evenly. |
| d15 | 292.200 | recovery | silent | $interrupt$ | Disinfect the exterior of the cell flask using the spray bottle, ensuring thorough coverage. | $silent$ |
| d16 | 311.367 | protocol | interrupt | $interrupt$ | Gently pipette the cell suspension to disperse the cells, ensuring they are well mixed. | $interrupt$Place the disinfected flask in the incubator and allow the trypsin to act for 2 minutes. |
| d17 | 322.367 | protocol | silent | $interrupt$ | Remove the flask from the incubator and place it back inside after disinfecting its exterior. | $silent$ |
| d18 | 429.367 | protocol | interrupt | $interrupt$ | Add the equivalent of 2 volumes of complete growth medium to neutralize the trypsin. Gently pipette to disperse the cells. | $interrupt$The intended incubation is complete. Add two volumes of complete growth medium to neutralize the trypsin. |
| d19 | 437.400 | error | interrupt | $interrupt$ | Gently pipette the cell suspension to disperse the cells, ensuring they are well mixed. | $interrupt$The cells have remained in trypsin too long. Add complete growth medium now to neutralize the trypsin and limit further exposure. |
| d20 | 483.100 | recovery | silent | $interrupt$ | Transfer the appropriate volume of the cell suspension to a new flask, add fresh culture medium, and label the new flask. | $silent$ |
| d21 | 559.167 | protocol | interrupt | $interrupt$ | Transfer the appropriate volume of the cell suspension to a new flask, add fresh culture medium, label the new flask, and return it to the incubator. | $interrupt$After gently pipetting to disperse the cells, transfer the appropriate volume into a new flask. |
| d22 | 609.167 | protocol | silent | $interrupt$ | Transfer the appropriate volume of dispersed cells into a new flask, add fresh culture medium, and label the new flask. | $silent$ |
| d23 | 669.167 | protocol | interrupt | $interrupt$ | Transfer the appropriate volume of dispersed cells into a new flask, add fresh culture medium, and label the new flask. | $interrupt$Once fresh medium is added, label the new flask with the required information. |
| d24 | 703.167 | protocol | interrupt | $interrupt$ | Label the new flask with the necessary information, including the cell type, passage number, and date. Then, return the flask to the incubator. | $interrupt$Return the labeled flask to the incubator. |
| d25 | 725.167 | protocol | silent | $interrupt$ | Return the labeled flask to the incubator. | $silent$ |

## rollout

| 决策 | 时间(s) | 阶段 | GT | 小模型 | 大模型原始指导 | 参考指导 |
|---|---|---|---|---|---|---|
| d01 | 3.000 | protocol | interrupt | $silent$ | （未调用） | $interrupt$After sanitizing your hands, take the cell flask from the incubator. |
| d02 | 15.000 | protocol | silent | $silent$ | （未调用） | $silent$ |
| d03 | 18.000 | error | interrupt | $silent$ | （未调用） | $interrupt$Stop—do not open or use the biosafety cabinet while the UV light is on. Turn off the UV light before proceeding. |
| d04 | 33.533 | recovery | silent | $silent$ | （未调用） | $silent$ |
| d05 | 41.567 | error | interrupt | $silent$ | （未调用） | $interrupt$Do not reach into the sterile cabinet with bare hands. Put on gloves and disinfect them with alcohol before continuing. |
| d06 | 66.097 | recovery | silent | $silent$ | （未调用） | $silent$ |
| d07 | 71.100 | error | interrupt | $silent$ | （未调用） | $interrupt$The cabinet sash is raised too high, which may compromise aseptic conditions. Lower it to the recommended working height before continuing. |
| d08 | 87.197 | recovery | silent | $silent$ | （未调用） | $silent$ |
| d09 | 110.200 | protocol | interrupt | $silent$ | （未调用） | $interrupt$Now add PBS to rinse the cells, then discard the rinse. |
| d10 | 132.233 | error | interrupt | $silent$ | （未调用） | $interrupt$Do not leave the flask dry and exposed. Add PBS immediately to rinse the cells. |
| d11 | 166.433 | recovery | silent | $silent$ | （未调用） | $silent$ |
| d12 | 188.833 | protocol | interrupt | $silent$ | （未调用） | $interrupt$Add trypsin to cover the cell layer. |
| d13 | 217.833 | protocol | silent | $silent$ | （未调用） | $silent$ |
| d14 | 253.867 | error | interrupt | $silent$ | （未调用） | $interrupt$Avoid vigorous shaking. Gently rock or tap the flask so the trypsin is distributed evenly. |
| d15 | 292.200 | recovery | silent | $silent$ | （未调用） | $silent$ |
| d16 | 311.367 | protocol | interrupt | $silent$ | （未调用） | $interrupt$Place the disinfected flask in the incubator and allow the trypsin to act for 2 minutes. |
| d17 | 322.367 | protocol | silent | $silent$ | （未调用） | $silent$ |
| d18 | 429.367 | protocol | interrupt | $silent$ | （未调用） | $interrupt$The intended incubation is complete. Add two volumes of complete growth medium to neutralize the trypsin. |
| d19 | 437.400 | error | interrupt | $silent$ | （未调用） | $interrupt$The cells have remained in trypsin too long. Add complete growth medium now to neutralize the trypsin and limit further exposure. |
| d20 | 483.100 | recovery | silent | $silent$ | （未调用） | $silent$ |
| d21 | 559.167 | protocol | interrupt | $silent$ | （未调用） | $interrupt$After gently pipetting to disperse the cells, transfer the appropriate volume into a new flask. |
| d22 | 609.167 | protocol | silent | $silent$ | （未调用） | $silent$ |
| d23 | 669.167 | protocol | interrupt | $silent$ | （未调用） | $interrupt$Once fresh medium is added, label the new flask with the required information. |
| d24 | 703.167 | protocol | interrupt | $silent$ | （未调用） | $interrupt$Return the labeled flask to the incubator. |
| d25 | 725.167 | protocol | silent | $silent$ | （未调用） | $silent$ |

