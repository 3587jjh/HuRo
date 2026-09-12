CAPTION_SYSTEM_PROMPT = (
    "You caption egocentric hand-object interactions with per-hand attribution. "
    'Output strict JSON: {"think": "...", "left": "...", "right": "...", "bimanual": "..."}. '
    'Set a field to "n/a" when no coarse interaction event can be verified confidently. '
    "Describe only actions and object identities that are visibly supported by the shown frames. "
    "Prefer a supported caption over an uncertain one, whether shorter or longer. "
    "Use bimanual when a shared object or shared task is reasonably supported, including support-plus-manipulation or handoff. "
    "Never hallucinate."
)

MARKER_SPEC = (
    "Wrist trajectory overlays are drawn on top of the video:\n"
    "  - LEFT hand trajectory: GREEN line\n"
    "  - RIGHT hand trajectory: BLUE line\n"
    "  - Start position: small filled square\n"
    "  - Current position: small outlined circle with center dot\n"
    "  - Intermediate waypoints: small filled dots of gradually increasing size\n"
    "  - Time direction: square -> graduated dots -> circle\n"
    "  - Dashed line segments indicate brief gaps in tracking (interpolated, uncertain)\n"
    "  - A long tracking gap may split a trajectory into multiple disconnected segments\n\n"
    "These overlays are synthetic UI annotations, NOT physical objects in the scene. "
    "They are not stickers, wristbands, watches, bracelets, or any real object. "
    "You may mention overlays in think for reasoning, but never in left/right/bimanual. "
    "The green/blue overlay colors are not properties of the real objects. "
    "Do NOT infer an object's color or identity from overlay colors.\n\n"
)

CAPTION_PROMPT = (
    "I will provide a short egocentric video clip with wrist trajectory overlays.\n"
    "Each frame is labeled in the top-left corner for temporal reference.\n"
    "Frames are uniformly subsampled at a constant rate and are in time order.\n\n"
    + MARKER_SPEC +
    "Task: Identify hand-object interactions.\n"
    "Caption only visually verifiable, temporally distinguishable interaction events.\n"
    "Describe only discrete, visually verifiable manipulation events.\n"
    "Prefer concrete manipulation verbs that are directly supported by visible contact, displacement, or state change.\n"
    "Avoid generic or non-specific verbs when a more concrete manipulation verb is visibly supported.\n"
    "Each phrase should be usable as a standalone robotic action label, and multiple phrases may form a short temporally ordered sequence when multiple distinct events are clearly visible.\n"
    "Infer the verb from visible contact, displacement, or state change, not from scene context or object affordance.\n"
    "Describe only completed interaction events that are visibly supported in the shown frames.\n"
    "Do not invent a state change that is not visibly completed.\n\n"

    "Return JSON immediately (no preamble):\n"
    '1. "think": brief plain-sentence reasoning about observed motion per hand, max 60 words.\n'
    '2. "left": LEFT hand action(s), 1 to 3 phrases in time order.\n'
    '3. "right": RIGHT hand action(s), 1 to 3 phrases in time order.\n'
    '4. "bimanual": A hand-neutral description of the shared task, 1 to 3 phrases in time order. '
    "Use it when the two hands reasonably appear to contribute to the same object or the same task, "
    "including simultaneous cooperation, support-plus-manipulation, or sequential handoff across time. "
    "bimanual is not a concatenation of left and right. "
    "Describe only the shared task in hand-neutral language. "
    "If the hands clearly act on different objects or different tasks, set bimanual to \"n/a\". "
    "Do NOT mention left hand, right hand, both hands, or any hand reference in bimanual.\n\n"

    "Phrase format (apply to left, right, bimanual independently):\n"
    "   - Each phrase must be at least two words.\n"
    "   - Use 1 to 3 phrases in time order.\n"
    "   - Separate phrases with comma or 'and'. No trailing period.\n"
    "   - Use articles (the/a) before identifiable nouns.\n"
    "   - Use base verb form.\n"
    "   - 'it' is allowed only after the first phrase within the same field. "
    "The first phrase must never use 'it'.\n"
    "   - If more than 3 distinct completed events are visible, keep only the first 3 clearly visible completed events in time order.\n"
    "   - Do not split preparatory sub-steps within a single state change. Describe the completed manipulation event rather than preparatory motion alone.\n"
    "   - Do not compress clearly separable state changes into one generic summary. Each visually distinct state change should be described as a separate event.\n"
    "   - Prefer concrete manipulation verbs tied to visible evidence over generic control verbs.\n"
    "   - Do not use vague or non-specific object references when a more specific visible noun is available.\n"
    "   - Prefer a specific visible noun when possible. Avoid generic category nouns when a more specific visible noun is available.\n"
    "   - Do not use pronouns other than allowed 'it'.\n"
    "   - Include location (from/into/on + concrete noun) only when clearly visible.\n"
    "   - Use the same noun across fields for the same object.\n"
    "   - Never output words like overlay, marker, dot, trail, or circle.\n\n"

    "Do not underuse bimanual when a shared object or shared task is reasonably visible. "
    "If one hand mainly supports, positions, or stabilizes an object while the other hand manipulates that same object or advances that same task, "
    "bimanual should usually describe that shared task rather than default to \"n/a\".\n\n"

    "Anti-prediction rule:\n"
    "   - Do not predict the next likely action beyond the visible frames.\n"
    "   - Describe an event only if that event itself is visibly supported in the clip.\n"
    "   - If a hand reaches toward an object but the object is not clearly grasped or displaced, do not describe the grasp or lift.\n"
    "   - Do not name a target object unless that object is visually identifiable in the shown frames.\n"
    "   - If unsure whether an additional event is completed, leave it out.\n\n"

    "Anti-hallucination rule:\n"
    "   - Do not infer an action from scene context, object affordance, or likely next steps alone.\n"
    "   - Name an action only when the action itself is visually supported by visible contact, displacement, or state change.\n"
    "   - Do not infer a specific object category unless diagnostic visual features are visible in the shown frames.\n"
    "   - Prefer omission over a guessed action or guessed object identity.\n"
    "   - Trajectory overlays show wrist position only. They are not evidence of contact, grasp, displacement, object identity, or task completion.\n\n"

    'When to set "n/a":\n'
    '   - A side\'s action or object is unclear -> that side is "n/a".\n'
    "   - If the clip is dominated by continuous tool control or sustained motion without a clear event boundary, output all fields as \"n/a\".\n"
    '   - If the whole clip is uncaptionable, output all fields as "n/a".\n'
    "   - If any other person (not the camera wearer) is visible, output all fields as \"n/a\".\n\n"

    "Output a single JSON object. No code fences, no text outside the JSON.\n"
    "Use valid JSON (double quotes only, no trailing commas).\n"
)

VERIFY_SYSTEM_PROMPT = (
    "You verify egocentric hand-object interaction captions against video evidence. "
    "Judge visual grounding only. Do NOT judge whether a caption is weak or strong for robotic supervision; that is handled separately. "
    "Decide whether each shown field should be kept or rejected entirely as n/a. "
    "Do NOT rewrite, trim, or partially edit phrases inside a field. "
    "Valid JSON keys are only: think, global_na, reason, left_na, right_na, bimanual_na. "
    "If a field is not shown in the proposed caption, omit its *_na key. "
    'Set reason to "n/a" ONLY if all shown *_na flags are false. '
    'If global_na is true, reason MUST start with "global:" and name which rule matched. '
    "think: 1 sentence max (<=25 words). "
    "Never invent extra keys."
)

VERIFY_PROMPT_TEMPLATE = (
    "I will provide a short egocentric video clip with wrist trajectory overlays.\n"
    "Each frame is labeled in the top-left corner for temporal reference.\n"
    "Frames are uniformly subsampled at a constant rate and are in time order.\n\n"
    + MARKER_SPEC +
    "A caption was generated for this video. Only the non-n/a fields are shown.\n"
    "Verify them against the video evidence.\n"
    "Judge visual grounding only. Do not reject a field merely because it would be weak or generic for robotic supervision if it is still grounded in the video.\n\n"
    "Proposed caption (non-n/a fields only):\n"
    "{caption_json}\n\n"
    "Verification is two steps. Do Step 1 first; if global_na=true, skip Step 2.\n\n"

    "STEP 1 - Global rejection (set global_na=true and ALL shown *_na=true):\n"
    "  Apply global_na=true if ANY of these is true:\n"
    "  1. Walking - the clip is dominated by locomotion, with broad scene motion and step-like camera bounce across most frames. "
    "Object contact during walking does not by itself make the clip captionable.\n"
    "  2. Too dark or unclear - hands or objects are not clearly discernible.\n"
    "  3. The clip is dominated by continuous operation or continuous control rather than discrete interaction events.\n"
    "  3a. Exception: do NOT apply the continuous-control rule when the clip shows multiple distinct object interactions "
    "or multiple visible state changes over time, even if the hands remain active throughout. "
    "Sequential interactions involving different objects or locations are discrete events, not continuous control.\n"
    "  3b. State-change sequences on the same object are also discrete events, not continuous control. "
    "If successive actions produce clearly verifiable changes in state or configuration, "
    "treat them as separate events rather than one continuous operation. "
    "Do NOT apply the continuous-control rule to such sequences.\n"
    "  4. Reject as continuous control when the motion is better explained as ongoing guidance, support, adjustment, pushing, pulling, "
    "or maintaining contact during one uninterrupted operation, without a clear event boundary or visibly verifiable state change.\n"
    "  5. Minor directional adjustments, repeated oscillatory motion, or sustained contact during continuous operation "
    "do NOT by themselves create separate captionable events.\n"
    "  6. Apply global_na=true only when the clip is better explained as one uninterrupted ongoing operation "
    "on the same object or tool, without clearly verifiable intermediate state changes for most frames.\n"
    "  7. When the overall clip is continuous control, do NOT keep a field merely because one hand remains in contact, "
    "provides support, guides motion, or makes minor adjustments during that continuous operation.\n"
    "  8. Another person or human-like figure (not the camera wearer) is visible anywhere in the clip. "
    "This includes any body part or human-shaped form, even partially visible or in the background. "
    "The camera wearer's own hands and forearms are expected and do NOT trigger this rule. "
    "If you see additional hands, arms, or body that cannot belong to the camera wearer, reject. "
    "If unsure, reject conservatively.\n"
    "  9. Self-contact: the camera wearer is manipulating their own body, or clothing/accessories currently being worn. "
    "The target of the hand action is the wearer's own person or worn items, not an external object. "
    "Objects held in the hand or placed on a surface are NOT self-contact. "
    "If unsure, reject conservatively.\n\n"

    "STEP 2 - Per-field rejection (only if global_na=false):\n"
    "  Reject a shown field when the caption is contradicted by the video OR when the described action or object lacks direct visual support.\n"
    "  If the interaction is subtle or partially occluded but clearly present, keep the field.\n"
    "  Do not reject a field merely because the object category is broad or the contact is passive.\n"
    "  DO reject a field when the named action or object is more specific than what is visually verifiable.\n"
    "  Do not treat scene context, typical object use, or likely next steps as direct visual support.\n"
    "  Do not reject a field merely because it contains multiple phrases. "
    "Keep the field when the overall coarse interaction sequence is clearly supported by distinct visible evidence.\n\n"

    "  Per-hand (left_na / right_na):\n"
    "  9. Caption-video inconsistency - wrong action, wrong object, wrong ordering, or described action is not observed.\n"
    "  9a. Future-action prediction - reject if a caption describes an event that is not clearly supported in the shown frames.\n"
    "  9b. Object over-specification - reject if the caption names a more specific object category "
    "than what is clearly identifiable in the shown frames. "
    "A plausible guess from scene context is not sufficient.\n"
    "  9c. Unsupported contact - reject if the caption describes contact, grasp, or displacement "
    "that is not directly visible in the shown frames, even if the action is plausible given the scene.\n\n"

    "  Bimanual (bimanual_na):\n"
    "  10. Reject bimanual only when the two hands clearly contribute to different objects or different tasks. "
    "Keep bimanual when a shared object or shared task is reasonably supported, including simultaneous cooperation, "
    "support-plus-manipulation, stabilization-plus-manipulation, or sequential handoff across time. "
    "Do not reject bimanual merely because one hand's role is mainly support, positioning, or stabilization "
    "while the other hand performs the more visible manipulation.\n\n"

    "Output rules:\n"
    "  - The *_na flags are the decision. reason is only a brief explanation consistent with those flags.\n"
    "  - You MUST output one *_na flag (true or false) for EVERY field shown above. Do NOT omit any shown field's flag.\n"
    "  - If global_na=true, every shown *_na flag MUST be true.\n"
    "  - Set a field's *_na to true when the caption is contradicted by the video or lacks direct visual support.\n"
    '  - Set reason to "n/a" ONLY if all shown *_na flags are false.\n'
    '  - If global_na=true, reason MUST start with "global:" followed by the rule tag: '
    "other_person, self_contact, walking, dark, or continuous_tool. "
    'Example: "global:other_person", "global:self_contact".\n'
    "  - Keep think to 1 sentence (<=25 words).\n\n"

    "Output JSON (no code fences, no text outside):\n"
    '{{"think": "brief reasoning", "global_na": true/false, '
    '"<shown_field>_na": true/false (for each shown field), '
    '"reason": "n/a or explanation"}}\n'
)

LABEL_VERIFY_SYSTEM_PROMPT = (
    "You judge whether action labels are strong enough for robotic manipulation pretraining. "
    "Judge label quality only. Do NOT judge whether the video evidence is sufficient; visual grounding is handled separately. "
    "Decide whether each shown field should be kept or dropped entirely. "
    "Do NOT rewrite, trim, or partially edit phrases inside a field. "
    "Valid JSON keys are only: think, reason, left_drop, right_drop, bimanual_drop. "
    "If a field is not shown, omit its *_drop key. "
    'Set reason to "n/a" ONLY if all shown *_drop flags are false. '
    "When any field is dropped, use one short reason label such as support_only, generic_summary, continuous_tool, vague_object, weak_verb, cross_hand, or prediction. "
    "think: 1 sentence max (<=15 words). "
    "Never invent extra keys."
)

LABEL_VERIFY_PROMPT_TEMPLATE = (
    "Evaluate each field for robotic manipulation pretraining quality.\n"
    "Judge label quality only. Do NOT judge whether the video evidence is sufficient; that is handled separately.\n\n"

    "DROP a field if any of the following apply:\n"
    "1. It describes support-only or hold-only behavior, with no clear discrete manipulation event or state change.\n"
    "2. It is a generic task summary rather than a specific manipulation event.\n"
    "3. It describes continuous operation or continuous control rather than a discrete event.\n"
    "   Exception: a field describing 2-3 temporally ordered discrete manipulation events is NOT continuous,\n"
    "   even on the same object. State changes count as discrete events.\n"
    "4. It refers to an object too vaguely to supervise a manipulation event.\n"
    "5. It uses wording that is too weak or generic to define a concrete manipulation event.\n"
    "6. The field text explicitly refers to the other hand or includes the other hand's action.\n"
    "7. It is predictive, speculative, or otherwise unsuitable as a standalone robotic manipulation label.\n\n"

    "KEEP a field if it describes a discrete completed manipulation event with an object concrete enough for robotic supervision.\n"
    "Do not drop a field merely because the object category is broad if it is still concrete enough to supervise a manipulation event.\n"
    "Drop a field when the wording is more generic than needed for robotic supervision, even if it is plausible.\n\n"

    "IMPORTANT coverage rules:\n"
    "- Do NOT drop a field merely because it contains 2-3 temporally ordered actions.\n"
    "  Keep multi-action fields when each action is a concrete, discrete manipulation event.\n"
    "- For bimanual, keep when a shared object or shared task is clearly expressed,\n"
    "  even if one hand mainly supports while the other manipulates.\n"
    "- Do NOT prefer shorter labels over richer but equally concrete labels.\n\n"

    "Candidate (non-n/a fields only):\n"
    "{caption_json}\n\n"

    "Output rules:\n"
    "  - The *_drop flags are the decision.\n"
    "  - You MUST output one *_drop flag (true or false) for EVERY field shown above. Do NOT omit any shown field's flag.\n"
    '  - Set reason to "n/a" ONLY if all shown *_drop flags are false.\n'
    "  - If any field is dropped, reason should be one short label such as support_only, generic_summary, continuous_tool, vague_object, weak_verb, cross_hand, or prediction.\n"
    "  - Keep think to 1 sentence (<=15 words).\n\n"

    "Output JSON (no code fences, no text outside):\n"
    '{{"think": "brief reasoning", '
    '"<shown_field>_drop": true/false (for each shown field), '
    '"reason": "n/a or one short reason label"}}\n'
)