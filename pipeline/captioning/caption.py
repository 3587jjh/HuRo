import warnings
warnings.filterwarnings('ignore')

import os.path as osp
import glob
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

# Qwen3.5 weights auto-download into the HF cache ($HF_HOME / $HF_HUB_CACHE) and are reused from there.
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
VLM_CACHE_DIR = HF_HUB_CACHE

from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

import re as _re
import json as _json
from .prompts import (CAPTION_SYSTEM_PROMPT, CAPTION_PROMPT,
                      VERIFY_SYSTEM_PROMPT, VERIFY_PROMPT_TEMPLATE,
                      LABEL_VERIFY_SYSTEM_PROMPT, LABEL_VERIFY_PROMPT_TEMPLATE)
from .validate import (extract_from_json_perhand, normalize_and_validate_perhand,
                       split_actions_canonical, parse_verification_output,
                       parse_label_verify_output,
                       classify_verification_reason, normalize_punct,
                       postcheck_assembled_triplet)
from .tracker import RejectionTracker

MODEL_ID = "Qwen/Qwen3.5-9B"

# Consensus thresholds: per-field minimum agreement for survival
_CONSENSUS_THRESHOLDS = {
    "left":      {"min_agree": 2, "min_ratio": 0.4},
    "right":     {"min_agree": 2, "min_ratio": 0.4},
    "bimanual":  {"min_agree": 1, "min_ratio": 0.25},  # bimanual is naturally sparser
}


def _dbg(msg: str, window_log: Optional[dict]):
    """Append msg to the window_log trace."""
    if window_log is not None:
        window_log.setdefault("prints", []).append(msg)

_vlm_model = None
_vlm_processor = None
_think_end_token_id = None


def _get_think_end_token_id():
    global _think_end_token_id
    if _think_end_token_id is None:
        _think_end_token_id = _vlm_processor.tokenizer.convert_tokens_to_ids("</think>")
    return _think_end_token_id


def _extract_response(gen_ids):
    """Extract response text after </think> from GENERATED-ONLY token IDs.
    If no </think> found, decodes everything (no thinking happened)."""
    id_list = gen_ids.tolist() if hasattr(gen_ids, 'tolist') else gen_ids
    end_id = _get_think_end_token_id()
    if end_id is None or end_id < 0:
        return _vlm_processor.decode(id_list, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
    try:
        idx = len(id_list) - id_list[::-1].index(end_id)
        response_ids = id_list[idx:]
    except ValueError:
        response_ids = id_list
    return _vlm_processor.decode(response_ids, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)


def _snapshot_has_weights(snap: str) -> bool:
    """Whether a snapshot holds every weight shard its index names."""
    index = osp.join(snap, "model.safetensors.index.json")
    if not osp.isfile(index):
        return osp.isfile(osp.join(snap, "model.safetensors"))  # single-shard repo
    try:
        with open(index) as f:
            shards = set(_json.load(f)["weight_map"].values())
    except (OSError, ValueError, KeyError):
        return False
    return bool(shards) and all(osp.isfile(osp.join(snap, name)) for name in shards)


def _pick_snapshot_if_exists(vlm_cache_dir: str) -> str | None:
    repo_dir = osp.join(vlm_cache_dir, f"models--{MODEL_ID.replace('/', '--')}")
    candidates = []
    ref_main = osp.join(repo_dir, "refs", "main")
    if osp.exists(ref_main):
        rev = open(ref_main).read().strip()
        candidates.append(osp.join(repo_dir, "snapshots", rev))
    candidates.extend(sorted(glob.glob(osp.join(repo_dir, "snapshots", "*")), reverse=True))
    for snap in candidates:
        if osp.isdir(snap) and _snapshot_has_weights(snap):
            return snap
    return None


def load_vlm_model(cache_dir: Optional[str] = None):
    """Load Qwen3.5 model and processor. Call once at startup."""
    global _vlm_model, _vlm_processor
    if _vlm_model is not None:
        return _vlm_model, _vlm_processor

    cd = cache_dir or VLM_CACHE_DIR
    snap = _pick_snapshot_if_exists(cd)
    if snap is None:
        snap = snapshot_download(repo_id=MODEL_ID, cache_dir=cd)

    kwargs = dict(
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    _vlm_model = Qwen3_5ForConditionalGeneration.from_pretrained(snap, **kwargs)
    _vlm_processor = AutoProcessor.from_pretrained(snap, local_files_only=True)
    return _vlm_model, _vlm_processor


def _build_vlm_inputs(frames_uint8: List[np.ndarray], user_text: str, system_prompt: str,
                      enable_thinking: bool = True,
                      force_json_prefix: bool = False,
                      fps: Optional[float] = None):
    """Build VLM inputs: frames as native video.
    Frames are already subsampled externally, so no fps-based resampling is needed.
    `fps` is the rate of the given frames and sets their '<t seconds>' timestamps."""
    processor = _vlm_processor
    assert _vlm_model is not None and processor is not None, "Call load_vlm_model() first"

    video_object = np.stack(frames_uint8, axis=0)  # (T, H, W, C) uint8
    # The '<t seconds>' timestamps come from video_metadata. Without it the processor assumes 24 fps
    video_kwargs = {} if fps is None else {"video_metadata": [
        {"fps": fps, "total_num_frames": len(video_object), "frames_indices": list(range(len(video_object)))}]}

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_object},
            {"type": "text", "text": user_text},
        ]},
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        do_sample_frames=False,  # frames already subsampled externally
        **video_kwargs,
    )

    # Force JSON start: append '{"think":"' tokens so the model continues from there
    if force_json_prefix:
        prefix_ids = processor.tokenizer.encode('{"think":"', add_special_tokens=False)
        prefix_t = torch.tensor([prefix_ids], dtype=inputs["input_ids"].dtype)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix_t], dim=1)
        inputs["attention_mask"] = torch.cat([
            inputs["attention_mask"],
            torch.ones((1, len(prefix_ids)), dtype=inputs["attention_mask"].dtype),
        ], dim=1)
        if "mm_token_type_ids" in inputs:
            inputs["mm_token_type_ids"] = torch.cat([
                inputs["mm_token_type_ids"],
                torch.zeros((1, len(prefix_ids)), dtype=inputs["mm_token_type_ids"].dtype),
            ], dim=1)

    # Workaround: processor emits per-temporal-patch vision blocks but a single video_grid_thw entry.
    # Expand [[T, H, W]] → T rows of [[1, H, W]] to match.
    if "video_grid_thw" in inputs:
        vg = inputs["video_grid_thw"]  # (num_videos, 3)
        expanded = []
        for i in range(vg.shape[0]):
            t, h, w = vg[i].tolist()
            if t > 1:
                expanded.append(torch.tensor([[1, h, w]] * t, dtype=vg.dtype))
            else:
                expanded.append(vg[i:i+1])
        inputs["video_grid_thw"] = torch.cat(expanded, dim=0)

    inputs = inputs.to(_vlm_model.device)

    return inputs


def _build_text_only_inputs(user_text: str, system_prompt: str,
                            force_json_prefix: bool = False):
    """Build VLM inputs for text-only verification (no video)."""
    processor = _vlm_processor
    assert _vlm_model is not None and processor is not None, "Call load_vlm_model() first"

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )

    if force_json_prefix:
        prefix_ids = processor.tokenizer.encode('{"think":"', add_special_tokens=False)
        prefix_t = torch.tensor([prefix_ids], dtype=inputs["input_ids"].dtype)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix_t], dim=1)
        inputs["attention_mask"] = torch.cat([
            inputs["attention_mask"],
            torch.ones((1, len(prefix_ids)), dtype=inputs["attention_mask"].dtype),
        ], dim=1)

    inputs = inputs.to(_vlm_model.device)
    return inputs


def _sample_captions(frames_uint8: List[np.ndarray], num_return_sequences: int,
                     temperature: float, top_p: float, max_new_tokens: int = 256,
                     do_sample: bool = True,
                     clip_id: str = "", window: str = "",
                     window_log: Optional[dict] = None,
                     fps: Optional[float] = None,
                     ) -> List[Tuple[Dict[str, str], str, Optional[str]]]:
    """Sample per-hand captions from the VLM.
    Returns list of (narr_dict, think, reject_reason). narr_dict keys: think, left, right, bimanual."""
    inputs = _build_vlm_inputs(frames_uint8, CAPTION_PROMPT, CAPTION_SYSTEM_PROMPT,
                               enable_thinking=False, force_json_prefix=True, fps=fps)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p,
                          repetition_penalty=1.0)
    else:
        gen_kwargs.update(do_sample=False)

    input_len = inputs.input_ids.shape[1]
    with torch.inference_mode():
        ids = _vlm_model.generate(**gen_kwargs)

    outs = [_extract_response(ids[i, input_len:]) for i in range(ids.shape[0])]

    results = []
    for i, raw in enumerate(outs):
        # Strip code fence wrappers (```json ... ```) before prefix check
        raw_s = raw.strip()
        if raw_s.startswith('```'):
            first_nl = raw_s.find('\n')
            if first_nl >= 0:
                raw_s = raw_s[first_nl + 1:]
            if raw_s.rstrip().endswith('```'):
                raw_s = raw_s.rstrip()[:-3].rstrip()
            raw = raw_s
        # Restore JSON prefix if force_json_prefix consumed the leading '{'
        if not raw.lstrip().startswith('{'):
            raw = '{"think":"' + raw
        _dbg(f"  [caption] raw[{i}]: {repr(raw)}", window_log)

        _raw_trunc = raw[:2000]
        sample_entry = {"index": i, "raw_output": _raw_trunc,
                        "raw_truncated": len(raw) > 2000}

        # Per-hand JSON extraction
        narr_dict, think, json_reason = extract_from_json_perhand(raw)
        if json_reason is not None:
            # Separate content outcome (all fields n/a) from parse failures
            if json_reason == "all_na":
                reason_label = "content_all_na"
                _dbg(f"  [caption] content outcome: all fields n/a", window_log)
            else:
                reason_label = json_reason
                think = ""
                _dbg(f"  [caption] JSON fail: {json_reason}", window_log)
            sample_entry["json_reason"] = json_reason
            sample_entry["validation"] = None
            if window_log is not None:
                window_log["samples"].append(sample_entry)
            results.append((narr_dict, think, reason_label))
            continue

        sample_entry["think"] = think[:200] if think else ""

        # Pre-validation phrase counts (before normalize_and_validate_perhand)
        pre_val_phrases = {}
        pre_val_counts = {}
        for f in ("left", "right", "bimanual"):
            text = narr_dict.get(f, "n/a")
            if not isinstance(text, str) or not text.strip() or text.strip().lower() in ("n/a", "na", "none"):
                pre_val_phrases[f] = []
                pre_val_counts[f] = 0
                continue
            t = text.strip().splitlines()[0].strip()
            t = t.strip("\"'\u201c\u201d\u2018\u2019")
            if t.endswith("."):
                t = t[:-1].strip()
            if t and t[0].isalpha():
                t = t[0].lower() + t[1:]
            t = normalize_punct(t)
            # Try split with truncate=True to capture 4+ as 3 (not parse error)
            phrases = split_actions_canonical(t, max_phrases=3, truncate=True)
            if phrases is not None:
                pre_val_phrases[f] = list(phrases)
                pre_val_counts[f] = len(phrases)
                raw_phrases = split_actions_canonical(t, max_phrases=99, truncate=False)
                if raw_phrases is not None and len(raw_phrases) > 3:
                    pre_val_counts[f] = len(raw_phrases)  # true generated count
            else:
                pre_val_phrases[f] = []
                pre_val_counts[f] = -1
        sample_entry["pre_validation_phrases"] = pre_val_phrases
        sample_entry["pre_validation_action_counts"] = pre_val_counts

        # Normalize and validate per-hand (per-phrase dropping)
        narr_dict, reason, drops = normalize_and_validate_perhand(
            narr_dict, return_drops=True)
        drop_str = ""
        for fld, fld_drops in drops.items():
            if fld_drops:
                drop_str += f" {fld}:{[(p, r) for _, p, r in fld_drops]}"
        _dbg(f"  [caption] validated: L={narr_dict['left']!r} R={narr_dict['right']!r} "
              f"B={narr_dict['bimanual']!r} reason={reason}"
              + (f" | dropped:{drop_str}" if drop_str else ""), window_log)

        sample_entry["json_reason"] = None
        sample_entry["validation"] = {
            "left": narr_dict["left"], "right": narr_dict["right"],
            "bimanual": narr_dict["bimanual"],
            "reason": reason,
            "drops": {k: [(idx, p, r) for idx, p, r in v] for k, v in drops.items() if v},
        }

        if reason is not None:
            _dbg(f"  [caption] REJECT {reason}", window_log)
            if window_log is not None:
                window_log["samples"].append(sample_entry)
            results.append((narr_dict, think, reason))
            continue

        _dbg(f"  [caption] PASS: L={narr_dict['left']!r} R={narr_dict['right']!r} "
              f"B={narr_dict['bimanual']!r}", window_log)
        if window_log is not None:
            window_log["samples"].append(sample_entry)
        results.append((narr_dict, think, None))

    return results


def _batch_generate_text_only(all_inputs, max_new_tokens=256):
    """Batched greedy decoding for text-only inputs (no visual tensors).
    all_inputs: list of batch-size-1 processor dicts. Returns one generated-token tensor per input."""
    n = len(all_inputs)
    if n == 0:
        return []
    if n == 1:
        gen_kwargs = dict(**all_inputs[0], max_new_tokens=max_new_tokens,
                          num_return_sequences=1, do_sample=False)
        with torch.inference_mode():
            ids = _vlm_model.generate(**gen_kwargs)
        input_len = all_inputs[0]["input_ids"].shape[1]
        return [ids[0, input_len:]]

    pad_token_id = _vlm_processor.tokenizer.pad_token_id or 0
    input_ids_list = [inp["input_ids"][0] for inp in all_inputs]
    input_lens = [ids.shape[0] for ids in input_ids_list]
    max_len = max(input_lens)

    padded_ids = []
    attn_masks = []
    for ids in input_ids_list:
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            padded_ids.append(torch.cat([
                torch.full((pad_len,), pad_token_id, dtype=ids.dtype, device=ids.device),
                ids]))
            attn_masks.append(torch.cat([
                torch.zeros(pad_len, dtype=torch.long, device=ids.device),
                torch.ones(ids.shape[0], dtype=torch.long, device=ids.device)]))
        else:
            padded_ids.append(ids)
            attn_masks.append(torch.ones(ids.shape[0], dtype=torch.long, device=ids.device))

    gen_inputs = {
        "input_ids": torch.stack(padded_ids, dim=0),
        "attention_mask": torch.stack(attn_masks, dim=0),
    }
    gen_kwargs = dict(**gen_inputs, max_new_tokens=max_new_tokens,
                      num_return_sequences=1, do_sample=False)
    with torch.inference_mode():
        ids = _vlm_model.generate(**gen_kwargs)

    results = []
    padded_input_len = max_len
    for i in range(n):
        results.append(ids[i, padded_input_len:])
    return results


def _text_verify_candidates(
    candidates: List[Dict[str, str]],
    window_log: Optional[dict] = None,
    source_sample_indices: Optional[List[int]] = None,
    batch_size: int = 6,
) -> List[Tuple[Dict[str, str], Optional[str]]]:
    """Text-only label-strength gate (batched, no video).
    Returns list of (patched_narr_dict, reject_reason_or_None). Dropped fields are set to n/a."""

    results = [None] * len(candidates)

    candidate_meta = []  # (ci, narr_dict, caption_obj, shown_fields)
    all_inputs = []

    for ci, narr_dict in enumerate(candidates):
        caption_obj = {k: v for k, v in narr_dict.items()
                       if k in ("left", "right", "bimanual") and v != "n/a"}
        if not caption_obj:
            results[ci] = (narr_dict, None)
            continue
        shown_fields = tuple(caption_obj.keys())
        user_text = LABEL_VERIFY_PROMPT_TEMPLATE.format(caption_json=_json.dumps(caption_obj))
        inputs = _build_text_only_inputs(user_text, LABEL_VERIFY_SYSTEM_PROMPT,
                                          force_json_prefix=True)
        candidate_meta.append((ci, narr_dict, caption_obj, shown_fields))
        all_inputs.append(inputs)

    if not all_inputs:
        return results

    for chunk_start in range(0, len(all_inputs), batch_size):
        chunk_inputs = all_inputs[chunk_start:chunk_start + batch_size]
        chunk_meta = candidate_meta[chunk_start:chunk_start + batch_size]

        gen_results = _batch_generate_text_only(chunk_inputs, max_new_tokens=256)

        for batch_idx, (ci, narr_dict, caption_obj, shown_fields) in enumerate(chunk_meta):
            raw = _extract_response(gen_results[batch_idx])

            raw_s = raw.strip()
            if raw_s.startswith('```'):
                first_nl = raw_s.find('\n')
                if first_nl >= 0:
                    raw_s = raw_s[first_nl + 1:]
                if raw_s.rstrip().endswith('```'):
                    raw_s = raw_s.rstrip()[:-3].rstrip()
                raw = raw_s
            if not raw.lstrip().startswith('{'):
                raw = '{"think":"' + raw

            _dbg(f"  [text_verify] candidate[{ci}] input: {_json.dumps(caption_obj)}", window_log)
            _dbg(f"  [text_verify] candidate[{ci}] response: {repr(raw[:500])}", window_log)

            drop_flags, reason = parse_label_verify_output(raw, shown_fields=shown_fields)

            if reason == "parse_error":
                _dbg(f"  [text_verify] candidate[{ci}] parse_error (fail-open, keeping)", window_log)
                if window_log is not None:
                    _src_idx = source_sample_indices[ci] if source_sample_indices is not None else None
                    window_log.setdefault("text_verification", []).append({
                        "candidate_index": ci,
                        "source_sample_index": _src_idx,
                        "input_caption": dict(caption_obj),
                        "raw_output": raw[:2000],
                        "verdict": "PASS_FAIL_OPEN",
                        "reason": "parse_error",
                        "reason_code": "text_gate_parse_error",
                    })
                results[ci] = (narr_dict, None)
                continue

            reject_code = _classify_text_gate_reason(reason)

            patched = dict(narr_dict)
            changes = []
            for field in ("left", "right", "bimanual"):
                flag_key = f"{field}_drop"
                if drop_flags.get(flag_key, False) and patched[field] != "n/a":
                    changes.append(field)
                    patched[field] = "n/a"

            # Inconsistency: model gave a reason but set no drop flags
            reason_norm = (reason or "").strip().lower()
            inconsistent = (reason_norm not in ("n/a", "") and not changes
                            and not any(drop_flags.get(f"{f}_drop", False) for f in shown_fields))

            if window_log is not None:
                _src_idx = source_sample_indices[ci] if source_sample_indices is not None else None
                entry = {
                    "candidate_index": ci,
                    "source_sample_index": _src_idx,
                    "input_caption": dict(caption_obj),
                    "raw_output": raw[:2000],
                    "drop_flags": drop_flags,
                    "reason": reason,
                    "reason_code": reject_code,
                    "changes": changes,
                }
                if inconsistent:
                    entry["inconsistent_reason_only"] = True
                window_log.setdefault("text_verification", []).append(entry)

            if all(patched[f] == "n/a" for f in ("left", "right", "bimanual")):
                _dbg(f"  [text_verify] candidate[{ci}] REJECT ({reject_code})", window_log)
                results[ci] = (patched, reject_code)
            else:
                chg = f" changed={changes}" if changes else ""
                _dbg(f"  [text_verify] candidate[{ci}] PASS{chg}", window_log)
                results[ci] = (patched, None)

    return results


# Structured reason codes for text gate
_TEXT_GATE_REASON_CODES = {
    "tool": "continuous_tool",
    "vehicle": "continuous_tool",
    "support": "support_only",
    "hold": "support_only",
    "steady": "support_only",
    "generic": "generic_summary",
    "summary": "generic_summary",
    "vague": "vague_object",
    "unidentif": "vague_object",
    "cross": "cross_hand",
    "other hand": "cross_hand",
    "predict": "prediction",
    "weak": "weak_verb",
}


def _classify_text_gate_reason(reason: str) -> str:
    """Map text gate reason to structured code."""
    r = (reason or "").lower().strip()
    if r in ("n/a", ""):
        return "text_gate_other"
    for keyword, code in _TEXT_GATE_REASON_CODES.items():
        if keyword in r:
            return f"text_gate_{code}"
    return "text_gate_other"


def _verify_candidates(
    frames_uint8: List[np.ndarray],
    candidates: List[Dict[str, str]],
    window_log: Optional[dict] = None,
    source_sample_indices: Optional[List[int]] = None,
    fps: Optional[float] = None,
    batch_size: int = 3,
) -> List[Tuple[Dict[str, str], Optional[str]]]:
    """VLM visual verification gate (batched). All verification rejects are hard.
    Returns list of (patched_narr_dict, reject_reason_or_None).
    The reason is set only if ALL fields are killed."""

    results = [None] * len(candidates)

    candidate_meta = []  # (ci, narr_dict, caption_obj, shown_fields)
    all_inputs = []

    for ci, narr_dict in enumerate(candidates):
        caption_obj = {k: v for k, v in narr_dict.items()
                       if k in ("left", "right", "bimanual") and v != "n/a"}
        if not caption_obj:
            if window_log is not None:
                _src_skip = source_sample_indices[ci] if source_sample_indices is not None else None
                window_log["verification"].append({
                    "candidate_index": ci, "source_sample_index": _src_skip,
                    "verdict": "SKIP_ALL_NA",
                })
            results[ci] = (narr_dict, None)
            continue

        shown_fields = tuple(caption_obj.keys())
        user_text = VERIFY_PROMPT_TEMPLATE.format(caption_json=_json.dumps(caption_obj))
        inputs = _build_vlm_inputs(frames_uint8, user_text, VERIFY_SYSTEM_PROMPT,
                                    enable_thinking=False, force_json_prefix=True, fps=fps)
        candidate_meta.append((ci, narr_dict, caption_obj, shown_fields))
        all_inputs.append(inputs)

    if not all_inputs:
        return results

    for chunk_start in range(0, len(all_inputs), batch_size):
        chunk_end = min(chunk_start + batch_size, len(all_inputs))
        chunk_inputs = all_inputs[chunk_start:chunk_end]
        chunk_meta = candidate_meta[chunk_start:chunk_end]
        n = len(chunk_inputs)

        if n == 1:
            gen_inputs = chunk_inputs[0]
            input_lens = [chunk_inputs[0]["input_ids"].shape[1]]
        else:
            pad_token_id = _vlm_processor.tokenizer.pad_token_id or 0
            input_ids_list = [inp["input_ids"][0] for inp in chunk_inputs]
            input_lens = [ids.shape[0] for ids in input_ids_list]
            max_len = max(input_lens)

            padded_ids = []
            attn_masks = []
            for ids in input_ids_list:
                pad_len = max_len - ids.shape[0]
                if pad_len > 0:
                    padded_ids.append(torch.cat([
                        torch.full((pad_len,), pad_token_id, dtype=ids.dtype, device=ids.device),
                        ids]))
                    attn_masks.append(torch.cat([
                        torch.zeros(pad_len, dtype=torch.long, device=ids.device),
                        torch.ones(ids.shape[0], dtype=torch.long, device=ids.device)]))
                else:
                    padded_ids.append(ids)
                    attn_masks.append(torch.ones(ids.shape[0], dtype=torch.long, device=ids.device))

            visual_keys = {"pixel_values", "pixel_values_videos", "image_grid_thw",
                           "video_grid_thw"}
            gen_inputs = {
                "input_ids": torch.stack(padded_ids, dim=0),
                "attention_mask": torch.stack(attn_masks, dim=0),
            }
            if "mm_token_type_ids" in chunk_inputs[0]:
                mm_list = []
                for inp in chunk_inputs:
                    mm = inp["mm_token_type_ids"][0]
                    pad_len = max_len - mm.shape[0]
                    if pad_len > 0:
                        mm_list.append(torch.cat([
                            torch.zeros(pad_len, dtype=mm.dtype, device=mm.device), mm]))
                    else:
                        mm_list.append(mm)
                gen_inputs["mm_token_type_ids"] = torch.stack(mm_list, dim=0)

            ref = chunk_inputs[0]
            for key in ref:
                if key in ("input_ids", "attention_mask", "mm_token_type_ids"):
                    continue
                val = ref[key]
                if not isinstance(val, torch.Tensor):
                    gen_inputs[key] = val
                elif key in visual_keys:
                    gen_inputs[key] = val.repeat(n, *([1] * (val.dim() - 1)))
                else:
                    gen_inputs[key] = val.repeat_interleave(n, dim=0)

        gen_kwargs = dict(**gen_inputs, max_new_tokens=256, num_return_sequences=1,
                          do_sample=False)
        with torch.inference_mode():
            ids = _vlm_model.generate(**gen_kwargs)

        padded_input_len = max(input_lens) if n > 1 else input_lens[0]
        for batch_idx, (ci, narr_dict, caption_obj, shown_fields) in enumerate(chunk_meta):
            raw = _extract_response(ids[batch_idx, padded_input_len:])

            raw_s = raw.strip()
            if raw_s.startswith('```'):
                first_nl = raw_s.find('\n')
                if first_nl >= 0:
                    raw_s = raw_s[first_nl + 1:]
                if raw_s.rstrip().endswith('```'):
                    raw_s = raw_s.rstrip()[:-3].rstrip()
                raw = raw_s
            if not raw.lstrip().startswith('{'):
                raw = '{"think":"' + raw

            _dbg(f"  [verify] candidate[{ci}] input: {_json.dumps(caption_obj)}", window_log)
            _shown = raw[:500]
            _suffix = "…(truncated)" if len(raw) > 500 else ""
            _dbg(f"  [verify] candidate[{ci}] response: {repr(_shown)}{_suffix}", window_log)

            na_flags, reason = parse_verification_output(raw, shown_fields=shown_fields)
            _src_idx = source_sample_indices[ci] if source_sample_indices is not None else None

            if reason == "parse_error":
                _dbg(f"  [verify] candidate[{ci}] parse_error (fail-open, keeping)", window_log)

            global_na_flag = na_flags.get("global_na", False)
            reject_reason = classify_verification_reason(reason or "n/a", global_na=global_na_flag)

            reason_norm = (reason or "").strip().lower()
            has_any_field_na = any(
                na_flags.get(f"{f}_na", False) for f in ("left", "right", "bimanual"))
            if reason_norm == "n/a" and not global_na_flag and has_any_field_na:
                _dbg(f"  [verify] candidate[{ci}] contradictory output → trusting explicit flags",
                      window_log)
                reject_reason = "vlm_verify_contradiction"

            verify_entry = {
                "candidate_index": ci,
                "source_sample_index": _src_idx,
                "input_caption": dict(caption_obj),
                "raw_output": raw[:2000],
                "raw_truncated": len(raw) > 2000,
                "na_flags": na_flags,
                "parse_reason": reason,
                "classified_reason": reject_reason,
            }

            has_explicit_field_na = any(
                na_flags.get(f"{f}_na", False) for f in ("left", "right", "bimanual"))

            # Clean PASS / fail-open
            if not has_explicit_field_na and reason_norm in ("n/a", "parse_error"):
                _pass_type = "clean" if reason_norm == "n/a" else "fail-open"
                _dbg(f"  [verify] candidate[{ci}] result: PASS ({_pass_type})", window_log)
                verify_entry["verdict"] = f"PASS_{_pass_type.upper().replace('-', '_')}"
                verify_entry["patched"] = None
                verify_entry["changes"] = []
                if window_log is not None:
                    window_log["verification"].append(verify_entry)
                results[ci] = (narr_dict, None)
                continue

            # Inconsistency: reason given but no na flags set
            if not has_explicit_field_na and not global_na_flag:
                verify_entry["inconsistent_reason_only"] = True

            patched = dict(narr_dict)
            changes = []
            for field in ("left", "right", "bimanual"):
                flag_key = f"{field}_na"
                if na_flags.get(flag_key, False) and patched[field] != "n/a":
                    changes.append(field)
                    patched[field] = "n/a"

            if all(patched[f] == "n/a" for f in ("left", "right", "bimanual")):
                _dbg(f"  [verify] candidate[{ci}] result: REJECT ({reject_reason})", window_log)
                verify_entry["verdict"] = "REJECT"
                verify_entry["patched"] = dict(patched)
                verify_entry["changes"] = changes
                if window_log is not None:
                    window_log["verification"].append(verify_entry)
                results[ci] = (patched, reject_reason)
            else:
                chg = f" changed={changes}" if changes else " (no changes)"
                _dbg(f"  [verify] candidate[{ci}] result: PASS{chg}", window_log)
                verify_entry["verdict"] = "PASS"
                verify_entry["patched"] = dict(patched) if changes else None
                verify_entry["changes"] = changes
                if window_log is not None:
                    window_log["verification"].append(verify_entry)
                results[ci] = (patched, None)

    return results


def _field_sort_key(narr_dict, field, src_idx, text_freq):
    """Coverage-preserving sort key for field-wise selection (lower = better).
    Order: -n_phrases, -text_freq (stability), src_idx (determinism)."""
    text = narr_dict[field]
    n_phrases = len(split_actions_canonical(text) or [])
    tokens = len(text.split())
    return (-n_phrases, -text_freq, src_idx, tokens)


def _consensus_filter(verified, total_samples):
    """Per-field consensus filter over verified (narr_dict, src_idx) tuples.
    Returns patched tuples with fields below the _CONSENSUS_THRESHOLDS set to n/a."""
    if not verified:
        return verified

    field_counts = {"left": 0, "right": 0, "bimanual": 0}
    for narr, _ in verified:
        for f in field_counts:
            if narr.get(f, "n/a") != "n/a":
                field_counts[f] += 1

    fields_to_kill = set()
    for field, thresh in _CONSENSUS_THRESHOLDS.items():
        count = field_counts[field]
        ratio = count / total_samples if total_samples > 0 else 0
        if count < thresh["min_agree"] or ratio < thresh["min_ratio"]:
            fields_to_kill.add(field)

    if not fields_to_kill:
        return verified

    patched = []
    for narr, src_idx in verified:
        narr_out = dict(narr)
        for f in fields_to_kill:
            narr_out[f] = "n/a"
        patched.append((narr_out, src_idx))
    return patched


# Window-level any-reject reasons: a single hit by any sample drops the window
GLOBAL_ANY_REJECT_REASONS = frozenset({
    "vlm_verify_dark",
    "vlm_verify_walking",
    "vlm_verify_other_person",
    "vlm_verify_self_contact",
})


def _global_any_reject_gate(text_reject_reasons, visual_reject_reasons):
    """Return the first global any-reject reason code hit by any sample, or None."""
    for code in visual_reject_reasons:
        if code in GLOBAL_ANY_REJECT_REASONS:
            return code
    for code in text_reject_reasons:
        if code in GLOBAL_ANY_REJECT_REASONS:
            return code
    return None


def _count_actions(text: str) -> int:
    """Count the number of action phrases in a field value.
    Returns 0 for n/a, -1 for parse failure, 1/2/3 for valid phrases."""
    if text == "n/a":
        return 0
    phrases = split_actions_canonical(text)
    return len(phrases) if phrases is not None else -1


def _action_count_dist(narr_dicts, fields=("left", "right", "bimanual")):
    """Per-field action count distribution.
    Returns {field: {count: freq}}. count is 0 (n/a), -1 (parse_error), or 1/2/3."""
    dist = {f: {} for f in fields}
    for narr in narr_dicts:
        for f in fields:
            n = _count_actions(narr.get(f, "n/a"))
            dist[f][n] = dist[f].get(n, 0) + 1
    return dist


def get_caption(
    frames_with_markers: List[np.ndarray],
    n_samples: int = 5,
    n_batches: int = 1,
    temperature: float = 0.6,
    top_p: float = 0.95,
    tracker: Optional[RejectionTracker] = None,
    clip_id: str = "",
    window: str = "",
    fps: Optional[float] = None,
) -> Optional[Dict[str, str]]:
    """Produce one per-hand narration for a window via the staged gate pipeline (see Stage comments below).
    Returns dict with keys think/left/right/bimanual, or None if all attempts fail."""
    if tracker:
        tracker.window_start()

    window_log = {
        "clip_id": clip_id, "window": window,
        "inputs": {
            "n_samples": n_samples, "n_batches": n_batches,
            "temperature": temperature, "top_p": top_p,
            "n_frames": len(frames_with_markers),
        },
        "prints": [], "samples": [], "text_verification": [],
        "verification": [], "selection": {},
    }

    _FIELDS = ("left", "right", "bimanual")
    total_samples = n_samples * n_batches
    reject_counts = {}

    # ── Stage 1: Generate all batches, collect all candidates ──
    all_results = []
    for batch_idx in range(n_batches):
        _dbg(f"[DEBUG] caption batch {batch_idx+1}/{n_batches} (n_samples={n_samples})", window_log)
        results = _sample_captions(
            frames_with_markers, num_return_sequences=n_samples,
            temperature=temperature, top_p=top_p, do_sample=True,
            clip_id=clip_id, window=window,
            window_log=window_log, fps=fps,
        )
        all_results.extend(results)

    # ── Stage 2: Format filter (cheap deterministic) ──
    format_valid = []  # (narr_dict, src_idx)
    format_src_indices = []
    for sample_idx, (narr_dict, think, reason) in enumerate(all_results):
        if reason is None:
            format_valid.append((narr_dict, sample_idx))
            format_src_indices.append(sample_idx)
        else:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1

    _dbg(f"[DEBUG] format filter: {len(format_valid)}/{total_samples} passed", window_log)

    # ── Stage 3: Text gate (pooled, text-only) ──
    text_survivors = []
    text_reject_reasons = []  # structured codes for all format-valid candidates
    if format_valid:
        text_results = _text_verify_candidates(
            [narr for narr, _ in format_valid],
            window_log=window_log,
            source_sample_indices=format_src_indices,
            batch_size=n_samples,
        )
        for i, (patched_narr, t_reason) in enumerate(text_results):
            text_reject_reasons.append(t_reason)
            if t_reason is None:
                text_survivors.append((patched_narr, format_valid[i][1]))
            else:
                reject_counts[t_reason] = reject_counts.get(t_reason, 0) + 1

    _dbg(f"[DEBUG] text gate: {len(text_survivors)}/{len(format_valid)} passed", window_log)

    # ── Stage 4: Visual gate (pooled, with video) ──
    visual_survivors = []
    visual_reject_reasons = []  # structured codes for all text survivors
    any_gates_ran = bool(format_valid)
    if text_survivors:
        visual_results = _verify_candidates(
            frames_with_markers,
            [narr for narr, _ in text_survivors],
            window_log=window_log,
            source_sample_indices=[si for _, si in text_survivors],
            fps=fps, batch_size=n_samples,
        )
        for i, (patched_narr, v_reason) in enumerate(visual_results):
            visual_reject_reasons.append(v_reason)
            if v_reason is None:
                visual_survivors.append((patched_narr, text_survivors[i][1]))
            else:
                reject_counts[v_reason] = reject_counts.get(v_reason, 0) + 1

    n_post_visual_gate = len(visual_survivors)
    _dbg(f"[DEBUG] visual gate: {n_post_visual_gate}/{len(text_survivors)} passed", window_log)

    # ── Stage 5: Global any-reject gate (window-level, any single hit = drop) ──
    global_any_hit = None
    if visual_survivors:
        global_any_hit = _global_any_reject_gate(text_reject_reasons, visual_reject_reasons)
        if global_any_hit:
            _dbg(f"[DEBUG] global any-reject gate: {global_any_hit} → forcing all-n/a", window_log)
            reject_counts[global_any_hit] = reject_counts.get(global_any_hit, 0) + 1
            visual_survivors = []

    # ── Stage 6: Consensus filter ──
    if visual_survivors:
        visual_survivors = _consensus_filter(visual_survivors, total_samples)

    # ── Stage 7: Coverage-preserving selection ──
    # Compute text_freq AFTER consensus (on the final surviving pool only)
    field_winners = {}
    for fld in _FIELDS:
        text_counts = {}
        for narr, _ in visual_survivors:
            text = narr.get(fld, "n/a")
            if text != "n/a":
                norm_text = text.strip().lower()
                text_counts[norm_text] = text_counts.get(norm_text, 0) + 1

        candidates_for_field = []
        for narr, src_idx in visual_survivors:
            if narr[fld] != "n/a":
                norm_text = narr[fld].strip().lower()
                tf = text_counts.get(norm_text, 1)
                key = _field_sort_key(narr, fld, src_idx, tf)
                candidates_for_field.append((key, narr, src_idx))
        if candidates_for_field:
            best_key, best_narr, best_src = min(candidates_for_field, key=lambda x: x[0])
            field_winners[fld] = {
                "text": best_narr[fld],
                "n_phrases": -best_key[0],
                "text_freq": -best_key[1],
                "src_idx": best_key[2],
                "tokens": best_key[3],
            }
        else:
            field_winners[fld] = None

    assembled = {"think": ""}
    for fld in _FIELDS:
        assembled[fld] = field_winners[fld]["text"] if field_winners[fld] is not None else "n/a"

    # ── Stage 8: Final postcheck ──
    assembled, postcheck_meta = postcheck_assembled_triplet(assembled)
    for fld in _FIELDS:
        drop_key = f"{fld}_dropped" if fld != "bimanual" else "bimanual_dropped"
        if postcheck_meta.get(drop_key):
            reason_key = f"{fld}_drop_reason" if fld != "bimanual" else "bimanual_drop_reason"
            _dbg(f"[DEBUG] postcheck dropped {fld}: {postcheck_meta.get(reason_key)}",
                  window_log)
            postcheck_meta[f"{fld}_original_winner"] = field_winners.get(fld)
            field_winners[fld] = None

    # Stage stats
    stage_stats = {}
    postval_narrs = [narr for narr, _, reason in all_results if reason is None]
    stage_stats["post_validation"] = {
        "n_valid": len(postval_narrs),
    }
    _text_entries = window_log.get("text_verification", [])
    _text_n_inconsistent = sum(1 for e in _text_entries if e.get("inconsistent_reason_only"))
    _text_n_parse_error = sum(1 for e in _text_entries if e.get("reason") == "parse_error")
    _visual_entries = window_log.get("verification", [])
    _visual_n_inconsistent = sum(1 for e in _visual_entries if e.get("inconsistent_reason_only"))

    stage_stats["post_text_gate"] = {
        "n_survivors": len(text_survivors),
        "n_parse_error": _text_n_parse_error,
        "n_inconsistent_reason_only": _text_n_inconsistent,
    }
    stage_stats["post_visual_gate"] = {
        "n_survivors": n_post_visual_gate,
        "n_inconsistent_reason_only": _visual_n_inconsistent,
    }
    stage_stats["post_consensus"] = {"n_survivors": len(visual_survivors)}
    stage_stats["final_selected"] = {
        "action_counts": _action_count_dist([assembled], _FIELDS),
        "postcheck": postcheck_meta,
    }
    window_log["stage_stats"] = stage_stats

    # ── Return logic ──
    has_active = any(assembled[f] != "n/a" for f in _FIELDS)

    if has_active:
        if tracker:
            tracker.passed()
            tracker.log_selection(
                clip_id=clip_id, window=window,
                field_winners=field_winners,
                reject_counts=reject_counts,
            )
        _dbg(f"[DEBUG] field-wise winners: L={assembled['left']!r} R={assembled['right']!r} "
              f"B={assembled['bimanual']!r} from {len(visual_survivors)} verified / "
              f"{total_samples} total | rejects: {reject_counts}", window_log)
        window_log["selection"] = {
            "n_verified": len(visual_survivors),
            "reject_counts": reject_counts,
            "field_winners": {fld: field_winners[fld] for fld in _FIELDS},
            "selected": dict(assembled),
            "postcheck": postcheck_meta,
            "outcome": "selected",
        }
        if tracker:
            tracker.log_window(window_log)
        return assembled

    # All fields n/a after assembly, when verified candidates exist or the gates ran
    if visual_survivors or any_gates_ran:
        all_na = {"think": "", "left": "n/a", "right": "n/a", "bimanual": "n/a"}
        if tracker:
            # No segment is written for this window
            tracker.reject(global_any_hit or "all_na_after_gates", clip_id=clip_id, window=window,
                           detail=f"rejects={reject_counts}")
            tracker.log_selection(
                clip_id=clip_id, window=window,
                field_winners={"left": None, "right": None, "bimanual": None},
                reject_counts=reject_counts,
            )
        _dbg(f"[DEBUG] all fields n/a → returning all-n/a | rejects: {reject_counts}",
              window_log)
        window_log["selection"] = {
            "n_verified": len(visual_survivors),
            "reject_counts": reject_counts,
            "field_winners": {"left": None, "right": None, "bimanual": None},
            "selected": dict(all_na),
            "postcheck": postcheck_meta,
            "outcome": "all_na_after_gates",
        }
        if tracker:
            tracker.log_window(window_log)
        return all_na

    # All samples failed
    window_log["selection"] = {
        "n_verified": 0,
        "reject_counts": reject_counts,
        "field_winners": {"left": None, "right": None, "bimanual": None},
        "selected": None,
        "outcome": "all_failed",
    }
    if tracker:
        for narr_dict, think, r in all_results:
            if r in ("content_all_na", "all_na", "caption_na"):
                tracker.reject("caption_na", clip_id=clip_id, window=window,
                               think=think)
                break
        else:
            tracker.reject("all_samples_failed", clip_id=clip_id, window=window,
                           detail=f"rejects={reject_counts}")
        tracker.log_window(window_log)
    return None
