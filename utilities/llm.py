from typing import Optional, Tuple, Dict, Any, Sequence, List
import json, re
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ----------------------------- Prompt + JSON helpers -----------------------------

def _strip_fences(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip("` \n\t")

def _iter_json_objects(text: str):
    """Yield JSON dicts by scanning for '{' and decoding. Skips preface."""
    t = _strip_fences(text)
    t = re.sub(r"(?s)^.+?\{", "{", t, count=1)
    dec = json.JSONDecoder()
    i, n = 0, len(t)
    while i < n:
        j = t.find("{", i)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(t, j)
            if isinstance(obj, dict):
                yield obj
            i = end
        except json.JSONDecodeError:
            i = j + 1

def _first_obj_with_keys(text: str, keys: Sequence[str]) -> Optional[Dict[str, Any]]:
    for obj in _iter_json_objects(text):
        if any(k in obj for k in keys):
            return obj
    return None

def _last_obj_with_key(text: str, key: str) -> Optional[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    for obj in _iter_json_objects(text):
        if key in obj:
            found.append(obj)
    return found[-1] if found else None


# ----------------------------- Parsers (JSON-only preferred) -----------------------------

def parse_size_ratio(text: str, default: float = 1.0, lo: float = 0.1, hi: float = 10.0) -> float:
    obj = _first_obj_with_keys(text, ("size_ratio", "y_ratio"))
    if obj is not None:
        k = "size_ratio" if "size_ratio" in obj else "y_ratio"
        try:
            v = float(obj[k])
            if lo <= v <= hi:
                return v
        except Exception:
            pass
    # last-resort: single float
    m = re.search(r'\b([-+]?\d*\.?\d+)\b', _strip_fences(text))
    if m:
        try:
            v = float(m.group(1))
            if lo <= v <= hi:
                return v
        except Exception:
            pass
    return default

def parse_penetration(text: str, default: bool = False) -> bool:
    obj = _first_obj_with_keys(text, ("penetration", "is_penetration"))
    if obj is None:
        return default
    val = obj.get("penetration", obj.get("is_penetration"))
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("true", "false"):
            return v == "true"
    if isinstance(val, (int, float)):
        return bool(val)
    return default

def parse_contact_ratio(text: str, default: Optional[float] = None, lo: float = 0.0, hi: float = 1.0) -> Optional[float]:
    obj = _last_obj_with_key(text, "contact_ratio")  # prefer LAST JSON
    if obj is None:
        return default
    try:
        v = float(obj["contact_ratio"])
        if lo <= v <= hi:
            return v
    except Exception:
        pass
    return default


# ----------------------------- Text-only LLM session -----------------------------

class LLMSession:
    """
    Text-only, chat-style wrapper.
    - Uses AutoTokenizer + AutoModelForCausalLM
    - No images; history kept as alternating user/assistant turns
    - Greedy decoding (deterministic)
    """
    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",  # keep your default string unchanged
        prompt_file: str = "utilities/llm_prompts.yaml",
        device: str = "auto",
        load_in_4bit: bool = True,
        torch_dtype: torch.dtype = torch.float16
    ):
        print(f"Init LLMSession with {model_id} model")
        self.model_id = model_id
        self.device = device
        quant = BitsAndBytesConfig(load_in_4bit=True) if load_in_4bit else None

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if device == "auto" else None,
            torch_dtype=torch_dtype,
            quantization_config=quant,
            trust_remote_code=True
        ).eval()

        with open(prompt_file, "r") as f:
            self.prompts = yaml.safe_load(f)

        self.history: List[Dict[str, str]] = []

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """Use tokenizer chat template if available; else simple concat."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: simple "role: content" join
            parts = []
            for m in messages:
                parts.append(f"{m.get('role','user')}: {m.get('content','')}")
            parts.append("assistant:")
            text = "\n".join(parts)
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)

    def ask(
        self,
        prompt_template_key: str,
        object1: str,
        object2: str,
        wanted_alignment: str,
        max_new_tokens: int = 256,
        use_history: bool = True
    ) -> Tuple[str, str]:
        template = self.prompts[prompt_template_key]
        current_prompt = template.format(object1=object1, object2=object2, wanted_alignment=wanted_alignment)

        msgs = []
        if use_history:
            msgs.extend(self.history)
        msgs.append({"role": "user", "content": current_prompt})

        inputs = self._apply_chat_template(msgs)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # greedy
            temperature=0.0
        )
        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        self.history.append({"role": "assistant", "content": decoded})
        return current_prompt, decoded

    def reset_history(self):
        self.history = []


# ----------------------------- High-level getters -----------------------------

def get_prior_scale(llm_session: LLMSession, object1: str, object2: str, wanted_alignment: str) -> float:
    p, r = llm_session.ask(
        prompt_template_key="prior_scale_general",
        object1=object1, object2=object2, wanted_alignment=wanted_alignment
    )
    v = parse_size_ratio(r)
    print("[prior_scale] raw:", r)
    print("[prior_scale] parsed:", v)
    return v

def get_penetration_bool(llm_session: LLMSession, object1: str, object2: str, wanted_alignment: str) -> bool:
    p, r = llm_session.ask(
        prompt_template_key="penetration_check",
        object1=object1, object2=object2, wanted_alignment=wanted_alignment
    )
    v = parse_penetration(r)
    print("[penetration] raw:", r)
    print("[penetration] parsed:", v)
    return v

def get_contact_ratio(llm_session: LLMSession, object1: str, object2: str, wanted_alignment: str) -> Optional[float]:
    p, r = llm_session.ask(
        prompt_template_key="contact_ratio",
        object1=object1, object2=object2, wanted_alignment=wanted_alignment
    )
    v = parse_contact_ratio(r)
    print("[contact] raw:", r)
    print("[contact] parsed:", v)
    return v


# ----------------------------- Demo main (text-only model) -----------------------------

def main():
    # Keep your class default as-is; here we use a TEXT model explicitly:
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    session = LLMSession(model_id=model_id, prompt_file="llm_prompts.yaml")

    obj1 = "nigiri fish"
    obj2 = "nigiri rice"
    wanted = "fish slice resting on the rice, fitting naturally"

    print("== Testing prior scale ==")
    _ = get_prior_scale(session, object1=obj1, object2=obj2, wanted_alignment=wanted)

    print("\n== Testing penetration ==")
    _ = get_penetration_bool(session, object1=obj1, object2=obj2, wanted_alignment=wanted)

    print("\n== Testing contact ratio ==")
    _ = get_contact_ratio(session, object1=obj1, object2=obj2, wanted_alignment=wanted)


if __name__ == "__main__":
    main()
