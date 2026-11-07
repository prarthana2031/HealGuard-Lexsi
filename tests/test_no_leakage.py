import json
import hashlib
import pytest

def get_seeds(filepath):
    with open(filepath, "r") as f:
        return {json.loads(line)["seed"] for line in f}

def test_sealed_split_has_no_overlap():
    """Ensure no seed in sealed appears in any other split."""
    sealed = get_seeds("data/final_sealed/sealed_traces.jsonl")
    train = get_seeds("data/train_clean/expert_traces.jsonl")
    dev = get_seeds("data/dev/expert_traces.jsonl")
    drift = get_seeds("data/drift_targeted/expert_traces.jsonl")
    gate = get_seeds("data/gate_regression/expert_traces.jsonl")
    
    assert len(sealed.intersection(train)) == 0, "Sealed overlaps with train!"
    assert len(sealed.intersection(dev)) == 0, "Sealed overlaps with dev!"
    assert len(sealed.intersection(drift)) == 0, "Sealed overlaps with drift!"
    assert len(sealed.intersection(gate)) == 0, "Sealed overlaps with gate!"

def test_sealed_hash():
    """Ensure the sealed data hash matches the committed hash."""
    with open("data/final_sealed/sealed_traces.jsonl", "rb") as f:
        content = f.read()
        # Normalize CRLF to LF to avoid git line-ending issues
        normalized = content.replace(b"\r\n", b"\n")
        file_hash = hashlib.sha256(normalized).hexdigest()
        
    with open("data/splits/sealed_hash.txt", "r") as f:
        expected_hash = f.read().strip()
        
    assert file_hash == expected_hash, "Sealed set has been modified!"