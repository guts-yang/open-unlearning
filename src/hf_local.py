"""Resolve Hugging Face repo IDs to already-downloaded hub snapshots.

Avoids hf-mirror HEAD timeouts when $HF_HUB_CACHE already has the files.
"""
import os


def resolve_hf_snapshot(repo_id, kind="models"):
    """Return local snapshot dir for `org/name`, or the original id if missing."""
    if not repo_id or os.path.isdir(repo_id):
        return repo_id
    if "/" not in repo_id:
        return repo_id
    hub = (
        os.getenv("HF_HUB_CACHE")
        or os.getenv("HUGGINGFACE_HUB_CACHE")
        or os.path.join(os.getenv("HF_HOME", ""), "hub")
    )
    if not hub:
        return repo_id
    org, name = repo_id.split("/", 1)
    base = os.path.join(hub, f"{kind}--{org}--{name}")
    ref = os.path.join(base, "refs", "main")
    snap = None
    if os.path.isfile(ref):
        with open(ref, encoding="utf-8") as f:
            snap = f.read().strip()
    else:
        snaps_dir = os.path.join(base, "snapshots")
        if os.path.isdir(snaps_dir):
            names = [n for n in os.listdir(snaps_dir) if not n.startswith(".")]
            if names:
                snap = names[0]
    if not snap:
        return repo_id
    path = os.path.join(base, "snapshots", snap)
    return path if os.path.isdir(path) else repo_id
