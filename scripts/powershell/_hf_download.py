"""Download exactly the HF files referenced by the active PocketTTS model config.

The app loads weights via hf_hub_download() against HUGGINGFACE_HUB_CACHE, so this
populates that cache with only the pinned files the config actually references —
not entire repos (which would pull every language/variant). The URI set is parsed
from the same YAML pocket-tts resolves at load time, so it can't drift.

Usage: python _hf_download.py [model_path]   # model_path defaults to env POCKET_TTS_MODEL_PATH or 'english'
"""
import os
import re
import sys
from pathlib import Path


def resolve_config_yaml(model_path: str) -> Path:
    """Mirror pocket-tts's config resolution: a .yaml path, or a variant name."""
    p = Path(model_path).expanduser()
    if p.suffix == '.yaml' and p.is_file():
        return p.resolve()
    from pocket_tts import __file__ as pkg_file
    cfg_dir = Path(pkg_file).resolve().parent / 'config'
    candidate = cfg_dir / f'{model_path}.yaml'
    if candidate.is_file():
        return candidate
    sys.exit(f"error: cannot resolve model config '{model_path}' (looked for {candidate})")


def main() -> None:
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('POCKET_TTS_MODEL_PATH', 'english') or 'english'
    yaml_path = resolve_config_yaml(model_path)

    text = yaml_path.read_text(encoding='utf-8')
    # hf://<repo_id>/<file/path>@<revision>   (revision optional; repo may contain dots/hyphens)
    uri_re = re.compile(r'hf://([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)/(\S+?)(?:@([0-9a-f]{7,40}))?(?=[\s\'"\],]|$)')
    seen: dict[tuple[str, str, str | None], None] = {}
    for repo_id, filename, revision in uri_re.findall(text):
        seen[(repo_id, filename, revision)] = None

    if not seen:
        sys.exit(f"error: no hf:// URIs found in {yaml_path}")

    from huggingface_hub import hf_hub_download
    cache_dir = os.environ.get('HUGGINGFACE_HUB_CACHE') or os.environ.get('HF_HOME')
    print(f"Config: {yaml_path}")
    if cache_dir:
        print(f"Cache : {cache_dir}")
    for repo_id, filename, revision in seen:
        rev = f"@{revision}" if revision else ''
        print(f"  hf_hub_download({repo_id!r}, {filename!r}{rev})")
        path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"    -> {path} ({size_mb:.1f} MB)")

    print("Done.")


if __name__ == '__main__':
    main()
