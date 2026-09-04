"""The executor's zero-LLM guardrail, enforced on the DEPENDENCY surface.

`~/Development/CLAUDE.md` standing rule: *LLM calls are confined to research —
the executor has ZERO exposure.* `alpha-engine-config-I7723` is the instance
where that guardrail was violated: a live Anthropic diagnosis call site AND
the `anthropic` SDK back in the trading box's venv, pulled in transitively by
`krepis`'s old `flow-doctor[diagnosis]` extra. The call site was removed
(`alpha-engine-config-PR7725`, flow-doctor `diagnosis.enabled: false`); the
SDK left the lock on 2026-08-25 (`crucible-executor-PR492`/`PR493`, which
raised flow-doctor to 0.16.0 and recompiled).

I7723's remaining deliverable — verbatim: *"the guard asserting `anthropic`
does not appear in `crucible-executor`'s resolved requirements"* — is this
file. Until it existed, nothing failed if a transitive bump put the SDK back.

WHY THE EXISTING GUARDS DO NOT COVER THIS, measured 2026-09-04:

  - `.github/workflows/provider-linkage-guard.yml` (`alpha-engine-config-I9295`)
    scans `nousergon-lib/scripts/provider_linkage_guard.py::DEFAULT_EXTENSIONS`
    = {.py .ts .tsx .js .mjs .sh .bash .yaml .yml .json .toml .cfg .ini .plist
    .service .timer}. `requirements.in` has suffix `.in` — in NEITHER that set
    nor `DOC_EXTENSIONS`; `requirements.txt` is `.txt`, i.e. `DOC_EXTENSIONS`,
    scanned only under an opt-in `--include-docs` this repo's caller does not
    pass. The dependency surface is outside that guard entirely.
  - `tests/test_eod_reconcile_logic.py::test_eod_reconcile_does_not_import_anthropic`
    pins ONE source file. A distribution installed into the trading venv needs
    no `import` anywhere in this tree to be present on the box.

So the two live guards cover the CALL SITE and this one covers the INSTALLED
SURFACE. They are complementary, not redundant.

Scope note: this asserts on the repo's declared and locked dependency files,
which is what CI can see. It does not assert on the box's actual venv — that
is `RefreshExecutorDeploy`'s job, and note that `pip install -r` does not
UNINSTALL a distribution dropped from the lock, so a clean lock is necessary
and not sufficient for a clean box (recorded on `alpha-engine-config-I7723`).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Distribution names that MUST NOT resolve into this repo's environment.
# Vendor-as-data, mirroring `provider_linkage_guard.py::PROVIDERS`: adding a
# vendor is a row here, not a new test. `krepis` is deliberately absent — it
# is the router adapter boundary and the one legitimate SDK holder in the
# fleet (principle 8), and this repo depends on it for `ssm_log_capture`,
# `trading_calendar` and `console_url`, none of which are LLM surfaces.
_FORBIDDEN_DISTRIBUTIONS = frozenset(
    {
        "anthropic",
        "openai",
        "cohere",
        "mistralai",
        "google-generativeai",
        "google-genai",
        "litellm",
        "langchain",
        "langchain-core",
        "llama-index",
        "transformers",
        "vertexai",
        "replicate",
        "together",
    }
)

# Files a resolved or declared dependency can enter through.
_DEPENDENCY_FILES = ("requirements.in", "requirements.txt", "pyproject.toml")

# A requirement line's distribution name: the leading name token, before any
# extras bracket, version specifier, environment marker, or VCS `@` form.
_DIST_TOKEN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalise(name: str) -> str:
    """PEP 503 normalisation — `Google_GenerativeAI` and `google-generativeai`
    are the same distribution and a guard that misses one is not a guard."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


_FORBIDDEN_NORMALISED = frozenset(_normalise(n) for n in _FORBIDDEN_DISTRIBUTIONS)


def _declared_distributions(path: Path) -> set[str]:
    """Every distribution name a dependency file mentions on a non-comment line.

    Deliberately over-inclusive for `pyproject.toml`: any quoted or bare
    requirement-shaped token is considered. A false positive here is a loud
    test failure someone reads; a false negative is the SDK back on the box.
    """
    found: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        for chunk in re.split(r"[\"',\[\]=]+", line) if path.suffix == ".toml" else [line]:
            m = _DIST_TOKEN.match(chunk)
            if m:
                found.add(_normalise(m.group(1)))
    return found


@pytest.mark.parametrize("filename", _DEPENDENCY_FILES)
def test_no_llm_sdk_in_dependency_file(filename: str) -> None:
    path = _REPO_ROOT / filename
    assert path.is_file(), (
        f"{filename} is missing from the repo root. This guard reads the "
        "dependency surface; if the file moved, re-point it rather than "
        "deleting the assertion (alpha-engine-config-I7723)."
    )

    offenders = sorted(_declared_distributions(path) & _FORBIDDEN_NORMALISED)
    assert not offenders, (
        f"{filename} resolves LLM provider SDK(s): {', '.join(offenders)}.\n"
        "The executor is the live trader and has ZERO LLM exposure by standing "
        "rule (~/Development/CLAUDE.md; alpha-engine-config-I7723). An SDK "
        "here reaches the trading box's venv at the postclose SF's "
        "RefreshExecutorDeploy stage.\n"
        "This is almost always TRANSITIVE — the 2026-05 instance came in via "
        "krepis's `flow-doctor[diagnosis]` extra. Fix it at the source (drop "
        "the extra / raise the flow-doctor floor) and recompile the lock. Do "
        "NOT fix it by routing the call through the krepis router: routing "
        "preserves a call site the guardrail forbids."
    )


def test_forbidden_list_is_normalised_and_non_empty() -> None:
    """A guard whose vendor table silently emptied would pass every assertion
    above while checking nothing — the fleet's own detection-blindness shape."""
    assert _FORBIDDEN_NORMALISED, "the forbidden-distribution table is empty"
    assert "anthropic" in _FORBIDDEN_NORMALISED
    assert _normalise("Google_Generative.AI") == "google-generative-ai"


def test_guard_would_catch_a_reintroduced_sdk(tmp_path: Path) -> None:
    """Deliberately-broken case: the assertion must FAIL on a lock that carries
    the SDK. Without this, a parsing regression renders the guard vacuously
    green and indistinguishable from a clean repo."""
    fake = tmp_path / "requirements.txt"
    fake.write_text("arcticdb==6.23.0\nanthropic==0.117.0\npandas==2.3.3\n", encoding="utf-8")
    assert _declared_distributions(fake) & _FORBIDDEN_NORMALISED == {"anthropic"}

    fake_toml = tmp_path / "pyproject.toml"
    fake_toml.write_text('dependencies = ["openai>=1.0", "pandas~=2.3"]\n', encoding="utf-8")
    assert _declared_distributions(fake_toml) & _FORBIDDEN_NORMALISED == {"openai"}
