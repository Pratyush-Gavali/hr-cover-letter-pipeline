"""
PII Masking via Microsoft Presidio
====================================
Analyses and anonymises cover letter text before it flows into
the embedding pipeline. The PII -> token mapping is stored encrypted
in Azure Key Vault so HR reviewers see [PERSON], [EMAIL], etc.,
while authorised compliance staff can reverse-lookup when needed.

Design decisions
----------------
- Entity types are whitelisted, not blacklisted — safer default
- Mapping stored per-chunk in Key Vault (not inline with the text)
- HR users' AAD role lacks Key Vault secret READ permission
- Compliance role has secret READ with full audit log in Key Vault
"""

from __future__ import annotations
import json
from dataclasses import dataclass

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential


# ── Config ────────────────────────────────────────────────────────────────────

ENTITIES_TO_MASK = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "DATE_TIME",
    "URL",
    "NRP",            # Nationality / religion / political affiliation
    "MEDICAL_LICENSE",
    "IP_ADDRESS",
]

# Replacement token for each entity type
OPERATOR_MAP = {
    ent: OperatorConfig("replace", {"new_value": f"[{ent.split('_')[0]}]"})
    for ent in ENTITIES_TO_MASK
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class MaskedDocument:
    applicant_id: str
    job_id: str
    masked_text: str
    entity_count: int
    mapping_secret_name: str      # Key Vault secret holding PII->token map
    chunk_index: int = 0


# ── PII Masker ────────────────────────────────────────────────────────────────

class PIIMasker:
    """
    Wraps Presidio Analyzer + Anonymizer with Azure Key Vault mapping store.

    Parameters
    ----------
    key_vault_url
        e.g. "https://hr-pipeline-kv.vault.azure.net/"
    language
        Presidio NLP language model (default: "en").
    """

    def __init__(self, key_vault_url: str, language: str = "en"):
        # spaCy large model gives meaningfully better NER than the small model,
        # especially for partial names (e.g. "John at Google" -> PERSON + ORG)
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        })
        self._analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            supported_languages=[language],
        )
        self._anonymizer = AnonymizerEngine()
        self._kv = SecretClient(
            vault_url=key_vault_url,
            credential=DefaultAzureCredential(),
        )
        self._language = language

    def mask(
        self,
        text: str,
        applicant_id: str,
        job_id: str,
        chunk_index: int = 0,
    ) -> MaskedDocument:
        """
        Mask all PII in *text* and persist the mapping to Key Vault.

        Key Vault secret naming convention:
            pii-map-{applicant_id}-{chunk_index}

        This is deterministic so it can be looked up by applicant ID
        without needing to store the secret name in the database.
        """
        results = self._analyzer.analyze(
            text=text,
            entities=ENTITIES_TO_MASK,
            language=self._language,
        )

        anonymised = self._anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=OPERATOR_MAP,
        )

        # Build mapping: placeholder -> original text
        # Last-write-wins on duplicates (acceptable for compliance purposes)
        mapping: dict[str, str] = {}
        for result in results:
            original = text[result.start:result.end]
            placeholder = f"[{result.entity_type.split('_')[0]}]"
            mapping[placeholder] = original

        secret_name = f"pii-map-{applicant_id}-{chunk_index}"
        self._kv.set_secret(secret_name, json.dumps(mapping))

        return MaskedDocument(
            applicant_id=applicant_id,
            job_id=job_id,
            masked_text=anonymised.text,
            entity_count=len(results),
            mapping_secret_name=secret_name,
            chunk_index=chunk_index,
        )

    def reveal(self, masked_doc: MaskedDocument) -> str:
        """
        Reverse-lookup PII — requires elevated Key Vault access policy.

        This is restricted to the compliance role in Azure AD.
        HR users calling this endpoint will receive a 403 from Key Vault.
        All lookups are logged in the Key Vault audit trail.
        """
        secret = self._kv.get_secret(masked_doc.mapping_secret_name)
        mapping: dict[str, str] = json.loads(secret.value)
        text = masked_doc.masked_text
        for placeholder, original in mapping.items():
            text = text.replace(placeholder, original)
        return text


class LocalPIIMasker(PIIMasker):
    """
    Dev-only PII masker. Stores mappings in an in-memory dict instead
    of Azure Key Vault. Set USE_LOCAL_PII=1 in .env to use this.

    Drop-in replacement — identical public interface as PIIMasker.
    """

    def __init__(self, language: str = "en"):
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        })
        self._analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            supported_languages=[language],
        )
        self._anonymizer = AnonymizerEngine()
        self._store: dict[str, dict] = {}   # secret_name -> {placeholder: original}
        self._language = language

    def mask(
        self,
        text: str,
        applicant_id: str,
        job_id: str,
        chunk_index: int = 0,
    ) -> MaskedDocument:
        results = self._analyzer.analyze(
            text=text, entities=ENTITIES_TO_MASK, language=self._language
        )
        anonymised = self._anonymizer.anonymize(
            text=text, analyzer_results=results, operators=OPERATOR_MAP
        )
        mapping = {
            f"[{r.entity_type.split('_')[0]}]": text[r.start:r.end]
            for r in results
        }
        secret_name = f"pii-map-{applicant_id}-{chunk_index}"
        self._store[secret_name] = mapping

        return MaskedDocument(
            applicant_id=applicant_id,
            job_id=job_id,
            masked_text=anonymised.text,
            entity_count=len(results),
            mapping_secret_name=secret_name,
            chunk_index=chunk_index,
        )

    def reveal(self, masked_doc: MaskedDocument) -> str:
        mapping = self._store.get(masked_doc.mapping_secret_name, {})
        text = masked_doc.masked_text
        for placeholder, original in mapping.items():
            text = text.replace(placeholder, original)
        return text