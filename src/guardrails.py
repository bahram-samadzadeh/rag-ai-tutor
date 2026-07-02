from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.core.credentials import AzureKeyCredential

from .config import settings

SEVERITY_THRESHOLD = 2


def get_safety_client() -> ContentSafetyClient:
    return ContentSafetyClient(
        settings.CONTENT_SAFETY_ENDPOINT,
        AzureKeyCredential(settings.CONTENT_SAFETY_KEY),
    )


def is_safe(client: ContentSafetyClient, text: str) -> bool:
    result = client.analyze_text(AnalyzeTextOptions(text=text))
    return all(cat.severity < SEVERITY_THRESHOLD for cat in result.categories_analysis)