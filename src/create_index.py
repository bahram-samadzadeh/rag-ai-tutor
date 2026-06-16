"""Create the Azure AI Search index for the RAG pipeline.

Defines a schema with a vector field sized to the embedding model
(text-embedding-3-small -> 1536 dimensions) and an HNSW configuration
for approximate nearest-neighbour search. Running this is idempotent:
create_or_update_index overwrites the existing index definition.
"""
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchProfile,
)

from .config import settings


def create_index() -> None:
    """Create or update the search index with a vector field."""
    client = SearchIndexClient(
        endpoint=settings.SEARCH_ENDPOINT,
        credential=AzureKeyCredential(settings.SEARCH_KEY),
    )

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        # content is searchable so we keep a keyword/text fallback alongside
        # vector search (hybrid retrieval is stronger than vector alone).
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            # Must match the embedding model's output dimensions exactly,
            # or document uploads are rejected.
            vector_search_dimensions=settings.EMBED_DIMENSIONS,
            vector_search_profile_name="hnsw-profile",
        ),
    ]

    # HNSW is the standard ANN algorithm; cosine is the right metric for
    # OpenAI text embeddings, which are normalised for cosine similarity.
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-config",
                parameters=HnswParameters(metric="cosine"),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw-config",
            )
        ],
    )

    index = SearchIndex(
        name=settings.INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
    )

    result = client.create_or_update_index(index)
    print(f"Index '{result.name}' created/updated successfully.")


if __name__ == "__main__":
    create_index()
