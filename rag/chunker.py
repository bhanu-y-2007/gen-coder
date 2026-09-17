def create_chunks(
    documents,
    chunk_size=500,
    chunk_overlap=100
):
    """
    Split documents into overlapping chunks while
    preserving metadata.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size")

    chunks = []
    step = max(1, chunk_size - chunk_overlap)

    for document in documents:

        text = document["text"]
        metadata = document["metadata"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": metadata.copy()
                })

            start += step

    return chunks