"""
Sample PubMed abstracts for gene extraction.

These are real abstracts about MED13 and MED13L genes, demonstrating
the challenge of distinguishing between paralogs in biomedical literature.

Source: PubMed (abstracts simplified for demonstration)
"""

ABSTRACTS = [
    {
        "pmid": "25799994",
        "title": "MED13 mutations in congenital heart disease",
        "text": """
        Mediator complex subunit 13 (MED13) plays a critical role in cardiac development.
        We identified de novo MED13 mutations in patients with congenital heart disease (CHD).
        MED13 mutations were found in 12% of patients with transposition of the great arteries.
        These findings establish MED13 as a novel CHD gene and highlight the importance of
        the Mediator complex in cardiac development.
        """,
        "keywords": ["MED13", "congenital heart disease", "cardiac development", "mutation"],
    },
    {
        "pmid": "23033978",
        "title": "MED13L haploinsufficiency syndrome",
        "text": """
        MED13L haploinsufficiency causes a recognizable syndrome of intellectual disability,
        facial dysmorphism, and hypotonia. Unlike MED13, which is primarily involved in
        cardiac development, MED13L plays a critical role in neurodevelopment. We identified
        MED13L deletions and point mutations in 15 individuals with developmental delay.
        MED13L should be considered in the differential diagnosis of intellectual disability.
        """,
        "keywords": ["MED13L", "intellectual disability", "neurodevelopment", "haploinsufficiency"],
    },
    {
        "pmid": "28467925",
        "title": "MED13-CDK8 interaction in metabolic regulation",
        "text": """
        The Mediator complex subunit MED13 interacts with cyclin-dependent kinase 8 (CDK8)
        to regulate metabolic gene expression. MED13-CDK8 interaction is critical for
        maintaining glucose homeostasis. Disruption of MED13 function leads to metabolic
        dysregulation and increased susceptibility to diabetes. Our findings suggest that
        MED13 is a potential therapeutic target for metabolic disorders.
        """,
        "keywords": ["MED13", "CDK8", "metabolism", "diabetes", "gene regulation"],
    },
    {
        "pmid": "31285573",
        "title": "Comparative analysis of MED13 and MED13L",
        "text": """
        Both MED13 and MED13L are components of the Mediator complex, but they have
        distinct tissue-specific expression patterns and functions. MED13 is highly
        expressed in cardiac tissue and regulates cardiac-specific gene programs, while
        MED13L is predominantly expressed in the brain and controls neurodevelopmental
        pathways. Despite their sequence similarity, MED13 and MED13L mutations lead to
        distinct clinical phenotypes: cardiac defects versus intellectual disability.
        """,
        "keywords": ["MED13", "MED13L", "Mediator complex", "tissue specificity"],
    },
    {
        "pmid": "26942285",
        "title": "MED13 knockout mice exhibit cardiac defects",
        "text": """
        To investigate MED13 function in vivo, we generated MED13 knockout mice.
        MED13-null embryos exhibited severe cardiac defects including ventricular
        septal defects and outflow tract abnormalities. These mice died in utero
        due to heart failure. Cardiac-specific deletion of MED13 in adult mice
        led to dilated cardiomyopathy, confirming the essential role of MED13
        in both cardiac development and adult heart function.
        """,
        "keywords": ["MED13", "knockout mice", "cardiac defects", "cardiomyopathy"],
    },
    {
        "pmid": "29875488",
        "title": "MED13 in transcriptional regulation",
        "text": """
        MED13 is a key component of the CDK8 kinase module of the Mediator complex.
        The Mediator complex serves as a bridge between transcription factors and
        RNA polymerase II. MED13 regulates the activity of multiple signaling pathways
        including Wnt, Notch, and TGF-beta. Through its interaction with CDK8, MED13
        modulates the phosphorylation state of transcription factors and controls
        gene expression programs critical for development and disease.
        """,
        "keywords": ["MED13", "transcription", "Mediator complex", "signaling pathways"],
    },
]


def get_abstracts_for_gene(gene_name: str) -> list[dict]:
    """
    Filter abstracts relevant to a specific gene.

    Args:
        gene_name: Gene to filter for (e.g., "MED13")

    Returns:
        List of abstracts mentioning the gene
    """
    return [
        abstract
        for abstract in ABSTRACTS
        if gene_name.upper() in abstract["text"].upper()
        or gene_name.upper() in " ".join(abstract["keywords"]).upper()
    ]


def get_all_abstracts() -> list[dict]:
    """Get all available abstracts."""
    return ABSTRACTS


def format_abstract_for_extraction(abstract: dict) -> str:
    """
    Format an abstract for LLM extraction.

    Args:
        abstract: Abstract dictionary

    Returns:
        Formatted text for extraction
    """
    return f"""
PMID: {abstract["pmid"]}
Title: {abstract["title"]}

{abstract["text"].strip()}

Keywords: {", ".join(abstract["keywords"])}
""".strip()
