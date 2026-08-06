"""
vectorizer.py

PROCESS stage (part 1) of the recommendation pipeline: "Bridging the
Language Barrier Through Vector Mapping".

Machines don't understand skill names like "Python" or "Cloud Computing"
directly - they need numerical arrays (vectors) in a shared vocabulary
space. This module builds that shared space with TF-IDF (Term Frequency
x Inverse Document Frequency), which rewards specific/descriptive skills
and penalizes generic ones that appear across many roles.

Critically, the SAME fitted vectorizer is reused to transform both the
job-role corpus and the user's input skills, so they land in the exact
same vector space - a mismatch here (e.g. fitting two separate
vectorizers) would silently break the similarity math.
"""

from dataclasses import dataclass
from typing import List

from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_loader import JobRole, normalize_skill


@dataclass
class VectorSpace:
    """
    The fitted TF-IDF vectorizer plus the resulting item (job role)
    matrix, all sharing one vocabulary.
    """

    vectorizer: TfidfVectorizer
    item_matrix: spmatrix  # shape: (n_job_roles, vocab_size)
    job_roles: List[JobRole]

    @property
    def vocabulary_size(self) -> int:
        return len(self.vectorizer.vocabulary_)


def build_vector_space(job_roles: List[JobRole]) -> VectorSpace:
    """
    Fit a TF-IDF vectorizer on the full job-role skill corpus and
    transform every job role into its TF-IDF vector.

    Args:
        job_roles: Job roles loaded via data_loader.load_job_roles().

    Returns:
        A VectorSpace containing the fitted vectorizer and item matrix.
    """
    corpus = [role.skills_text for role in job_roles]

    # token_pattern matches underscore-joined tokens like "cloud_computing"
    # as single terms (default sklearn pattern already treats "_" as a
    # word character, so no custom pattern is strictly required, but we
    # make it explicit here for clarity and robustness).
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    item_matrix = vectorizer.fit_transform(corpus)

    return VectorSpace(vectorizer=vectorizer, item_matrix=item_matrix, job_roles=job_roles)


def vectorize_user_skills(vector_space: VectorSpace, raw_skills: List[str]) -> spmatrix:
    """
    Normalize and vectorize the user's entered skills using the SAME
    fitted vectorizer as the job-role corpus, so the resulting vector
    lives in the identical vocabulary space.

    Skills the user enters that never appear in any job role's skill
    list are simply ignored by the vectorizer (out-of-vocabulary terms
    contribute nothing) - this is expected and does not raise an error.

    Args:
        vector_space: A VectorSpace returned by build_vector_space().
        raw_skills: The user's raw skill strings, any casing/spacing.

    Returns:
        A 1-row sparse TF-IDF vector for the user profile.
    """
    normalized = [normalize_skill(skill) for skill in raw_skills if skill.strip()]
    user_text = " ".join(normalized)
    return vector_space.vectorizer.transform([user_text])


if __name__ == "__main__":
    # Standalone sanity check: python -m src.vectorizer
    from src.data_loader import load_job_roles

    roles = load_job_roles()
    space = build_vector_space(roles)
    print(f"Vocabulary size: {space.vocabulary_size}")
    print(f"Item matrix shape: {space.item_matrix.shape}")

    user_vector = vectorize_user_skills(space, ["Python", "Cloud Computing", "Automation"])
    print(f"User vector shape: {user_vector.shape}")
    print(f"User vector non-zero terms: {user_vector.nnz}")
