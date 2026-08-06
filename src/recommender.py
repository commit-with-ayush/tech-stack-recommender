"""
recommender.py

PROCESS stage (part 2) + OUTPUT stage of the recommendation pipeline:
implements Steps 2-4 of the 4-step ranking pipeline (Scoring, Sorting,
Filtering) using Cosine Similarity - chosen over Euclidean distance
because it measures the *angle* between vectors (orientation of
preferences) rather than raw magnitude, so a role with a long skill
list isn't unfairly penalized against one with a short list.
"""

from dataclasses import dataclass
from typing import List

from sklearn.metrics.pairwise import cosine_similarity

from src.vectorizer import VectorSpace, vectorize_user_skills

MIN_USER_SKILLS = 3  # per the brief: minimum 3 inputs for sufficient data density


class InsufficientSkillsError(Exception):
    """Raised when the user provides fewer than MIN_USER_SKILLS skills."""


@dataclass
class Recommendation:
    """A single scored job role, ready for display."""

    role_name: str
    similarity_score: float  # cosine similarity, in [0, 1] for TF-IDF vectors

    @property
    def match_percentage(self) -> float:
        return round(self.similarity_score * 100, 1)


def score_job_roles(vector_space: VectorSpace, user_vector) -> List[Recommendation]:
    """
    Step 2 (Scoring): compute the Cosine Similarity between the user's
    TF-IDF vector and every job role's TF-IDF vector.

    Args:
        vector_space: A fitted VectorSpace (job roles + shared vocabulary).
        user_vector: The user's TF-IDF vector from vectorize_user_skills().

    Returns:
        A list of Recommendation objects, one per job role, in the
        original (unsorted) dataset order.
    """
    similarity_scores = cosine_similarity(user_vector, vector_space.item_matrix)[0]

    return [
        Recommendation(role_name=role.name, similarity_score=float(score))
        for role, score in zip(vector_space.job_roles, similarity_scores)
    ]


def sort_recommendations(recommendations: List[Recommendation]) -> List[Recommendation]:
    """
    Step 3 (Sorting): order recommendations by similarity score,
    descending, so the most relevant roles surface first.

    Args:
        recommendations: Unsorted recommendations from score_job_roles().

    Returns:
        A new list, sorted by similarity_score descending.
    """
    return sorted(recommendations, key=lambda rec: rec.similarity_score, reverse=True)


def filter_top_n(recommendations: List[Recommendation], top_n: int = 3) -> List[Recommendation]:
    """
    Step 4 (Filtering): truncate the sorted list to the top N results,
    preventing "choice overload" by only showing the most relevant matches.

    Args:
        recommendations: Sorted recommendations (see sort_recommendations()).
        top_n: How many results to keep.

    Returns:
        The first `top_n` recommendations.
    """
    return recommendations[:top_n]


def recommend(
    vector_space: VectorSpace,
    raw_skills: List[str],
    top_n: int = 3,
) -> List[Recommendation]:
    """
    Run the full 4-step ranking pipeline end-to-end: validate input,
    vectorize, score, sort, and filter.

    Args:
        vector_space: A fitted VectorSpace (see vectorizer.build_vector_space()).
        raw_skills: The user's raw skill strings (must be >= MIN_USER_SKILLS).
        top_n: How many top recommendations to return.

    Returns:
        The top_n highest-scoring Recommendation objects, sorted descending.

    Raises:
        InsufficientSkillsError: If fewer than MIN_USER_SKILLS non-empty
            skills were provided.
    """
    non_empty_skills = [skill for skill in raw_skills if skill.strip()]

    if len(non_empty_skills) < MIN_USER_SKILLS:
        raise InsufficientSkillsError(
            f"Please provide at least {MIN_USER_SKILLS} skills "
            f"(got {len(non_empty_skills)})."
        )

    user_vector = vectorize_user_skills(vector_space, non_empty_skills)
    scored = score_job_roles(vector_space, user_vector)
    sorted_results = sort_recommendations(scored)
    return filter_top_n(sorted_results, top_n=top_n)


if __name__ == "__main__":
    # Standalone sanity check: python -m src.recommender
    from src.data_loader import load_job_roles
    from src.vectorizer import build_vector_space

    roles = load_job_roles()
    space = build_vector_space(roles)

    results = recommend(space, ["Python", "Cloud Computing", "Automation"], top_n=3)
    for rank, rec in enumerate(results, start=1):
        print(f"{rank}. {rec.role_name} (match: {rec.match_percentage}%)")

    print("\nTesting insufficient skills error:")
    try:
        recommend(space, ["Python"], top_n=3)
    except InsufficientSkillsError as error:
        print(f"OK - caught expected error: {error}")
