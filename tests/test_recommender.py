"""
test_recommender.py

Sanity tests for the Tech Stack Recommender pipeline. Validates each
stage (data loading, vectorization, scoring, sorting, filtering) and
the end-to-end recommend() function.

Run with:
    python -m pytest tests/
"""

import pytest

from src.data_loader import DatasetError, JobRole, load_job_roles, normalize_skill
from src.recommender import (
    InsufficientSkillsError,
    MIN_USER_SKILLS,
    filter_top_n,
    recommend,
    score_job_roles,
    sort_recommendations,
)
from src.vectorizer import build_vector_space, vectorize_user_skills


@pytest.fixture(scope="module")
def job_roles():
    return load_job_roles("data/raw_skills.csv")


@pytest.fixture(scope="module")
def vector_space(job_roles):
    return build_vector_space(job_roles)


# ---------------- Ingestion stage (data_loader) ----------------

def test_load_job_roles_returns_expected_count(job_roles) -> None:
    assert len(job_roles) == 18
    assert all(isinstance(role, JobRole) for role in job_roles)


def test_job_role_skills_are_normalized(job_roles) -> None:
    cloud_architect = next(role for role in job_roles if role.name == "Cloud Architect")
    assert "cloud_computing" in cloud_architect.skills
    # Ensure no raw multi-word skills leaked through unnormalized.
    assert all(" " not in skill for skill in cloud_architect.skills)


def test_normalize_skill_handles_spacing_and_case() -> None:
    assert normalize_skill("Cloud Computing") == "cloud_computing"
    assert normalize_skill("  python  ") == "python"
    assert normalize_skill("CI/CD") == "ci_cd"


def test_load_job_roles_missing_file_raises() -> None:
    with pytest.raises(DatasetError):
        load_job_roles("data/does_not_exist.csv")


# ---------------- Scoring stage (vectorizer) ----------------

def test_vector_space_shape(job_roles, vector_space) -> None:
    n_roles = len(job_roles)
    assert vector_space.item_matrix.shape[0] == n_roles
    assert vector_space.item_matrix.shape[1] == vector_space.vocabulary_size


def test_vectorize_user_skills_shares_vocabulary(vector_space) -> None:
    user_vector = vectorize_user_skills(vector_space, ["Python", "SQL", "Machine Learning"])
    assert user_vector.shape[1] == vector_space.vocabulary_size
    assert user_vector.nnz == 3  # three matched, in-vocabulary terms


def test_vectorize_user_skills_ignores_unknown_terms(vector_space) -> None:
    # "Underwater Basket Weaving" isn't in any job role's skill list.
    user_vector = vectorize_user_skills(
        vector_space, ["Python", "SQL", "Underwater Basket Weaving"]
    )
    assert user_vector.nnz == 2  # only the two known skills contribute


# ---------------- Scoring + Sorting + Filtering (recommender) ----------------

def test_score_job_roles_returns_one_score_per_role(job_roles, vector_space) -> None:
    user_vector = vectorize_user_skills(vector_space, ["Python", "SQL", "Databases"])
    scored = score_job_roles(vector_space, user_vector)
    assert len(scored) == len(job_roles)
    assert all(0.0 <= rec.similarity_score <= 1.0 for rec in scored)


def test_sort_recommendations_is_descending(job_roles, vector_space) -> None:
    user_vector = vectorize_user_skills(vector_space, ["Python", "SQL", "Databases"])
    scored = score_job_roles(vector_space, user_vector)
    sorted_scored = sort_recommendations(scored)
    scores = [rec.similarity_score for rec in sorted_scored]
    assert scores == sorted(scores, reverse=True)


def test_filter_top_n_truncates_correctly(job_roles, vector_space) -> None:
    user_vector = vectorize_user_skills(vector_space, ["Python", "SQL", "Databases"])
    scored = sort_recommendations(score_job_roles(vector_space, user_vector))
    top_3 = filter_top_n(scored, top_n=3)
    assert len(top_3) == 3
    assert top_3 == scored[:3]


def test_recommend_end_to_end_returns_relevant_top_match(vector_space) -> None:
    results = recommend(
        vector_space,
        ["Python", "Cloud Computing", "Automation"],
        top_n=3,
    )
    assert len(results) == 3
    # Cloud Architect explicitly lists cloud_computing + automation,
    # so it should be the strongest match for this skill set.
    assert results[0].role_name == "Cloud Architect"
    # Results must be sorted descending by score.
    scores = [rec.similarity_score for rec in results]
    assert scores == sorted(scores, reverse=True)


def test_recommend_raises_on_insufficient_skills(vector_space) -> None:
    with pytest.raises(InsufficientSkillsError):
        recommend(vector_space, ["Python"], top_n=3)


def test_recommend_raises_with_exactly_below_minimum(vector_space) -> None:
    too_few = ["Python"] * (MIN_USER_SKILLS - 1)
    with pytest.raises(InsufficientSkillsError):
        recommend(vector_space, too_few, top_n=3)


def test_recommend_succeeds_at_exactly_minimum_skills(vector_space) -> None:
    exactly_min = ["Python", "SQL", "Databases"][:MIN_USER_SKILLS]
    results = recommend(vector_space, exactly_min, top_n=3)
    assert len(results) == 3
