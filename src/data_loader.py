"""
data_loader.py

INPUT stage of the recommendation pipeline (Step 1: Ingestion, item side).

Loads data/raw_skills.csv - the catalog of job roles ("items") and the
skills that define each one - and validates it before any vectorization
happens. Also provides skill normalization so that user-entered skills
and dataset skills map to the exact same vocabulary (per the brief's
warning that naming discrepancies break the similarity math).
"""

import csv
import os
from dataclasses import dataclass
from typing import List


class DatasetError(Exception):
    """Raised when the skills dataset is missing, malformed, or empty."""


@dataclass
class JobRole:
    """A single job role ("item") and its defining skill set."""

    name: str
    skills: List[str]  # normalized skill tokens, e.g. ["python", "cloud_computing"]

    @property
    def skills_text(self) -> str:
        """Space-joined normalized skills, ready for TF-IDF vectorization."""
        return " ".join(self.skills)


def normalize_skill(raw_skill: str) -> str:
    """
    Normalize a single skill string into a canonical token so that
    multi-word skills (e.g. "Cloud Computing") become single vocabulary
    entries (e.g. "cloud_computing") instead of being split into
    separate, less meaningful words.

    Args:
        raw_skill: The raw skill string as typed by a user or stored in
            the dataset (any casing, may have surrounding whitespace).

    Returns:
        A normalized, lowercase, underscore-joined skill token.
    """
    cleaned = raw_skill.strip().lower()
    cleaned = cleaned.replace("/", "_").replace("-", "_")
    cleaned = "_".join(cleaned.split())  # collapse internal whitespace runs
    return cleaned


def load_job_roles(csv_path: str = "data/raw_skills.csv") -> List[JobRole]:
    """
    Load and validate the job-roles dataset from a CSV file with columns
    "role" and "skills" (skills separated by semicolons within the cell).

    Args:
        csv_path: Path to the CSV file.

    Returns:
        A list of JobRole records, each with normalized skill tokens.

    Raises:
        DatasetError: If the file is missing, has the wrong columns,
            contains no rows, or has a row with a blank role/skills.
    """
    if not os.path.isfile(csv_path):
        raise DatasetError(f"Dataset not found at '{csv_path}'.")

    job_roles: List[JobRole] = []
    seen_role_names = set()

    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None or set(reader.fieldnames) != {"role", "skills"}:
            raise DatasetError(
                f"Expected columns ['role', 'skills'], found {reader.fieldnames}."
            )

        for row_number, row in enumerate(reader, start=2):  # header is row 1
            role_name = row["role"].strip()
            raw_skills = row["skills"].strip()

            if not role_name:
                raise DatasetError(f"Row {row_number}: 'role' is empty.")
            if not raw_skills:
                raise DatasetError(f"Row {row_number}: 'skills' is empty.")
            if role_name in seen_role_names:
                raise DatasetError(f"Row {row_number}: duplicate role '{role_name}'.")

            seen_role_names.add(role_name)

            normalized_skills = [
                normalize_skill(skill) for skill in raw_skills.split(";") if skill.strip()
            ]

            if not normalized_skills:
                raise DatasetError(f"Row {row_number}: '{role_name}' has no valid skills.")

            job_roles.append(JobRole(name=role_name, skills=normalized_skills))

    if not job_roles:
        raise DatasetError(f"Dataset at '{csv_path}' contains no rows.")

    return job_roles


def describe_job_roles(job_roles: List[JobRole]) -> str:
    """
    Build a short, human-readable summary of the loaded dataset: how many
    roles were loaded and the size of the resulting shared vocabulary.

    Args:
        job_roles: A list of JobRole records from load_job_roles().

    Returns:
        A formatted multi-line string summary.
    """
    unique_skills = sorted({skill for role in job_roles for skill in role.skills})

    lines = [
        "=" * 50,
        "DATASET SUMMARY: Job Roles",
        "=" * 50,
        f"Job roles loaded : {len(job_roles)}",
        f"Unique skills    : {len(unique_skills)}",
        "Sample roles     :",
    ]
    for role in job_roles[:5]:
        lines.append(f"  - {role.name}: {', '.join(role.skills)}")
    lines.append("=" * 50)
    return "\n".join(lines)


if __name__ == "__main__":
    # Standalone sanity check: python -m src.data_loader
    roles = load_job_roles()
    print(describe_job_roles(roles))
