"""
main.py

CLI entry point for the Tech Stack Recommender.

Runs the full pipeline end-to-end:

    Step 1 (Ingestion) -> load_job_roles() + collect user skills
    Step 2 (Scoring)   -> build_vector_space() + recommend()
    Step 3 (Sorting)   -> handled inside recommend()
    Step 4 (Filtering) -> handled inside recommend()

Usage:
    python -m src.main --skills "Python" "Cloud Computing" "Automation"
    python -m src.main --top-n 5 --skills "Java" "SQL" "APIs"
    python -m src.main                     # interactive prompt
"""

import argparse
import sys
from typing import List

from src.data_loader import DatasetError, load_job_roles
from src.recommender import InsufficientSkillsError, MIN_USER_SKILLS, Recommendation, recommend
from src.vectorizer import build_vector_space

DEFAULT_DATA_PATH = "data/raw_skills.csv"


def parse_args(argv: List[str]) -> argparse.Namespace:
    """
    Parse command-line arguments for the recommender CLI.

    Args:
        argv: Argument list (excluding the program name), e.g. sys.argv[1:].

    Returns:
        Parsed arguments with `.skills`, `.top_n`, and `.data_path`.
    """
    parser = argparse.ArgumentParser(
        description="Recommend the best-matching job roles for a given set of skills."
    )
    parser.add_argument(
        "--skills",
        nargs="+",
        metavar="SKILL",
        help=f"Your skills, space-separated, quote multi-word skills "
        f'(e.g. --skills "Python" "Cloud Computing" "Automation"). '
        f"Minimum {MIN_USER_SKILLS} required.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top recommendations to display (default: 3).",
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help=f"Path to the job roles dataset (default: {DEFAULT_DATA_PATH}).",
    )
    return parser.parse_args(argv)


def prompt_for_skills() -> List[str]:
    """
    Interactively prompt the user to type in their skills, one per line,
    until they enter a blank line (requiring at least MIN_USER_SKILLS).

    Returns:
        The list of skills the user typed.
    """
    print(f"Enter your skills one at a time (minimum {MIN_USER_SKILLS}).")
    print("Press Enter on a blank line when you're done.\n")

    skills: List[str] = []
    while True:
        skill = input(f"Skill #{len(skills) + 1} (or press Enter to finish): ").strip()
        if not skill:
            if len(skills) >= MIN_USER_SKILLS:
                break
            print(f"  Please enter at least {MIN_USER_SKILLS} skills before finishing.")
            continue
        skills.append(skill)

    return skills


def print_recommendations(recommendations: List[Recommendation]) -> None:
    """Print a clean, ranked summary of the recommended job roles."""
    print("\n" + "=" * 50)
    print(f"TOP {len(recommendations)} RECOMMENDED CAREER PATHS")
    print("=" * 50)
    for rank, rec in enumerate(recommendations, start=1):
        print(f"{rank}. {rec.role_name:<28} (match: {rec.match_percentage}%)")
    print("=" * 50)


def run(argv: List[str]) -> int:
    """
    Run the full CLI pipeline: parse args, load data, gather skills,
    generate recommendations, and print the results.

    Args:
        argv: Argument list (excluding the program name).

    Returns:
        Process exit code (0 on success, 1 on a handled error).
    """
    args = parse_args(argv)

    try:
        job_roles = load_job_roles(args.data_path)
    except DatasetError as error:
        print(f"Error loading dataset: {error}", file=sys.stderr)
        return 1

    vector_space = build_vector_space(job_roles)

    user_skills = args.skills if args.skills else prompt_for_skills()

    try:
        recommendations = recommend(vector_space, user_skills, top_n=args.top_n)
    except InsufficientSkillsError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print_recommendations(recommendations)
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
