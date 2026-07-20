import argparse
import dataclasses
import json
import logging
import os

from issue_worker.config import load_config
from issue_worker.data_adapters.github_issue_adapter import GitHubIssueAdapter

logger = logging.getLogger("issue_worker.main")
logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Fields written to the agent-visible file. Everything else on
# RawIssueRecord is treated as grading-only.
AGENT_VISIBLE_FIELDS = {
    "source",
    "source_id",
    "title",
    "body",
    "state",
    "labels",
    "created_at",
    "url",
    "collected_At",
}

def split_record(record)->tuple[dict,dict]:
    """Split a RawIssueRecord into (agent_visible_dict, grading_only_dict).

    Both dicts carry source_id so the two files can be joined later.
    """
    record_dict = dataclasses.asdict(record)

    agent_visible = {
        key:value 
        for key , value in record_dict.items() 
        if key in AGENT_VISIBLE_FIELDS

    }
    grading_only = {
        key : value
        for key , value in record_dict.items()
        if key not in AGENT_VISIBLE_FIELDS or key =="source_id"
    }

    return agent_visible, grading_only

def run(limit :int) ->None:
    
    config = load_config()

    os.makedirs(os.path.dirname(config.output_path),exist_ok = True)
    os.makedirs(os.path.dirname(config.grading_key_path),exist_ok=True)

    adapter = GitHubIssueAdapter(config)

    collected = 0
    with open(config.output_path, "a") as raw_file, open(
        config.grading_key_path, "a"
    ) as grading_file:
        for record in adapter.fetch_records(limit):
            agent_visible, grading_only = split_record(record)

            raw_file.write(json.dumps(agent_visible)+"\n")
            grading_file.write(json.dumps(grading_only) + "\n")
            collected += 1

    logger.info(
        f"wrote {collected} records to {config.output_path} and {config.grading_key_path}")
    
def main()->None:
    parser = argparse.ArgumentParser(description="File 17 : collect Githubissues for issue worker")
    parser.add_argument(
        "--limit",
        type = int,
        default = 20,
        help = "Number of new issue records to collect this run (default :20)"
        )
    args = parser.parse_args()

    run(limit = args.limit)


if __name__ == "__main__":
    main()
