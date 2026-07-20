import json
import logging
import os
import re
import time
from typing import Iterable, Optional
import requests

from issue_worker.config import CollectorConfig
from issue_worker.data_adapters.base import DataSourceAdapter, RawIssueRecord

logger = logging.getLogger("issue_worker.github_adapter")
logging.basicConfig(
    level=logging.INFO,
    format= "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

GITHUB_API_BASE = "https://api.github.com"

class GitHubIssueAdapter(DataSourceAdapter):
    """Pulls closed, PR-linked issues from a target GitHub repo."""

    def __init__(self, config : CollectorConfig):
        self.config = config
        self.session = requests.Session()
        headers = {"Accept": "application/vnd.github+json"}
        if config.github_token:
            headers["Authorization"] = f"Bearer {config.github_token}"
            logger.info("Using authenticated Github requests (5000 req?hr limit).")
        else:
            logger.warning(
                "No GITHUB_TOKEN found, running unauthenticated (60 req/hr limit)"
            )
        self.session.headers.update(headers)
        self._already_collected = self._load_checkpoint()
    
    def source_name(self) -> str:
        return "github_issue"
    
    # ---------- public API ----------

    def fetch_records(self, limit) -> Iterable[RawIssueRecord]:
        collected = 0
        scanned = 0
        next_url = self._get_first_page_url()
        while collected < limit and next_url:
            issues, next_url = self._request_with_backoff(next_url)

            if not issues:
                logger.info("No more issues returned by API - stopping")
                break

            for issue in issues:
                if collected >= limit:
                    break
                scanned += 1
                issue_number = str(issue["number"])
                logger.debug(f"Scanning issue #{issue_number}")

                if issue_number in self._already_collected:
                    continue

                if "pull_request" in issue:
                    continue

                try:
                    record = self._build_record(issue)
                except Exception as e:
                    logger.error(f"Skipping issue #{issue_number} due to error :{e}")
                    continue

                if record is None:
                    continue

                self._save_checkpoint(issue_number)
                collected += 1
                yield record

            
        logger.info(f"Finished collection. Scanned {scanned} issues, collected {collected} records.")


    # ---------- internals ----------

    def _get_first_page_url(self) -> str:
        url = f"{GITHUB_API_BASE}/repos/{self.config.repo_owner}/{self.config.repo_name}/issues"
        params = {"state": "closed", "per_page": 50, "sort": "updated", "direction": "desc"}
        # build the full URL with params baked in, since subsequent calls use the Link url directly with no extra params
        from requests.models import PreparedRequest
        req = PreparedRequest()
        req.prepare_url(url, params)
        return req.url

    def _request_with_backoff(self, url: str, params: dict | None = None) -> tuple[list[dict], Optional[str]]:
        for attempt in range(1, self.config.max_retries +1):
            response = self.session.get(url, params=params, timeout=self.config.request_timeout_secs)

            remaining = response.headers.get("X-RateLimit-Remaining")
            if remaining is not None and int(remaining) <=1:
                reset_at = int(response.headers.get("X-RateLimit-Reset",time.time()+60))
                sleep_secs = max(reset_at - int(time.time()),1)
                logger.warning(f"Rate limit nearly exhausted. Sleeping {sleep_secs}s until reset.")
                time.sleep(sleep_secs)

            if response.status_code == 200:
                next_url = response.links.get("next",{}).get("url")
                return response.json(),next_url
            
            if response.status_code in (403, 429):
                backoff = 2** attempt
                logger.warning(
                    f"Got {response.status_code} on attempt {attempt}/{self.config.max_retries}."
                    f"Backing off {backoff}s"
                )
                time.sleep(backoff)
                continue

            logger.error(f"Unexpectedf status {response.status_code} : {response.text[:200]}")
            response.raise_for_status()
        
        logger.error(f"Giving up after {self.config.max_retries} retries on {url}")
        return [], None 
    
    def _build_record(self, issue:dict)-> Optional[RawIssueRecord]:
        issue_number = issue["number"]
        body = issue.get("body") or ""
        linked_pr_number, linked_pr_diff_url = self._get_linked_pr_from_timeline(issue_number)
        if linked_pr_number is None:
            return None
        comments = self._get_comments(issue_number)
        return RawIssueRecord(
            source = self.source_name(),
            source_id = str(issue_number),
            title = issue.get("title", ""),
            body=body,
            state = issue.get("state","closed"),
            labels = [label["name"] for label in issue.get("labels",[])],
            created_at = issue.get("created_at"),
            closed_At = issue.get("closed_at"),
            linked_fix_id= str(linked_pr_number),
            linked_fix_diff_url = linked_pr_diff_url,
            comments=comments,
            url = issue.get("html_url",""),
            collected_At=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),

        )
    
    def _get_linked_pr_from_timeline(self, issue_number:int)->tuple[Optional[int], Optional[str]]:
        """Looks at the issue's timeline for a cross-referenced, merged PR
            that closes this issue. Returns (pr_number, diff_url) or (None, None)
            if no merged linked PR is found.
            """
        url = (
                f"{GITHUB_API_BASE}/repos/{self.config.repo_owner}/{self.config.repo_name}"
                f"/issues/{issue_number}/timeline"
            )
        events, _ = self._request_with_backoff(url, params={"per_page" :100})

        for event in events:
            if event.get("event") != "cross-referenced":
                continue

            source_issue = event.get("source",{}).get("issue",{})
            pull_request = source_issue.get("pull_request")

            if pull_request is None:
                continue
            if pull_request.get("merged_at") is None:
                continue
            pr_number = source_issue.get("number")
            if pr_number is None:
                continue

            diff_url = (f"https://github.com/{self.config.repo_owner}/{self.config.repo_name}/pull/{pr_number}.diff")

            return pr_number, diff_url
        
        return None,None

    

    def _get_comments(self, issue_number: int) -> list[dict]:
        url = (
            f"{GITHUB_API_BASE}/repos/{self.config.repo_owner}/{self.config.repo_name}"
            f"/issues/{issue_number}/comments"
        )
        raw_comments, _ = self._request_with_backoff(url, params={"per_page": 30})
        return [
            {
                "author": c.get("user", {}).get("login", "unknown"),
                "body": c.get("body", ""),
                "created_at": c.get("created_at", ""),
            }
            for c in raw_comments
        ]
    
    # ---------- checkpointing ----------

    def _load_checkpoint(self)->set[str]:
        if not os.path.exists(self.config.checkpoint_path):
            return set()
        with open(self.config.checkpoint_path,"r") as f:
            try:
                return set(json.load(f))
            except json.JSONDecodeError:
                logger.warning("checkpoint file unreadable - starting fresh")
                return set()
    
    def _save_checkpoint(self, issue_number :str)->None:
        self._already_collected.add(issue_number)
        os.makedirs(os.path.dirname(self.config.checkpoint_path), exist_ok=True)
        with open(self.config.checkpoint_path, "w") as f:
            json.dump(sorted(self._already_collected),f)



