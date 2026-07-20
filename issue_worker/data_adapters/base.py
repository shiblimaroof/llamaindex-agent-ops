from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional

@dataclass
class RawIssueRecord:

    source : str
    source_id : str

    #  --- agent-visible: safe for  retrieval index ---
    title : str
    body : str
    state : str         # open / closed
    labels : list[str] = field(default_factory = list)
    created_at : Optional[str] = None

    # --- grading-only: the answer key, never index this for retrieval ---

    closed_At : Optional[str] = None
    linked_fix_id : Optional[str] = None                # e.g. linked PR number
    linked_fix_diff_url : Optional[str] = None
    comments : list[dict] = field(default_factory=list) # [{author, body, created_at}]

    url : str = ""
    collected_At : str = ""

class DataSourceAdapter(ABC):
    """contract every data source adapter must implement"""

    @abstractmethod
    def fetch_records(self, limit:int) -> Iterator[RawIssueRecord]:
        "Yield normalized records, one at a time, up to 'limit'."
        raise NotImplementedError
    
    @abstractmethod
    def source_name(self) ->str:
        "short identifier for this adapter, used in logs and checkpoints"
        raise NotImplementedError



