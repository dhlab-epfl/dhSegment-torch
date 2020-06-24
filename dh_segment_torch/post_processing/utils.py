from typing import List, Any


def recursively_merge(lists: List[List[Any]]) -> List[Any]:
    lists = [item for list_ in lists for item in list_]
    if len(lists) > 0 and isinstance(lists[0], list):
        return recursively_merge(lists)
    return lists