from typing import List, Any


def merge_lists(lists: List[List[Any]], recursive=False) -> List[Any]:
    lists = [item for list_ in lists for item in list_]
    if recursive and len(lists) > 0 and isinstance(lists[0], list):
        return merge_lists(lists)
    return lists