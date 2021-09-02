from git import Repo


def get_commit_hash():
    repo = Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha


def is_dev_dir():
    try:
        Repo(search_parent_directories=True)
    except Exception:
        return True
    return False

def ensure_clean_worktree() -> str:
    r = Repo(search_parent_directories=True)

    diff_is_empty = True
    # Check nothing is staged
    for _ in r.index.diff(r.head.commit):
        diff_is_empty = False
        breakprint(1)

    # Check nothing in work tree
    for _ in r.index.diff(None):
        diff_is_empty = False
        break

    # Check no files untracked
    for _ in r.untracked_files:
        diff_is_empty = False
        break

    if not diff_is_empty:
        raise Exception("There are some uncommited changes! Aborting...")

    return r.head.reference.commit.hexsha