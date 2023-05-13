from typing import TextIO, Iterator, Tuple, List, Union, Sequence, Optional
from enum import IntEnum
from collections.abc import Sequence as Seq
import itertools

from abctk.obj.ID import RecordID, SimpleRecordID, RecordIDParser

# TODO: introduce TypeGuard (> 3.10)
Tree = Union[str, Sequence["Tree"]]
def is_terminal(tree) -> bool:
    return isinstance(tree, str)

def inspect_pre_terminal(tree) -> Optional[tuple[str, str]]:
    if is_terminal(tree):
        return None
    elif (res := inspect_nonterminal(tree)):
        if len(res[1]) == 1 and isinstance(res[1][0], str):
            return res[0], res[1][0]
        else:
            return None
    else:
        return None

def inspect_nonterminal(tree) -> Optional[Tuple[str, Sequence[Tree]]]:
    if is_terminal(tree):
        return None
    elif isinstance(tree, Seq) and len(tree) > 0:
        label = tree[0]
        if isinstance(label, str):
            return tree[0], tree[1:]
        else:
            return None
    else:
        return None

def get_label(tree: Tree) -> str:
    if is_terminal(tree):
        return tree # type: ignore
    elif (res := inspect_nonterminal(tree)):
        return res[0]
    else:
        raise TypeError
    
def healthcheck(tree, deep: bool = False) -> bool:
    return (
        is_terminal(tree)
        or (
            bool(res := inspect_nonterminal(tree))
            and (
                not deep or all(
                healthcheck(child, deep) for child in tree[1:]
                )
            )
        )
    )

def is_comment(tree: Tree) -> bool:
    return bool(
        not is_terminal(tree)
        and (res := inspect_nonterminal(tree))
        and res[0] == "COMMENT"
    )

def merge_comments(trees: Sequence[Tree]) -> Iterator[Tree]:
    last_substantial_tree: Optional[Tree] = None
    comment_trees: List[Tree] = []
    comment_trees_init: List[Tree] = []
    for tree in itertools.chain(trees, ( ("", ), )):
        if is_comment(tree):
            if last_substantial_tree:
                comment_trees.append(tree)
            else:
                comment_trees_init.append(tree)
        else:
            # yield the previous tree with comments
            if (
                last_substantial_tree
                and (res := inspect_nonterminal(last_substantial_tree))
            ):
                yield [res[0], *comment_trees_init, *res[1], *comment_trees]

                # clear comments
                comment_trees_init.clear()
                comment_trees.clear()
            elif is_terminal(last_substantial_tree):
                yield ["", *comment_trees_init, last_substantial_tree, *comment_trees] # type: ignore

                # clear comments
                comment_trees_init.clear()
                comment_trees.clear()
            elif last_substantial_tree is None:
                pass
            else:
                raise TypeError(f"Illegal tree found: {last_substantial_tree}")

            # register the tree as the last one
            last_substantial_tree = tree

    yield from comment_trees_init
    yield from comment_trees

class LexCategory(IntEnum):
    PAREN_OPEN = 1
    PAREN_CLOSE = 2
    NODE = 127

def lexer(stream: TextIO) -> Iterator[tuple[LexCategory, str]]:
    current_char: str = stream.read(1)

    buffer: list[str] = []

    while current_char:
        if current_char == "(":
            if buffer:
                yield (LexCategory.NODE, "".join(buffer))
                buffer.clear()
            yield (LexCategory.PAREN_OPEN, current_char)

        elif current_char == ")":
            if buffer:
                yield (LexCategory.NODE, "".join(buffer))
                buffer.clear()

            yield (LexCategory.PAREN_CLOSE, current_char)
        elif current_char == " ":
            if buffer:
                yield (LexCategory.NODE, "".join(buffer))
                buffer.clear()
        else:
            buffer.append(current_char)

        current_char: str = stream.read(1)

    if buffer:
        yield (LexCategory.NODE, "".join(buffer))
        buffer.clear()

def yield_tree(lexemes: Iterator[tuple[LexCategory, str]]) -> Iterator[Tree]:
    subtree_stack: list[list] = []

    for (lexcat, node) in lexemes:
        if lexcat == LexCategory.PAREN_OPEN:
            # (
            new_subtree = []
            if subtree_stack:
                parent_subtree = subtree_stack[-1]
                if parent_subtree:
                    parent_subtree.append(new_subtree)
                else:
                    parent_subtree.extend(("", new_subtree))
            subtree_stack.append(new_subtree)
        elif lexcat == LexCategory.PAREN_CLOSE:
            # )
            if subtree_stack:
                complete_subtree = subtree_stack.pop()

                if not subtree_stack:
                    yield complete_subtree
            else:
                raise IndexError
        else:
            # string node
            if subtree_stack:
                subtree_stack[-1].append(node)
            else:
                yield node

    if subtree_stack:
        raise ValueError("Unclosed tree")

def split_ID_from_tree(
    tree: Tree, 
    ID_parser: Optional[RecordIDParser] = None
) -> Tuple[RecordID, Tree]:
    ID_parser = ID_parser or RecordIDParser()

    if inspect_nonterminal(tree):
        _, roots, maybe_ID = tree[0], tree[1:-1], tree[-1]

        if (
            (res := inspect_pre_terminal(maybe_ID))
            and res[0] == "ID"
        ):
            # If ID found
            ID_parsed = ID_parser.parse(res[1])

            # Reform the tree root
            root_comment_merged = tuple(merge_comments(roots))
            len_root_comment_merged = len(root_comment_merged)

            if len_root_comment_merged == 0:
                return ID_parsed, ""
            elif len_root_comment_merged == 1:
                return ID_parsed, root_comment_merged[0]
            else:
                return ID_parsed, ["", *root_comment_merged]
        else:
            return SimpleRecordID(""), tree
    else:
        return SimpleRecordID(""), tree
