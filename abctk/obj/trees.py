from typing import Callable, NamedTuple, TextIO, Iterator, Tuple, List, Union, Sequence, Optional
from enum import IntEnum
from collections.abc import Sequence as Seq
import itertools
import re

from abctk.obj.ID import RecordID, SimpleRecordID, RecordIDParser

# TODO: introduce TypeGuard (> 3.10)
Tree = Union[str, Sequence["Tree"]]
"""
The union type of types which are counted as a tree.
"""

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

_RE_WHITESPACE = re.compile(r"\s+")
def lexer(stream: TextIO) -> Iterator[tuple[LexCategory, str]]:
    """
    Tokenize trees in the S-expression format to facilitate parsing of them.

    Yields
    ------
    lexical_category : LexCategory
        The type of the token.

    word : str
        The actual string.
    """
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
        elif _RE_WHITESPACE.match(current_char):
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
    """
    Parse trees in the S-expression format.
    Data should be tokenized with :func:`lexer` beforehand.

    Yields
    ------
    lexical_category : LexCategory
        The type of the token.

    word : str
        The actual string.
    """

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
    """
    Split the ID (if there is any) and the content in `tree`.
    The ID is tagged in the way the CorpusSearch project [1]_ recommends.

    Arguments
    ---------
    tree

    ID_parser
        A parser of IDs. A default is used when no instance is provided.

    References
    ----------
    .. [1] https://corpussearch.sourceforge.net/CS-manual/YourCorpus.html#ID
    """
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

def iter_leaves_with_branches(tree: Tree)-> Iterator[Tuple[Tree, ...]]:
    if inspect_nonterminal(tree):
        pointer_stack: List[Tuple[Tree, int, int]] = [(tree, 1, len(tree))]
        while pointer_stack:
            current_node, idx_current_child, count_children = pointer_stack.pop()
            if idx_current_child < count_children:
                pointer_stack.append(
                    (current_node, idx_current_child + 1, count_children)
                )

                current_leftmost_child = current_node[idx_current_child]

                if inspect_nonterminal(current_leftmost_child):
                    pointer_stack.append(
                        (current_leftmost_child, 1, len(current_leftmost_child))
                    )
                else:
                    yield (
                        *(label for label, _, _ in pointer_stack),
                        current_leftmost_child,
                    )
            # else:
                # all children are consumed
                # just discard
    else:
        yield (tree, )

class GRVCell(NamedTuple):
    """
    Represents a cell of an encoded tree.
    Used for :func:`encode_GRV` and :func:`decode_GRV`
    """
    lexeme: str
    lex_cat: str
    height_diff: int
    phrase_cat: str

def encode_GRV(tree: Tree) -> Iterator[GRVCell]:
    """
    Encode `tree` in the way described by [1]_. 
    Relative scale is adopted.

    Notes
    -----
    There must be no unary nodes (except for lexical nodes).
    If any, they must be collapsed beforehand.

    References
    ----------
    .. [1] Gómez-Rodríguez, C., & Vilares, D. (2018). Constituent Parsing as Sequence Labeling. In: Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1314–1324. https://doi.org/10.18653/v1/D18-1162
    """
    iter_leaves: Iterator[tuple[str, ...]] = (
        tuple(get_label(node) for node in branch)
        for branch in iter_leaves_with_branches(tree)
    )
    prev_height: int = 0

    if (current_leaf := next(iter_leaves, None)):
        for next_leaf in iter_leaves:
            match_idx = next(
                (
                    count_common_ancestors
                    for count_common_ancestors, (current_node, next_node)
                    in enumerate(zip(current_leaf[:-2], next_leaf[:-2]))
                    if current_node != next_node
                ),
                len(current_leaf) - 2
            )

            yield GRVCell(
                current_leaf[-1], current_leaf[-2],
                (match_idx - prev_height),
                current_leaf[match_idx - 1]
            )
            prev_height = match_idx

            current_leaf = next_leaf
        
        yield GRVCell(
            current_leaf[-1], current_leaf[-2],
            0, "",
        )

def decode_GRV(cells: Iterator[GRVCell]):
    """
    Decode a tree encoded in the way described by [1]_. 
    Relative scale is assumed.

    Notes
    -----
    Collapsed unary nodes is to be expanded manually after the decoding.

    References
    ----------
    .. [1] Gómez-Rodríguez, C., & Vilares, D. (2018). Constituent Parsing as Sequence Labeling. In: Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1314–1324. https://doi.org/10.18653/v1/D18-1162
    """
    # Initial cell
    initial_cell = next(cells)
    new_node: Tree = ["" ]
    tree_pointer: list[Tree] = [new_node]
    for _ in range(initial_cell.height_diff - 1):
        child: Tree = ["" ]
        tree_pointer[-1].append(child) # type: ignore
        tree_pointer.append(child)
    
    tree_pointer[-1][0] = initial_cell.phrase_cat # type: ignore
    lex_node: Tree = [initial_cell.lex_cat, initial_cell.lexeme]
    tree_pointer[-1].append(lex_node) # type: ignore
    
    for cell in cells:
        if cell.height_diff > 0:
            # grow edges
            for _ in range(cell.height_diff):
                child = ["" ]
                tree_pointer[-1].append(child) # type: ignore
                tree_pointer.append(child)

            tree_pointer[-1][0] = cell.phrase_cat # type: ignore
            tree_pointer[-1].append([cell.lex_cat, cell.lexeme]) # type: ignore
        elif cell.height_diff == 0:
            # adjoint lexeme to the pointer
            # (the relevant node on the last branch)
            tree_pointer[-1].append([cell.lex_cat, cell.lexeme]) # type: ignore
        else:
            # adjoint lexeme to the pointer
            # (the relevant node on the last branch)
            tree_pointer[-1].append([cell.lex_cat, cell.lexeme]) # type: ignore

            # move back the pointer
            tree_pointer = tree_pointer[:cell.height_diff]

            tree_pointer[-1][0] = cell.phrase_cat # type: ignore

    return tree_pointer[0]

def split_lexical_nodes(
    tree: Tree, 
    splitter: Iterator[int],
    lex_filter: Callable[[str], bool] = lambda _: True,
) -> Tree:
    """
    Split lexical nodes of `tree` which match with `lex_filter`.
    `splitter` specifies the number of characters of each split.
    """
    _, result = _split_lexical_nodes_internal(
        tree,
        splitter,
        lex_filter,
    )
    return result

def _split_lexical_nodes_internal(
    tree: Tree, 
    splitter: Iterator[int],
    lex_filter: Callable[[str], bool] = lambda _: True,
) -> Tuple[Iterator[int], Tree]:
    if is_terminal(tree):
        return splitter, tree
    elif (res := inspect_pre_terminal(tree)) and lex_filter(res[1]):
        # a pre-lexical node is found

        lex_cat, lex = res

        len_node_char = len(lex)
        remaining_split_char_len: int = next(splitter)

        if len_node_char > remaining_split_char_len:
            # splitting the lexical node is necessary

            #       ======
            #    ============
            # ===================   --- node
            # ^                     --- current_node_char_pos
            # ****                  --- remaining_split_char_len

            current_node_char_pos: int = 0
            # get the lexical category
            lex_category_part = f"{lex_cat}-PART"

            # do the first splitting
            #       ======
            #    ============
            # ====|==============   --- split-children
            #      ^                --- forward current_node_char_pos
            #      ****             --- retrieve remaining_split_char_len
            split_children = [
                [
                    lex_category_part, 
                    lex[current_node_char_pos:remaining_split_char_len]
                ]
            ]
            current_node_char_pos += remaining_split_char_len
            remaining_split_char_len = next(splitter)

            # do the remaining splitting
            while len_node_char - current_node_char_pos > remaining_split_char_len:
                #       ======
                #    ============
                #     |==============   --- node
                #      ^                --- current_node_char_pos
                #      *******          --- remaining_split_char_len remaining_split_char_len
                split_children.append(
                    [
                        lex_category_part, 
                        lex[current_node_char_pos:(current_node_char_pos + remaining_split_char_len)]
                    ]
                )
                current_node_char_pos += remaining_split_char_len
                remaining_split_char_len = next(splitter)
            # === END WHILE ===

            # append the last piece

            #       ======
            #    ============
            #     |      |    |==   --- node
            #                  ^    --- current_node_char_pos
            #                  ******  --- remaining_split_char_len
            split_children.append(
                [
                    lex_category_part, 
                    lex[current_node_char_pos:]
                ]
            )

            if (diff := remaining_split_char_len - len_node_char + current_node_char_pos) > 0:
                splitter = itertools.chain((diff, ), splitter)

            return splitter, [lex_cat, *split_children]


        elif (diff := remaining_split_char_len - len_node_char) > 0:
            #       ======
            #    ============
            # ==================   --- node
            # *************************  --- remaining_split_char_len

            # no splitting

            # the difference between the two will be carried over
            return itertools.chain( (diff, ), splitter), list(tree)
        else:
            return splitter, list(tree)
    elif not is_comment(tree) and (res := inspect_nonterminal(tree)):
        new_children = []
        for child in res[1]:
            splitter, new_child = _split_lexical_nodes_internal(child, splitter, lex_filter)
            new_children.append(new_child)

        return splitter, [res[0], *new_children]
    else:
        raise TypeError(
            f"Incorrect type {type(tree)} of the argument tree {tree}"
        )