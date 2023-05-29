from typing import Callable, NamedTuple, TextIO, Iterator, Tuple, List, Union, Sequence, Optional, TypeVar, Generic, Iterable, Literal, Any
from enum import IntEnum
import itertools
from io import StringIO
import re

from abctk.obj.ID import RecordID, SimpleRecordID, RecordIDParser

X = TypeVar("X")
class LexCategory(IntEnum):
    PAREN_OPEN = 1
    PAREN_CLOSE = 2
    NODE = 127

_RE_WHITESPACE = re.compile(r"\s+")

class Tree(NamedTuple):
    "A named tuple representing a tree."
    label: Any
    children: Sequence["Tree"] = tuple()

    def __item__(self, index: int) -> "Tree":
        return self.children[index]

    def __len__(self) -> int:
        return len(self.children)

    def __str__(self):
        buffer = StringIO()
        self.print_stream(buffer)
        return buffer.getvalue()

    def solidify(self) -> "Tree":
        """
        Noramlize by replacing all of the child containers with :class:`tuple`s.

        Notes
        -----
        Non-destructive.
        """
        return Tree(
            self.label,
            tuple(child.solidify() for child in self.children)
        )

    def print_stream(self, stream: TextIO, node_printer: Callable[[Any], str] = str):
        if self.is_terminal():
            stream.write(node_printer(self.label))
            return
        else:
            stream.write(f"({node_printer(self.label)}")
            for child in self.children:
                stream.write(" ")
                child.print_stream(stream, node_printer)
            stream.write(")")

    def is_terminal(self) -> bool:
        return not self.children

    def is_nonterminal(self) -> bool:
        return bool(self.children)

    def inspect_unary(self) -> Optional[Tuple[Any, "Tree"]]:
        if len(self.children) == 1:
            return self.label, self.children[0]
        else:
            return None

    def inspect_preterminal(self) -> Optional[tuple[X, X]]:
        "Find if the tree is a pre-terminal subtree. A pre-terminal subtree means a subtree with a unary terminal node."
        if (res := self.inspect_unary()) and res[1].is_terminal():
            return res[0], res[1].label
        else:
            return None

    def is_comment(self) -> bool:
        return self.label == "COMMENT"

    @classmethod
    def lower_comments(cls, trees: Sequence["Tree"]) -> Iterator["Tree"]:
        """
        Lower comment nodes.

        Notes
        -----
        This function is non-destructive.
        A new tree instance is generated.
        """
        last_substantial_tree: Optional[Tree] = None
        comment_trees: List[Tree] = []
        comment_trees_init: List[Tree] = []
        for tree in itertools.chain(trees, ( Tree("", tuple()), )):
            if tree.is_comment():
                if last_substantial_tree:
                    comment_trees.append(tree)
                else:
                    comment_trees_init.append(tree)
            else:
                # yield the previous tree with comments
                if (
                    last_substantial_tree
                    and last_substantial_tree.is_nonterminal()
                ):
                    yield Tree(
                        last_substantial_tree.label,
                        (
                            *comment_trees_init, 
                            *last_substantial_tree.children,
                            *comment_trees
                        ),
                    )
                    # clear comments
                    comment_trees_init.clear()
                    comment_trees.clear()
                elif last_substantial_tree and last_substantial_tree.is_terminal():
                    yield Tree(
                        "",
                        (
                            *comment_trees_init, 
                            last_substantial_tree.label, 
                            *comment_trees,
                        )
                    )

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

    @staticmethod
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

    @classmethod
    def yield_tree_from_lexer(cls, forms: Iterator[tuple[LexCategory, str]]) -> Iterator["Tree"]:
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

        subtree_stack: List[
            Union[Tree, Literal[LexCategory.PAREN_OPEN]]
        ] = []

        for lexcat, word in forms:
            if subtree_stack:
                prev_subtree = subtree_stack.pop()
                if prev_subtree == LexCategory.PAREN_OPEN:
                    if lexcat == LexCategory.PAREN_OPEN:
                        new_subtree = Tree("", [])
                        if subtree_stack:
                            subtree_stack[-1].children.append(new_subtree) # type: ignore
                        subtree_stack.append(new_subtree)
                        subtree_stack.append(lexcat)
                    elif lexcat == LexCategory.PAREN_CLOSE:
                        new_subtree = Tree("", [])
                        if subtree_stack:
                            subtree_stack[-1].children.append(new_subtree) # type: ignore
                        subtree_stack.append(new_subtree)
                    else:
                        new_subtree = Tree(word, [])
                        if subtree_stack:
                            subtree_stack[-1].children.append(new_subtree) # type: ignore
                        subtree_stack.append(new_subtree)
                elif prev_subtree == LexCategory.PAREN_CLOSE:
                    raise Exception("Internal error: closing parenthesis left unprocessed")
                else:
                    if lexcat == LexCategory.PAREN_OPEN:
                        subtree_stack.append(prev_subtree)
                        subtree_stack.append(lexcat)
                    elif lexcat == LexCategory.PAREN_CLOSE:
                        # prev_subtree is closed, so just leave it popped out
                        if not subtree_stack:
                            # prev_subtree is a root, so yield it
                            yield prev_subtree
                    else:
                        # the current node is a terminal node
                        new_subtree = Tree(word, [])
                        # link it to its parent (i.e. prev_subtree)
                        prev_subtree.children.append(new_subtree) # type: ignore
                        # push back prev_subtree
                        subtree_stack.append(prev_subtree)
            else:
                if lexcat == LexCategory.PAREN_OPEN:
                    subtree_stack.append(lexcat)
                elif lexcat == LexCategory.PAREN_CLOSE:
                    raise ValueError("Redundant closing parenthesis")
                else:
                    yield Tree(word, [])
        if subtree_stack:
            raise ValueError("Unclosed tree")

    @classmethod
    def parse_stream(cls, stream: TextIO) -> Iterator["Tree"]:
        yield from cls.yield_tree_from_lexer(cls.lexer(stream))

    def split_ID_from_tree(
        self,
        ID_parser: Optional[RecordIDParser] = None
    ) -> Tuple[RecordID, "Tree"]:
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

        if self.is_nonterminal():
            roots, maybe_ID = self.children[:-1], self.children[-1]

            if (
                (res := maybe_ID.inspect_preterminal())
                and res[0] == "ID"
            ):
                # If ID found
                ID_parsed = ID_parser.parse(res[1]) # type: ignore

                # Reform the tree root
                root_comment_merged = tuple(Tree.lower_comments(roots))
                len_root_comment_merged = len(root_comment_merged)

                if len_root_comment_merged == 0:
                    return ID_parsed, Tree("")
                elif len_root_comment_merged == 1:
                    return ID_parsed, root_comment_merged[0]
                else:
                    return ID_parsed, Tree("", root_comment_merged)
            else:
                return SimpleRecordID(""), self
        else:
            return SimpleRecordID(""), self

    def iter_leaves_with_branches(self)-> Iterator[Tuple]:
        pointer_stack: List[Tuple[Tree, int]] = [(self, 0)]
        while pointer_stack:
            current_node, child_pointer = pointer_stack.pop()

            if current_node.is_terminal():
                yield (
                    *(node.label for node, _ in pointer_stack),
                    current_node.label
                )
            elif child_pointer < len(current_node.children):
                pointer_stack.append(
                    (current_node, child_pointer + 1)
                )
                pointer_stack.append(
                    (current_node.children[child_pointer], 0)
                )
            # else:
                # do nothing

    def iter_terminals(self) -> Iterator:
        pointer_stack = [self]
        while pointer_stack:
            current_node = pointer_stack.pop()
            if current_node.is_terminal():
                yield current_node.label
            elif current_node.is_nonterminal():
                pointer_stack.extend(reversed(current_node.children))

    def replace_terminals(self, terminals: Iterator) -> "Tree":
        """
        Note
        ----
        Non-destructive.
        """
        if self.is_terminal():
            return Tree(next(terminals))
        else:
            new_children = tuple(
                child.replace_terminals(terminals)
                for child in self.children
            )
            return Tree(self.label, new_children)

    def merge_unary_nodes(
        self, 
        concat: Callable[[X, X], X] = lambda x, y: f"{x}☆{y}", # type: ignore
    ) -> "Tree":
        if self.is_terminal() or self.inspect_preterminal():
            return self
        elif (res := self.inspect_unary()):
            label, only_child = res
            label = concat(label, only_child.label)
            return Tree(label, only_child.children).merge_unary_nodes(concat)
        else:
            return Tree(
                self.label,
                tuple(child.merge_unary_nodes(concat) for child in self.children)
            )

    def unfold_unary_nodes(
        self,
        splitter: Callable[[X], List[X]] = lambda x: x.split("☆"), # type: ignore
    ) -> "Tree":
        if self.is_terminal():
            return self
        else:
            label_split = splitter(self.label)

            latest_label = label_split.pop()
            result_tree = Tree(latest_label, self.children)

            while label_split:
                latest_label = label_split.pop()
                result_tree =  Tree(latest_label, (result_tree, ))

            return result_tree

class GRVCell(NamedTuple):
    """
    Represents a cell of an encoded tree.
    """
    form: Any
    lex_cat: Any
    height_diff: int
    phrase_cat: Any

    @classmethod
    def encode(cls, tree: Tree) -> Iterator["GRVCell"]:
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
        iter_leaves: Iterator[tuple] = (
            tuple(branch)
            for branch in tree.iter_leaves_with_branches()
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

                yield cls(
                    current_leaf[-1], current_leaf[-2],
                    (match_idx - prev_height),
                    current_leaf[match_idx - 1]
                )
                prev_height = match_idx

                current_leaf = next_leaf

            yield cls(
                current_leaf[-1], current_leaf[-2],
                0, "",
            )

    @classmethod
    def decode(cls, cells: Iterator["GRVCell"]) -> Tree:
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
        new_node: Tree = Tree("")
        tree_pointer: list[Tree] = [new_node]
        for _ in range(initial_cell.height_diff - 1):
            child: Tree = Tree("")
            tree_pointer[-1].append(child) # type: ignore
            tree_pointer.append(child)
        
        tree_pointer[-1][0] = initial_cell.phrase_cat # type: ignore
        lex_node: Tree = Tree(initial_cell.lex_cat, (initial_cell.form, ))
        tree_pointer[-1].append(lex_node) # type: ignore
        
        for cell in cells:
            if cell.height_diff > 0:
                # grow edges
                for _ in range(cell.height_diff):
                    child = Tree("")
                    tree_pointer[-1].append(child) # type: ignore
                    tree_pointer.append(child)

                tree_pointer[-1][0] = cell.phrase_cat # type: ignore
                tree_pointer[-1].append([cell.lex_cat, cell.form]) # type: ignore
            elif cell.height_diff == 0:
                # adjoint form to the pointer
                # (the relevant node on the last branch)
                tree_pointer[-1].append([cell.lex_cat, cell.form]) # type: ignore
            else:
                # adjoint form to the pointer
                # (the relevant node on the last branch)
                tree_pointer[-1].append([cell.lex_cat, cell.form]) # type: ignore

                # move back the pointer
                tree_pointer = tree_pointer[:cell.height_diff]

                tree_pointer[-1][0] = cell.phrase_cat # type: ignore

        return tree_pointer[0]