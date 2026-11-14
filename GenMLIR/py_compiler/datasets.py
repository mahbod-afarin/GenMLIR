from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GraphInfo:
    path: str
    num_nodes: int
    num_edges: int
    directed: bool


def _read_text(path: str, max_bytes: int = 2_000_000) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read(max_bytes)


def parse_snap_graph_header(path: str) -> Tuple[Optional[int], Optional[int], Optional[bool]]:
    """Read node count, edge count and directedness from a SNAP-style header.

    SNAP edge lists start with comment lines such as

        # Directed graph (each unordered pair of nodes is saved once): ...
        # Nodes: 5242 Edges: 28980

    Returns (num_nodes, num_edges, directed); a field is None if not found.
    """
    head = _read_text(path, max_bytes=20000)
    num_nodes = None
    num_edges = None
    directed = None

    m = re.search(r"Nodes:\s*(\d+)\s+Edges:\s*(\d+)", head)
    if m:
        num_nodes = int(m.group(1))
        num_edges = int(m.group(2))

    if "Directed graph" in head:
        directed = True
    if "Undirected graph" in head:
        directed = False

    return num_nodes, num_edges, directed


def infer_graph_num_nodes_from_edges(path: str, *, assume_1_based: bool = True) -> Tuple[int, int]:
    """Count nodes and edges by scanning the edge list.

    Used for files without a SNAP header, such as the OSM graph.
    """
    max_id = 0
    edges = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                u = int(parts[0])
                v = int(parts[1])
            except ValueError:
                continue
            edges += 1
            max_id = max(max_id, u, v)
    if assume_1_based:
        return max_id, edges
    return max_id + 1, edges


def load_graph_info(path: str, *, directed_default: bool = False) -> GraphInfo:
    """Return node and edge counts for a graph file."""
    n, e, directed = parse_snap_graph_header(path)

    if n is None or e is None:
        n2, e2 = infer_graph_num_nodes_from_edges(path)
        n = n or n2
        e = e or e2

    if directed is None:
        directed = directed_default

    return GraphInfo(path=path, num_nodes=int(n), num_edges=int(e), directed=bool(directed))


def read_fasta_one_sequence(path: str) -> str:
    """Return the concatenated sequence lines of a FASTA file, upper-cased."""
    seq: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line)
    return "".join(seq).upper()


def read_fastq_records(path: str, *, max_reads: Optional[int] = None) -> List[Tuple[str, str]]:
    """Read (name, sequence) records from a FASTQ file."""
    recs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            header = f.readline()
            if not header:
                break
            seq = f.readline()
            f.readline()  # '+' separator
            qual = f.readline()
            if not qual:
                break
            h = header.strip()
            name = h[1:].split()[0] if h.startswith("@") else h.split()[0]
            s = seq.strip().upper()
            if s:
                recs.append((name, s))
                if max_reads is not None and len(recs) >= max_reads:
                    break
    return recs


def reverse_complement(seq: str) -> str:
    table = str.maketrans({"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"})
    return seq.upper().translate(table)[::-1]


@dataclass(frozen=True)
class MafHit:
    query_name: str
    query_strand: str
    ref_start: int
    ref_size: int
    ref_strand: str
    ref_src_size: int


def parse_maf_first_hits(path: str) -> Dict[str, MafHit]:
    """Parse a MAF file and return the first hit of each query.

    Each alignment block has the form

        a
        s ref     <start> <size> <strand> <srcSize> <aligned text>
        s <query> <start> <size> <strand> <srcSize> <aligned text>
    """
    hits: Dict[str, MafHit] = {}
    cur_ref = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("a"):
                cur_ref = None
                continue
            if not line.startswith("s "):
                continue

            parts = line.split()
            if len(parts) < 7:
                continue
            src = parts[1]
            start = int(parts[2])
            size = int(parts[3])
            strand = parts[4]
            src_size = int(parts[5])

            if src == "ref":
                cur_ref = (start, size, strand, src_size)
                continue

            if cur_ref is None or src in hits:
                continue
            ref_start, ref_size, ref_strand, ref_src_size = cur_ref
            hits[src] = MafHit(
                query_name=src,
                query_strand=strand,
                ref_start=ref_start,
                ref_size=ref_size,
                ref_strand=ref_strand,
                ref_src_size=ref_src_size,
            )

    return hits
