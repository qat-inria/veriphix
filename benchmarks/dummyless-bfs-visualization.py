from __future__ import annotations

from collections import deque

import matplotlib.pyplot as plt
import networkx as nx


def odd_pair_bfs_steps(graph: nx.Graph):
    odd_nodes = {v for v in graph.nodes if graph.degree(v) % 2 == 1}

    visited = set()
    pred = {}
    odd_source = {}

    steps = []

    def snapshot(
        title,
        current=None,
        neighbor=None,
        queue=None,
        path=None,
        generators=None,
    ):
        steps.append(
            {
                "title": title,
                "current": current,
                "neighbor": neighbor,
                "queue": list(queue or []),
                "visited": set(visited),
                "pred": dict(pred),
                "odd_source": dict(odd_source),
                "path": list(path or []),
                "generators": list(generators or []),
            }
        )

    generators = []

    snapshot("Initial state")

    for start in odd_nodes:
        if start in visited:
            snapshot(f"Skip odd node {start}: already visited")
            continue

        visited.add(start)
        odd_source[start] = start
        queue = deque([start])

        snapshot(f"Start BFS from odd node {start}", current=start, queue=queue)

        while queue:
            u = queue.popleft()
            snapshot(f"Pop {u} from queue", current=u, queue=queue)

            for w in graph.neighbors(u):
                snapshot(f"Inspect edge {u} → {w}", current=u, neighbor=w, queue=queue)

                if w in visited:
                    snapshot(f"Skip {w}: already visited", current=u, neighbor=w, queue=queue)
                    continue

                pred[w] = u
                visited.add(w)
                queue.append(w)

                snapshot(f"Visit {w}; set pred[{w}] = {u}", current=u, neighbor=w, queue=queue)

                if w in odd_nodes:
                    src = odd_source[u]
                    odd_source[w] = w

                    path = []
                    node = w
                    while node != src:
                        path.append(node)
                        node = pred[node]
                    path.append(src)
                    path.reverse()

                    generators.append((src, w, path))

                    snapshot(
                        f"Reached new odd node {w}; generator R({src}, {w})",
                        current=u,
                        neighbor=w,
                        queue=queue,
                        path=path,
                        generators=generators,
                    )
                else:
                    odd_source[w] = odd_source[u]
                    snapshot(
                        f"{w} is even; inherit odd_source[{w}] = {odd_source[u]}",
                        current=u,
                        neighbor=w,
                        queue=queue,
                        generators=generators,
                    )

    snapshot("Done", generators=generators)
    return steps, odd_nodes


class BFSVisualizer:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.steps, self.odd_nodes = odd_pair_bfs_steps(graph)
        self.i = 0
        self.pos = nx.spring_layout(graph, seed=3)

        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.draw()

    def draw(self):
        self.ax.clear()
        step = self.steps[self.i]

        node_colors = []
        node_sizes = []

        for v in self.graph.nodes:
            if v == step["current"]:
                node_colors.append("tab:blue")
                node_sizes.append(900)
            elif v == step["neighbor"]:
                node_colors.append("tab:orange")
                node_sizes.append(900)
            elif v in step["path"]:
                node_colors.append("tab:green")
                node_sizes.append(850)
            elif v in self.odd_nodes:
                node_colors.append("tab:red")
                node_sizes.append(750)
            elif v in step["visited"]:
                node_colors.append("lightgray")
                node_sizes.append(650)
            else:
                node_colors.append("white")
                node_sizes.append(650)

        edge_colors = []
        widths = []

        path_edges = {tuple(sorted((a, b))) for a, b in zip(step["path"], step["path"][1:], strict=False)}
        for a, b in self.graph.edges:
            e = tuple(sorted((a, b)))
            if e in path_edges:
                edge_colors.append("tab:green")
                widths.append(4)
            elif (
                step["current"] is not None
                and step["neighbor"] is not None
                and e == tuple(sorted((step["current"], step["neighbor"])))
            ):
                edge_colors.append("tab:orange")
                widths.append(3)
            else:
                edge_colors.append("gray")
                widths.append(1.5)

        nx.draw_networkx_edges(
            self.graph,
            self.pos,
            ax=self.ax,
            edge_color=edge_colors,
            width=widths,
        )

        nx.draw_networkx_nodes(
            self.graph,
            self.pos,
            ax=self.ax,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors="black",
        )

        nx.draw_networkx_labels(self.graph, self.pos, ax=self.ax)

        queue_text = "Queue: " + str(step["queue"])
        visited_text = "Visited: " + str(sorted(step["visited"]))
        generators_text = "Generators:\n" + "\n".join(
            f"R({src}, {dst}) via {path}" for src, dst, path in step["generators"]
        )

        self.ax.set_title(
            f"Step {self.i + 1}/{len(self.steps)}\n{step['title']}",
            fontsize=13,
        )

        self.ax.text(
            1.02,
            0.95,
            queue_text + "\n\n" + visited_text + "\n\n" + generators_text,
            transform=self.ax.transAxes,
            va="top",
            fontsize=10,
            family="monospace",
        )

        self.ax.axis("off")
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in ["right", " ", "space"]:
            self.i = min(self.i + 1, len(self.steps) - 1)
            self.draw()
        elif event.key == "left":
            self.i = max(self.i - 1, 0)
            self.draw()
        elif event.key == "q":
            plt.close(self.fig)


if __name__ == "__main__":
    # Example graph
    G = nx.Graph()
    G.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (1, 5),
            (5, 6),
            (6, 7),
            (2, 8),
            (8, 9),
        ]
    )

    vis = BFSVisualizer(G)
    plt.show()
