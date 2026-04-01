from pyvis.network import Network

def export_tree_visualization(tree, tokenizer, output_path="prefix_tree.html"):
    """
    Generates a sleek, compressed HTML visualization of the prefix tree.
    Only shows branch points and terminal states. Hover over nodes to see full details.
    """
    # 1. Dark Theme & Sleek Canvas
    net = Network(
        height="100vh", width="100%", directed=True, layout=True, 
        bgcolor="#0d1117", font_color="#e6edf3"
    )

    # 2. Advanced Layout & Physics Options
    net.set_options("""
    var options = {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "levelSeparation": 280,
          "nodeSpacing": 80
        }
      },
      "physics": {
        "enabled": false
      },
      "edges": {
        "color": {"color": "#444c56", "highlight": "#58a6ff"},
        "smooth": {
            "type": "cubicBezier", 
            "forceDirection": "horizontal", 
            "roundness": 0.5
        },
        "width": 2
      },
      "nodes": {
        "shape": "box",
        "borderWidth": 2,
        "font": {"size": 15, "face": "monospace", "multi": "html"},
        "margin": 12,
        "shadow": {"enabled": true, "color": "rgba(0,0,0,0.8)", "size": 10}
      }
    }
    """)

    def walk_and_add(node, parent_vis_id=None):
        vis_id = str(node.node_id)
        
        # ── 1. Root Node ─────────────────────────────────────────────────────
        if not getattr(node, 'token_ids', None):
            net.add_node(
                vis_id, 
                label="<b>[ ROOT ]</b>", 
                title="Start of generation", 
                color={"background": "#21262d", "border": "#30363d"}, 
                font={"color": "#8b949e"}
            )
            for child in node.children.values():
                walk_and_add(child, vis_id)
            return

        # ── 2. Decode Text & Compress for Label ──────────────────────────────
        text = tokenizer.decode(node.token_ids)
        display_text = repr(text).strip("'")
        
        # Truncate for the visual box to keep the graph clean
        max_len = 25
        if len(display_text) > max_len:
            short_text = display_text[:max_len] + "..."
        else:
            short_text = display_text
            
        # ── 3. Determine Node State & Mentat Colors ──────────────────────────
        is_leaf = not bool(node.children)
        is_pruned = getattr(node, 'is_pruned', False)
        is_eos = getattr(node, 'is_eos', False)
        
        if is_pruned:
            # Dead End (Red)
            bg_color = "#3a1d1d"
            border_color = "#f85149"
            state_label = "Pruned ❌"
        elif is_leaf and is_eos:
            # Successfully Finished (Green)
            bg_color = "#17361e"
            border_color = "#2ea043"
            state_label = "Complete ✅"
        elif is_leaf and getattr(node, 'is_active', False):
            # Still generating (Cyan)
            bg_color = "#0d2b3e"
            border_color = "#38bdf8"
            state_label = "Active Leaf ⏳"
        else:
            # Branch Point (Spice Orange)
            bg_color = "#3d220b"
            border_color = "#f97316"
            state_label = "Branch Point 🔀"

        # ── 4. Rich HTML Tooltip (The "Expand" Feature) ──────────────────────
        hover_info = (
            f"<b>Node ID:</b> {node.node_id}<br>"
            f"<b>State:</b> {state_label}<br>"
            f"<b>Tokens:</b> {len(node.token_ids)}<br>"
            f"<b>Depth:</b> {node.depth}<br>"
            f"<b>Cum. Log-Prob:</b> {getattr(node, 'cumulative_log_prob', 0.0):.4f}<br>"
            f"<hr>"
            f"<b>Full Text Sequence:</b><br>"
            f"<div style='max-width: 300px; white-space: pre-wrap;'><i>{display_text}</i></div>"
        )

        # ── 5. Add to Graph ──────────────────────────────────────────────────
        net.add_node(
            vis_id, 
            label=f"<b>{short_text}</b>", 
            title=hover_info, 
            color={
                "background": bg_color, 
                "border": border_color, 
                "highlight": {"background": bg_color, "border": "#ffffff"}
            },
        )

        if parent_vis_id is not None:
            net.add_edge(parent_vis_id, vis_id)

        # Recurse
        for child in node.children.values():
            walk_and_add(child, vis_id)

    # Boot the recursion
    walk_and_add(tree.root)
    
    net.write_html(output_path)
    print(f"Mentat tree visualization saved to {output_path}")