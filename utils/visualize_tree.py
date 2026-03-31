from pyvis.network import Network

def export_tree_visualization(tree, tokenizer, output_path="prefix_tree.html"):
    """
    Generates an interactive HTML visual representation of the prefix tree.
    Unrolls multi-token nodes so every single generated token gets its own box,
    perfectly aligning the graph depth with the sequence timesteps.
    """
    net = Network(height="100vh", width="100%", directed=True, layout=True)

    # Force a Left-to-Right hierarchical layout. 
    # levelSeparation pushes the timesteps apart cleanly.
    net.set_options("""
    var options = {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "levelSeparation": 150,
          "nodeSpacing": 60
        }
      },
      "physics": {
        "enabled": false
      },
      "nodes": {
        "font": {
            "size": 14
        }
      }
    }
    """)

    def walk_and_add(node, parent_vis_id=None):
        # 1. Handle the Root Node
        if not node.token_ids:
            vis_id = str(node.node_id)
            net.add_node(vis_id, label="<ROOT>", title="Root Node", color="#D3D3D3", shape="box")
            if parent_vis_id is not None:
                net.add_edge(parent_vis_id, vis_id)
            
            # Recurse for children
            for child in node.children.values():
                walk_and_add(child, vis_id)
            return

        # 2. Unroll the token_ids list into individual "virtual" nodes
        L = len(node.token_ids)
        prev_vis_id = parent_vis_id

        for i, tid in enumerate(node.token_ids):
            # Create a unique ID for each unrolled token (e.g., "5_0", "5_1")
            vis_id = f"{node.node_id}_{i}"
            
            # Decode just this single token
            token_text = tokenizer.decode([tid])
            
            # Use repr() to visually expose leading spaces or newlines (e.g., ' Hello' vs 'Hello')
            display_text = repr(token_text).strip("'") 
            
            # Only the final token in the node inherits the node's true terminal state
            is_last = (i == L - 1)
            
            # Determine if this is a cascade-pruned internal node vs a directly-pruned leaf.
            # A cascade-pruned node has children (it was a branch point) but all its
            # descendants were pruned, so it got marked is_pruned by the upward cascade.
            node_is_pruned = getattr(node, 'is_pruned', False)
            is_cascade_pruned = node_is_pruned and bool(node.children)
            is_direct_pruned = node_is_pruned and not node.children

            if node_is_pruned:
                # Color ALL unrolled tokens of a pruned node to make
                # dead subtrees visually obvious at a glance.
                if is_cascade_pruned:
                    color = "#FFB347"          # Orange — cascade-pruned ancestor
                    state = "Cascade Pruned (all children dead)"
                else:
                    color = "#F08080"          # Red — directly pruned leaf
                    state = "Pruned"
            elif is_last:
                if getattr(node, 'is_eos', False):
                    color = "#ADD8E6"
                    state = "EOS Complete"
                elif node.is_active:
                    color = "#90EE90"
                    state = "Active Leaf"
                else:
                    color = "#D3D3D3"
                    state = "Branched (Parent)"
            else:
                color = "#D3D3D3"
                state = "Intermediate (Greedy Extend)"

            # Calculate the absolute timestep (depth) of this specific token
            token_depth = node.depth - L + 1 + i

            # Build the tooltip
            hover_info = (
                f"Token Depth: {token_depth}\n"
                f"Original Node ID: {node.node_id}\n"
                f"State: {state}\n"
            )
            # We only have the accurately tracked cumulative log prob at the end of the node block
            if is_last:
                hover_info += f"Cum. Log-Prob: {node.cumulative_log_prob:.3f}\n"

            # Add the unrolled token to the graph
            net.add_node(vis_id, label=display_text, title=hover_info, color=color, shape="box")

            # Connect it to the previous token (or the parent node)
            if prev_vis_id is not None:
                net.add_edge(prev_vis_id, vis_id)

            prev_vis_id = vis_id

        # 3. Connect the actual child nodes to the very last virtual token we just drew
        for child in node.children.values():
            walk_and_add(child, prev_vis_id)

    # Start the recursion, passing None as the initial parent
    walk_and_add(tree.root)
    
    net.write_html(output_path)
    print(f"Unrolled interactive tree visualization saved to {output_path}")