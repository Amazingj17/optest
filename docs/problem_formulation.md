# Problem formulation

For a DAG `G=(V,E)`, an edge `(p,i)` means that task `i` can be selected only
after predecessor `p` has received a placement. A complete action is
`(ready_task, feasible_node)`. The default ready semantics are offline list
scheduling; selection does not wait for wall-clock completion.

For task `i` on node `j`, the default execution model is
`T_exec(i,j)=workload_i/speed_j`. For a predecessor placed on another node,
`T_comm(p,i)=data_size(p,i)/bandwidth(node_p,j)+latency(node_p,j)`; same-node
communication is exactly zero. `DRT` is the maximum predecessor arrival time.
The simulator uses the first idle timeline gap after DRT for EST and sets
`EFT=EST+T_exec`. The objective is `C_max=max_i finish_i`.

Competition-facing evaluation is the paired ratio
`RL_makespan/HEFT_makespan` on the identical Scenario and simulator. The
default per-decision reward is the negative HEFT-normalized increase in partial
makespan. Communication weight and potential-based shaping are opt-in.
For this undiscounted telescoping objective, the recommended high- and
low-level discount factors are both `1.0`; using a smaller gamma changes the
weight assigned to early versus late makespan increments and is therefore an
objective-shaping choice rather than a neutral default.
