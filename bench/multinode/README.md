# Two-node kill test on rented boxes

Kill a whole Ray node mid-run and measure how the coordinator recovers.
No harness code changed for this stage; where the harness's current
behavior limits the scenario, that is documented below ("Known
failures") rather than patched.

Cluster shape: two Linux GPU boxes (Lambda or equivalent), Python 3.11,
the pinned `requirements.txt` (`ray==2.50.0`, `vllm==0.15.0`). Model:
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` - 1.1B params, ~2.2 GB fp16, fits
a 24 GB A10 comfortably. Everything below runs inside tmux.

## Firewall / networking (Lambda)

Required between the two nodes, TCP:

| Port        | What                                   |
|-------------|----------------------------------------|
| 6379        | GCS (head; workers join here)          |
| 8265        | Dashboard (head)                       |
| 10001       | Ray Client server (head; unused by the scripts but part of the standard set) |
| 10002-10100 | Worker/raylet ports (both nodes; pinned via `--min/max-worker-port`) |

Use the **private IPs** for `--node-ip-address` / `--address`: Lambda's
firewall applies to the public interface, so node-to-node traffic over
the private network in the same region usually bypasses it entirely.
Only open these ports in the public firewall if you must join nodes
over public IPs (don't - 6379 and 8265 are unauthenticated). To view
the dashboard, prefer an SSH tunnel: `ssh -L 8265:localhost:8265 head`.

## Sequence

Wall-time estimates in brackets; record your actuals.

1. **Sync the repo to both boxes** [1 min]:

       tar --exclude=.git --exclude=__pycache__ --exclude=.pytest_cache \
           --exclude='results/*.jsonl' -czf - . \
         | ssh ubuntu@<box> "mkdir -p ray-eval-harness && tar -xzf - -C ray-eval-harness"

2. **Head** [5-10 min: venv + pinned deps ~3-5 min, model download
   ~2.2 GB, ray start seconds]:

       cd ray-eval-harness
       bash bench/multinode/setup_head.sh <head-private-ip>

   Optional shared cache: if both boxes attach one shared filesystem,
   pass it as `HF_HOME` (2nd arg) on both nodes and the model downloads
   once. The scripts export `HF_HOME` **in the same shell that runs
   `ray start`** - actors inherit env from the raylet, not the driver,
   so exporting it only where you run `main.py` is not enough.

3. **Worker node** [5-10 min, same breakdown]:

       cd ray-eval-harness
       bash bench/multinode/setup_worker.sh <head-private-ip> <own-private-ip>

4. **Capture `ray status` before the run** (run.sh also snapshots it to
   `logs/ray_status_before.txt`). Expect 2 nodes and the full GPU count.

5. **Start the run on the head** [engine load ~1-2 min/actor in
   parallel, then the run; see estimates below]:

       bash bench/multinode/run.sh                       # workers = total GPUs
       # or, on a cluster with exactly WORKERS+STANDBY-1 GPUs:
       WORKERS=$((GPUS-1)) bash bench/multinode/run.sh   # see Known failure 1

   Note the `T0 epoch` line it prints.

6. **Check placement before the kill** (~T+60s): read
   `logs/actors_t60.txt` (or `ray list actors --detail`). Confirm:
   - the `ResultsAggregator` actor is on the **head** node, and
   - the victim node hosts a **busy** `VLLMWorker` (on the victim:
     `nvidia-smi` - a primary shows sustained GPU util; an idle,
     model-loaded standby sits near 0%).
   If either check fails, abort and re-launch (Known failure 2).

7. **Kill the worker node at T+120s** - from the head, so the kill
   timestamp lands on the head's clock:

       bash bench/multinode/kill_node.sh <worker-private-ip> <T0-epoch> 120

   What you should observe is documented in `kill_node.sh`'s header:
   the **error path** (refs resolve as errored with
   RayActorError ~1-30s after the kill, once GCS declares the node
   dead) is the expected primary; the **hang path** (per-ref eviction
   at `max(120, 2*60+30) = 150s`) is the backstop if detection stalls.
   Then: force-kill of the old handle -> **standby promotion in O(1)**
   ("replaced by standby") -> the failed batch's two halves retried
   per-task-budget within ~1s. The background standby refill needs a
   free GPU; with the dead node's GPU gone it times out (120s x 3) and
   the pool "stays one smaller" - expected, non-blocking.

8. **Capture `ray status` after** (run.sh snapshots
   `logs/ray_status_after.txt`): expect 1 node, and 1 dead node listed.

9. **Measure** [seconds]:

       python bench/multinode/measure.py --logs bench/multinode/logs \
           --json bench/multinode/logs/measure.json

## Run-length estimate

20,000 tasks, batch 64 (vLLM default), `MAX_NEW_TOKENS=80`, TinyLlama
on A10s: expect very roughly 1-4s per batch per worker, i.e. **~10-25
min end-to-end with 2 GPU workers** (one lost mid-run). The kill at
T+120s lands well inside the run. If your first run finishes before
T+120s (fast boxes), it will only be because engine load ate the lead
time - the kill delay is measured from `run.sh` T0, which includes the
~1-2 min engine load, so batches have typically been flowing for less
than 120s at kill time. Adjust the delay argument if needed.

## Numbers to record

From `measure.py` (also in `measure.json`):

- tasks/s **before**, **during**, **after** the kill
- **kill -> first successful batch on a replacement** (bounds are
  +-the 2s sampler interval; cross-check the "replaced by standby"
  log timestamp, also reported)
- **completed vs submitted** - must be `20000 / 20000 (100.00%)` PASS,
  with 0 duplicate task_ids (one-terminal-state-per-task invariant) -
  and the count of terminal FAILURE rows (0 on a clean recovery)
- **tasks retried as a result of the kill** (and the event count; a
  task re-queued twice counts twice, by construction)

From the run summary block in `run.log`: `Throughput`, `Wall time`,
`retry_splits`, the `standby` dict (expect `promotions: 1`,
`refill_failures: 3`, `final_pool_size: 0`), and per-worker stats.
Plus both `ray status` snapshots and `actors_t60.txt`.

## Known failure 1: `--workers = total GPUs` + `--standby 1` cannot start

Standbys are built by the same factory as primaries
(`coordinator._worker_factory`): on vLLM each claims `num_gpus=1`, and
the 120s init barrier waits on **workers + standbys** before the first
batch. With `--workers` equal to the cluster's GPU total, the standby
is unschedulable and the run aborts with the coordinator's own
diagnosis: `RuntimeError: Workers did not become healthy within 120s.
Most common cause: --workers (N) + --standby (1) exceeds available
GPUs...`. This is the documented standby-pool resource cost of `--standby`
(`main.py --help`: "budget for --workers + --standby actors"), so the
stage's literal flag combination is unsatisfiable on any cluster.
`run.sh` fails this preflight *before* launching. Resolutions:
provision `WORKERS + STANDBY` GPUs (e.g. a 2-GPU box plus a 1-GPU box
for `--workers 2 --standby 1`), or run `WORKERS=$((GPUS-1))`.
Verified against a real `ray==2.50.0`: a `num_gpus=1` actor also
requires 1 CPU for placement, so there is no `ray start` resource
trick (e.g. `--num-cpus=0` on a node) that reserves a GPU for the
standby without also making primaries unschedulable there.

## Known failure 2: actor placement is uncontrolled

The harness sets no scheduling strategy on any actor, so Ray's default
(hybrid packing) decides placement, and two placements break the demo.
Described here, not patched (a placement-group / node-affinity policy
is a future stage):

- **Standby on the victim node.** The kill takes the standby with it.
  If a primary also died: `_replace_worker` promotes the standby with
  **no health check at promotion** (it is a pure handle swap), the
  first batch sent to the dead handle errors, the health check is
  unreachable, the worker is replaced *again*, and the pool - now
  empty - falls back to `_blocking_replace`. With no free GPU left,
  that blocks the single-threaded driver for ~372s (3 attempts x 120s
  `ray.get` timeout + 2/4/6s sleeps) with zero dispatch or harvesting,
  then marks the slot dead. Each lap also burns one retry per affected
  batch (halved by split-retry), so with `max_retries=2` a batch that fails on
  the dead primary and again on the dead promoted standby is one
  failure from terminal. The run still finishes on the survivors and
  every task reaches exactly one terminal state, but tasks *can* end
  as terminal failures and completion-vs-success drops below a clean
  100%/100%. If only the (idle) standby died, nothing fails at all,
  the coordinator never notices (the pool is not health-monitored),
  and no promotion happens - a valid run, but not the recovery demo.
- **ResultsAggregator on the victim node.** Its `record_batch` calls
  are fire-and-forget refs drained only at end of run, so every result
  recorded after the kill is silently lost, the end-of-run
  `ray.get(self._write_refs)` raises `RayActorError`, the run crashes
  without a summary, and the JSONL is stranded (partial) in the dead
  node's filesystem - the coordinator has no aggregator fault
  tolerance. This also means "JSONL on the head" is only guaranteed by
  *checking*: `run.sh` passes an absolute `--output` under the head's
  repo, but the path is opened by the aggregator actor on whatever
  node hosts it.

Both are why step 6 exists: verify placement at ~T+60s and re-roll the
launch (cheap - one engine-load cycle) until the aggregator is on the
head and the victim node hosts a busy primary. In practice Ray's
packing usually lands the CPU-only aggregator on the head and spreads
GPU actors, but "usually" is exactly what this runbook is not allowed
to assume.
