import modal, time, pprint, statistics as st

app = modal.App("timetest-demo")
data = modal.Dict.from_name("timetest", create_if_missing=True)


def epoch_ns() -> int:
    """UTC wall-clock in integer nanoseconds."""
    return time.time_ns()


@app.cls(image=modal.Image.debian_slim())
class Engine:
    @modal.enter()
    def startup(self):
        # record when this *specific* container came alive
        data["ctr_started_ns"] = epoch_ns()

    @modal.method()
    def do_stuff(self, host_pre_call_ns: int) -> int:
        """
        host_pre_call_ns is sent by the host right before .remote().
        We store both host and container clocks to cross-check skew.
        """
        now_ns = epoch_ns()
        # Persist both numbers so any later container can read them
        data["ctr_method_ns"] = now_ns
        data["host_pre_call_ns_echo"] = host_pre_call_ns
        return now_ns  # also returned to host for convenience

    @modal.exit()
    def shutdown(self):
        data["ctr_destroyed_ns"] = epoch_ns()


@app.local_entrypoint()
def main():
    # ===== host side =====
    host_pre_call_ns = epoch_ns()
    data["host_pre_call_ns"] = host_pre_call_ns

    engine = Engine()
    # invoke the remote method, sending the host timestamp along
    ctr_method_ns = engine.do_stuff.remote(host_pre_call_ns)
    host_post_call_ns = epoch_ns()
    data["host_post_call_ns"] = host_post_call_ns

    # --------  pretty-print results  --------
    def ns_to_s(n):
        return f"{n / 1e9:,.6f} s"

    def delta(a, b):
        return ns_to_s(data[a] - data[b])

    print("\n=== Cross-container timing ===")
    print(
        f"Host pre-call  ➔ container method : {delta('ctr_method_ns', 'host_pre_call_ns')}"
    )
    print(
        f"Container up   ➔ method executed  : {delta('ctr_method_ns', 'ctr_started_ns')}"
    )
    print(
        f"Method return  ➔ host post-call   : {delta('host_post_call_ns', 'ctr_method_ns')}"
    )

    # Check clock skew (container – host) at call time
    skew_ns = data["ctr_method_ns"] - data["host_pre_call_ns_echo"]
    print(f"Observed host ⇄ container clock skew : {skew_ns / 1e6:.3f} ms")
