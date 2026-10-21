"""Original host and container resource reserves for complete-target runs."""


class Guard:
    def __init__(self, cfg):
        import psutil

        self.cfg = cfg
        self.start_swap = psutil.swap_memory().used
        self.min_ram = psutil.virtual_memory().available
        self.max_rss = 0

    def __call__(self, variables=0):
        import psutil

        avail = psutil.virtual_memory().available
        self.min_ram = min(self.min_ram, avail)
        self.max_rss = max(self.max_rss, psutil.Process().memory_info().rss)
        if (
            avail - self.cfg["bytes_per_transport_variable"] * variables
            < self.cfg["minimum_ram_bytes"]
        ):
            raise RuntimeError("Full-target exact transport would violate the 25GB RAM reserve")
        if psutil.swap_memory().used - self.start_swap > self.cfg["maximum_new_swap_bytes"]:
            raise RuntimeError("New swap exceeds the 2GB research limit")


def container_memory():
    """Read effective cgroup headroom, without allocating or probing a device."""
    from pathlib import Path

    root = Path("/sys/fs/cgroup")

    def number(name):
        path = root / name
        if not path.is_file():
            return None
        value = path.read_text().strip()
        return None if value == "max" else int(value)

    limit, used = (number("memory.max"), number("memory.current"))
    if used is not None:
        swap = number("memory.swap.current")
    else:
        limit, used = (
            number("memory/memory.limit_in_bytes"),
            number("memory/memory.usage_in_bytes"),
        )
        total = number("memory/memory.memsw.usage_in_bytes")
        swap = max(0, total - used) if total is not None and used is not None else None
    remaining = (
        max(0, limit - used) if limit is not None and limit < 2**60 and (used is not None) else None
    )
    return (remaining, swap)


class ContainerGuard(Guard):
    """Apply the existing reserve to both host RAM and the Notebook quota."""

    def __init__(self, cfg):
        super().__init__(cfg)
        _, self.start_container_swap = container_memory()

    def __call__(self, variables=0):
        super().__call__(variables)
        remaining, swap = container_memory()
        if remaining is not None:
            self.min_ram = min(self.min_ram, remaining)
            if (
                remaining - self.cfg["bytes_per_transport_variable"] * variables
                < self.cfg["minimum_ram_bytes"]
            ):
                raise RuntimeError(
                    "Full-target transport would violate the 25GB Notebook RAM reserve"
                )
        if swap is not None and self.start_container_swap is not None:
            if swap - self.start_container_swap > self.cfg["maximum_new_swap_bytes"]:
                raise RuntimeError("Notebook new swap exceeds the 2GB research limit")
