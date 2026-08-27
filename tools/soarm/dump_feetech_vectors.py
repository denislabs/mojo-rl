#!/usr/bin/env python
"""Print the reference SDK's own instruction packets, as Mojo test vectors.

`tests/robot/test_feetech_packet.mojo` asserts our codec reproduces these byte
for byte. Regenerate after any scservo_sdk bump:

    /path/to/lerobot-env/bin/python tools/soarm/dump_feetech_vectors.py

Writes NOTHING to any port -- `FakePort` records the bytes instead of sending
them, so this is safe with the arms plugged in, unplugged or absent.
"""

import scservo_sdk as scs


class FakePort:
    def __init__(self):
        self.tx = b""
        self.is_using = False

    def clearPort(self):  # noqa: N802
        pass

    def writePort(self, p):  # noqa: N802
        self.tx = bytes(p)
        return len(p)

    def setPacketTimeout(self, n):  # noqa: N802
        pass

    def readPort(self, n):  # noqa: N802
        return b""

    def isPacketTimeout(self):  # noqa: N802
        return True

    def getBytesAvailable(self):  # noqa: N802
        return 0


def cap(fn):
    p, ph = FakePort(), scs.PacketHandler(0)
    try:
        fn(ph, p)
    except Exception:  # the fake port never answers; we only want the tx
        pass
    return p.tx


def hexs(b):
    return ", ".join(f"0x{x:02X}" for x in b)


def sync_read(ph, p):
    g = scs.GroupSyncRead(p, ph, 56, 2)
    for i in range(1, 7):
        g.addParam(i)
    g.txPacket()


def sync_write(ph, p):
    g = scs.GroupSyncWrite(p, ph, 42, 2)
    for i, v in zip(range(1, 7), [100, 200, 300, 400, 500, 600]):
        g.addParam(i, [scs.SCS_LOBYTE(v), scs.SCS_HIBYTE(v)])
    g.txPacket()


CASES = [
    ("ping id=1", lambda ph, p: ph.ping(p, 1)),
    ("read id=1 addr=56 len=2", lambda ph, p: ph.readTxRx(p, 1, 56, 2)),
    ("write id=3 addr=42 val=2048 2B", lambda ph, p: ph.write2ByteTxRx(p, 3, 42, 2048)),
    ("write id=2 addr=40 val=1 1B", lambda ph, p: ph.write1ByteTxRx(p, 2, 40, 1)),
    ("sync_read addr=56 size=2 ids=1..6", sync_read),
    ("sync_write addr=42 size=2 ids=1..6", sync_write),
]

if __name__ == "__main__":
    for name, fn in CASES:
        print(f"# {name}\n{hexs(cap(fn))}\n")
