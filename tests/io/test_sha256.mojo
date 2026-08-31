# +--------------------------------------------------------------------------+ #
# | SHA-256 vs an independent implementation
# +--------------------------------------------------------------------------+ #
"""Gate `mojo_rl/io/sha256.mojo` against digests computed by Python `hashlib`.

    pixi run mojo run -I . tests/io/test_sha256.mojo

⚠ THE REFERENCE IS EMBEDDED, NOT RECOMPUTED. A gate that calls the same code
twice proves nothing, and a gate that shells out to `hashlib` at run time
would silently pass on a box where the shell-out failed. These 130 digests
came out of `hashlib.sha256` and are pinned here as literals: disagreeing with
them is a defect in this repo, by construction.

## What the 130 lengths buy

⚠ PADDING IS WHERE SHA-256 IMPLEMENTATIONS BREAK, and it breaks on LENGTH,
not on content. A message whose length mod 64 lands in [56, 63] cannot fit the
0x80 byte plus the 8-byte bit count in its final block, so it needs a SECOND
padding block. An implementation missing that case agrees on "abc" and on
every short string anyone tries by hand, and disagrees on real files. Lengths
0..129 cover every residue mod 64 twice, both boundaries (55/56, 119/120) and
the exact-block cases (0, 64, 128).

Also gated: the streaming split. The same bytes fed in 1-, 7- and 64-byte
chunks must give one digest — that is what makes `sha256_file`'s 8 MiB chunked
read legitimate.
"""

from mojo_rl.io.sha256 import Sha256, sha256_hex, sha256_string, sha256_file
from mojo_rl.io.fileio import write_file_atomic


comptime _N_CASES = 130

comptime _REF: InlineArray[StaticString, _N_CASES] = [
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "084fed08b978af4d7d196a7446a86b58009e636b611db16211b65a9aadff29c5",
    "323b730f87b4e7cc0948351a1c11b757b3026cda6784282576757bca21f12483", "6ab0dba1f4f1dfbb37b4f9eeb092c09fca4900ad32bdcd147d8dde35d6c87c35",
    "42a146d9caf95c0d29b3ea8e3574f3c47758bec2cdcb99e6f10381de77ab6d54", "c0a7188b4e87d64b5ff6dbedc69629b41ded38b08f0f79b85c5b63ed4a6b4646",
    "f3a8eb3d8bbb31db309b41f071ebabd46006c252a78b5d04dbc86b47cf8ed918", "ab764db46a4b504f034619a51475e900366a04738693b13a742845930daf74e4",
    "0865c1be255b33b69c4c1b7df3646cd2b7feab36f595044ab191176f1668d9ad", "172f3d817c5cccf034df5292b9dbc5490622eb8c21f8a342e695ece40044c4a7",
    "aadd73eb67f4e48bdb358638d0c42f341afcf9f60d717418d863a6f69238e01f", "74b55c9de0be1e88daf5efc4838719c5b96f707bb47f99f5f4a30d7269b0838d",
    "756a339579953083c882d4fd4249fc97f82a510069b3ef7384cd7c76dc1d46f8", "da92597518961b4c9909e090fcb1e11200e76b57a936ad0c54b2c431f9058ec3",
    "a6a744ad474a0ed0df8977a42cc288156c70bbff998472596c6f6dfdf591c684", "98b03249d75e642ef41f16fc71486ea85b551fc8c90324be773d3967852f790c",
    "9c94926dfb94433e790f2c209e2633b2dd3e922b2741ac687e164d488d1ff67c", "bbc485bd3e9865564c1d1fdf5cccf969c6435d86eda9256acf9bba7f5dd69eb7",
    "4929dcd438730a2d11486af19aca9d2c9a6187246f6477f54dbae76588fa7eaa", "e22bd9f643244050a9a8a809ff876f0319d603d01cac562fa369025e547c86ea",
    "cb0b638f9fd1fd3d3a5310ef9160d16a8a50e30b8ff1bbeba11897246ebc3275", "dd59e815a7a25db12b7764ca6a3c3c7589b5eb3cffe51b9532526832245e12a0",
    "7f5e1b7371adc30c9da908a8c45b863126839b9b92f6655ea1e30106b9434695", "0c2227ce84dedb8c713405705af85dd9b9782a6bb9f2d4de91273e198e64d509",
    "73281455a5e6186744d4ddbb1d5a10958c9d8e34f150c9f8912eaa4b8f3996ad", "fa9766ea344626dd9936ebfc2790476cf5f2d8c4c75e2a3d33c052cd20221f72",
    "5492dff8d285d13c0a037f75a4796474a5d2da0cf3cfbba82e9eec74f415d884", "8876c6114debb0978f5ffbbdd794f616d1cd0699d5934f37c94fe1bc1161a93f",
    "a6940de8ec80c86afee42391dcdd7d97b32c72ce8839775b1a2518c1487eb197", "52bbbbd714150af8ea8c40af59263ee53f6a161744800e5e5ff6d61a47702296",
    "f1eab075947c8bcf6f1c8d5dbac66efe80756dd32ccb31c6f527a36c09889cae", "721f1b54dd0f746ef819ba479fcd63b6117bfbe44acc432bf0cb0a639fd41766",
    "ab5f8b5cb9435354c7b58603592d5faf081e17ceb05f7a7c67f4b666f12ca457", "90f90857a7d1aad4925c6d5edbda0abfc8270e3fe5d6da2dd5eae75009cd7609",
    "c296690862799313d4df91200f74bfa61b3853ac4a8973a7417b8cf1ae7ab260", "028250999c5186460b292bc5533202c55561b60da19f0c034695a60b4c2ac5fe",
    "a7aba4ca5c33084110dd834aed55b0854a3b01cda64777e490ac02e7e7b7446d", "b11e919ce284b7e028be56562412b1eda22aeabd890995754fa3a9847c69a20c",
    "afa3a9b2b3c085cf8080ad146188b68dc850ce9b8b0627548e0f39ddba638a76", "90663c41a474e299b9eca62670670b088c213759b935c2d68319fa78d0099df4",
    "0873681bd0f82f74733bd4b4639467130c6ff71a09281210ed60c3dc95d6aa90", "fe877131379e5bbff19e5b14abafe41a724a6fe36e70bc3e2d0c6267da4dd52d",
    "d9fcbc5f1a9f3a7d79ef20f215fd96f7c069b30d37b42e8bbda878cbeffc6afb", "6c980fe8293c22ff9ced813a37e4619e5cd7f35c141082a49536a7f5d79f2790",
    "f990f6ed8d1cbc3a5d64bb9068493cb734c06cc91a6d755d2e8f938fa02d7a52", "98ce1f7bd0cb468ff2efc7e1cace2d2e56c4b61f7376e98194a6139579608513",
    "736b0a4e151af6d6f514103ecf7f038e23cbfa64d8f3d7387e238b6d7253b308", "9b6073af8d49f2fef313e24f9005e857d4726b01543bfe641adbb7a4f33c2211",
    "31cedec8e83dc0fb13e8ba27dfd62dd11aefa1923d78bfbade0eb4f339636144", "5649348a03738691f8dc40791868b8847833dee22142f777bf7c2a5091e262e6",
    "9898428b82ee6f679753036472bedb74701161801f29526e7201e0dcc600bcdc", "75e248b3da4cae96bfa15e251859d860c3bd8a137352771729d4dc45c6dac025",
    "8230c270cd48aa1a4303308ce02ea63eaffb2480124f34b20726806556522e1f", "9f17a1b3d8f9affdc77869490f25a03b2450c66409ec4b106ad7112cc66d2d36",
    "160bbf14b458c877b7049e7cb5771dd653930f97d20bdd8ee795c16062906233", "e7313d333c272e639f790978283f9eb392e843d0f29b7016828bb1daa4aac70b",
    "4324d65f3c103567f5589c710bc08f8523f929a9272e3af36fc968e52abc6c27", "35df609437dcfea3279283ab79fd554e2bf78f8f7ae2de532d8ee300b09e8f73",
    "9afd9e8bcdb57c7c3a445ee45dd8050df9e187efe7f35506dbbfd91262292f08", "3f67cd873703688f9cfe32074b65caec62498407c9556c91368f4575383c25ae",
    "06659a8b0876d0ea2a601ae653912d113996bcfcd772b262e4bd866984d3bfb3", "4a28a6c40a8e8306eda3f334638ec1028415607a8c6989d48b864ac79106b5f3",
    "5b50d7a374c35bf46f501a178cceb2cfe649cd8f96c4a21d57377fcbb0eec4ab", "81c80242132f230c3bd41b3e63bbcff16107339549214a99614ff26664625055",
    "39e3d7b6b5d075d37d053ad89b24b41bef4f3c29760c84447cab3f3be1882241", "aacca6ff74fdbb296d165a45cecfa04e5127bc008770fbbdd48006f2d2fae95e",
    "11fdf00350687cdd9dc4312de9734ad3ddf5eeb4d0c442ad34b05f3e484e8ffd", "06e3cd2ed0fa6071c44dc044441f1d16e92b7505ce10787230dc895fcc6622b2",
    "c23b56ccb21ebf78c5401dc6257f22e536b6d9b36214a68c899706f7cfc14e83", "ca32376f1ceca93d6bc43b8fcd47e6ec7b20dbac0ee3f62f4a0c4dad838bb7b7",
    "fb1907e541f9c81501e95cc95fbfacb263bab1990ae9b54272673d5107511d52", "54cac60524e8d20657ff88ee3951e22a506f3d8dc22c8fa848ce494672cb0d29",
    "99e356eadffa802796fb42722352b16dfc8a61806bf11bdd3590d0f0bfdaef5d", "64bee9b0cccaaa16864211d3337214839db10309e7156e3c0e168748682fa260",
    "3ebc4be43f93cac2357d1a462636e79762b7026fb80db09f2790cc9461d22bc1", "08182253830d4d2b1fa510c5abe3af40e0ad81738a23469ad29b449b07b1e7c8",
    "65cf2b7beb0ba72fdb797686cce8857dab0caf5ee1a0c7c103c21e00e6051020", "1f21d7fe24283682a46a35d86d909513487ff5292634890454ed882a66b28f12",
    "87fa111ce4e78d2db52185af0e4376493d725de46c91d007d19e7dd533d0a17a", "2a4808f9dfc6e27024e647b2bfe62399827947143e8ca89a5c012f6d3d8429a2",
    "c6b3377d81c23312e11ac4a33e9e87b00ec1cb167b13c97d69657b8e3820a74e", "5876ebcb920052ecaa77c05f36aad54d33ae2f24722b777b3a2f33d17c2a7d4a",
    "eb52f62be6fe52e68abcdaea3cd0b55bd48221c9c4f9eb8f59ffda517e10ed78", "2f0795e2f484bd5abe26322a2c2e125d1e5d93e57a1b57cafbcf3c622f158ae0",
    "442975e01ac274365ba2c42b68e320fa3b46b9d28039143ff1bfa443a7571325", "b690f6f0cc86111bb0bcd93678c57711596e1869158856e9e9e36f8764261f27",
    "b876ad022f68deab857e9da23c8e1f1736935f9cec1bf2c9a51aefde2317e663", "ff4ec8cd0185eaaa7c4d7e99798430b932b89f524d0e73ae9d2efc1b99275bf3",
    "b16ffce5f3d16040ca4ed8416f31632c7497967eb22769d4bdc4a959f8029b16", "bb743aa18f1db05620a1480c257b3164fbeb4760f2b848145e404de72931f87d",
    "4d39e6aed7920542bd34fabeb7b083538825f6aa996b431eb92c640c80c5420a", "85373e64524dff047f5423100fa3b9328efed49f49bf9ba752aa1e01708137f8",
    "f8368d09ac1e1afdbcd737ff9e751099b9109daf3036fa49ed1ff1609025beeb", "4d0d2d32697292391dc0199b3eacb82c5ad0dc38a2770adb3ce8044f869d128f",
    "5028666bd79edd2bede28bdeee2d7a95c01dbbd4284a8c5b30a998ee8d7ed3e0", "3a23c9132a8635ad65b1055484a4d241b70ea94ed89c4d96e22ceec9daae992c",
    "c9f1a5f79d7bea01a54f4edb41673722f627ee2e82dda324946b63cf4b9b16af", "9b5d66321a0d92249c95aca5a045580afd9b9ddcdb4ada96a94bee84d2d99716",
    "af680818335f180a11ea361e316a6100ae6dd5c83066e51867b08cb6e8ba600b", "5e4edd8413b312a4d85297f6e89146bacf0d9714495a1e83dc9f63c0037f0d30",
    "5a2cda2351d1cdd9dd7957e57c0b3c8522451f25b6494569b7e94388c46f0980", "48719cf73125f924294a140de9d41efe9fdad8e520c55c541127ea709ef0554e",
    "9f0fc4c34f14e7a45bef6a016dc2f4fdb0c858d61247d13c6d41241b2011c686", "9313cf73a730ac07b74d681e6a73eec8a1593f4e3ae7557ddd77b44c980e0571",
    "5af877de0ea99d70bd0a547c967dbecd4525d6bba9ab2917d54d6aa742874687", "af7d44bcc0959aaadbb7e6fb4cf6845c186e6a1cd2041132020f6aa7ac57911b",
    "0e99348634f0dd8367d7f75bb4fbad8297971625a5424b6eb1ec7f0ac453ae43", "9dfada1b2877f5075474bb3596d89600053831161f4a2fe6ecb3fbea85826bc5",
    "e0cfd9f04dd5ddba567539bcd694ce47cff5ac057cb56b80127fcd3a604f2a7a", "6a9c24a9d2f86056d9f3989044d43ebf4ed1723cb4ee1660c818e01225f42621",
    "4a70b5e28bfd48b3786770f8bc1c3c1967c9d5bc01872993580e22646e94c2c6", "67d9492e628fd376e0b2efec8ca2b99b123e202cf620deb270728df979b2f73e",
    "96b928cff8528dbb99602c709a65b846cb6467acb8b722f0d758e4dc27bfc508", "cded73eed7df41eecd1ead825a38ca8763a9e910d5181c8ad8f201cc8ff2cbe4",
    "5155ce162330c6949d409b5885ee847afbf0bad1fbae8fcc61cf754ba4e247fa", "c2030f07255ace70eea17370a5af9fabfad29be0bbf6a3f56f195531e2d7e0a7",
    "556cbf3481676c7f963e52baf7dd49ea0eb300e74d2a85b8e2a691973b41a888", "867b32635b088696679312aa02f4aa845828dfe86f339c411824f9eceb2863f8",
    "2c49b89c5aaaaaa69ae70251cc0803f14afaf1a2caa461fa6b083750bc07ce88", "9ce7368e4daf32341631b492e80359dc9f594b48453cd0dd5bf0b19279cc177e",
    "7836b787757e95e58b3ca5aec90b1b004e8deba1e50e9675af9cabf1a13a04b5", "1189a98a00c71bc1848ea8bdc9700b442bee0be7c3f45172303f1ab0b6f1617e",
    "60937afbb66acb0675a2f516774b22b7e8de46bfe263ea3d8f5d4872627f6045", "229c92a5abffa5300cee76bdc526a40ef85e0479d6eb13c8d875befe8e2c267c",
    "2f781a36cc860b1020a4713b6657730a78042b2f240c07aac35a386030a0cdb1", "b402a31c521e6cad6a9bc39d4448234905d72f412eff7f761210eed6bac737d9",
    "3d11be9920ed2c431ed505b9c790bb4a17398bb1e0977f978efbd357c3b3fbd2", "a8d23e75d936f303d248888d9b165ee543f4cbafcad3c9dd2a79bd84faa11d07",
    "d2742f1f4ac6bb7ca2b239ee18402ba8b3f9f8e652d2a72973c2b9ba11c08cf6", "307f8fc2c1622b92762e818d39a185d4d667ad49a4b07ceae1f4afa008a93ec4",
]


def _message(n: Int) -> List[UInt8]:
    """The same generator the reference used: `(i * 7 + 3) & 0xFF`."""
    var out = List[UInt8]()
    for i in range(n):
        out.append(UInt8((i * 7 + 3) & 0xFF))
    return out^


def main() raises:
    print("=== sha256 vs hashlib ===")

    # ── the three FIPS 180-4 vectors, spelled out ────────────────────
    var v_empty = sha256_string(String(""))
    var v_abc = sha256_string(String("abc"))
    var v_448 = sha256_string(
        String("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
    )
    if v_empty != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855":
        raise Error("FIPS vector 1 (empty) mismatch: " + v_empty)
    if v_abc != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad":
        raise Error("FIPS vector 2 (abc) mismatch: " + v_abc)
    if v_448 != "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1":
        raise Error("FIPS vector 3 (448-bit) mismatch: " + v_448)
    print("  3/3 FIPS 180-4 vectors")

    # ── every length 0..129 ──────────────────────────────────────────
    var want = materialize[_REF]()
    var compared = 0
    var differing = 0
    for n in range(_N_CASES):
        var msg = _message(n)
        var got = sha256_hex(msg)
        compared += 1
        if got != String(want[n]):
            differing += 1
            print(
                "  MISMATCH at n=" + String(n) + ": got " + got + ", hashlib "
                + String(want[n])
            )
    # ⚠ Print what was COMPARED beside what DIFFERED. "0 mismatches" over an
    # empty loop is the failure mode this line exists to make visible.
    print("  " + String(compared) + " lengths compared, " + String(differing)
          + " differing")
    if compared != _N_CASES:
        raise Error("vacuous: only " + String(compared) + " cases ran")
    if differing != 0:
        raise Error(String(differing) + " digests disagree with hashlib")

    # ── the split must not matter ────────────────────────────────────
    var msg = _message(129)
    var whole = sha256_hex(msg)
    for chunk in [1, 7, 64, 63, 128]:
        var h = Sha256()
        var off = 0
        while off < len(msg):
            var take = min(chunk, len(msg) - off)
            var piece = List[UInt8]()
            for i in range(take):
                piece.append(msg[off + i])
            h.update(piece)
            off += take
        var got = h.hex()
        if got != whole:
            raise Error(
                "chunked at " + String(chunk) + " gave " + got + ", whole gave "
                + whole
            )
    print("  5/5 chunk splittings agree")

    # ── sha256_file reads the same bytes ─────────────────────────────
    var path = String("/tmp/mojo_rl_sha256_gate.bin")
    var big = _message(129)
    write_file_atomic(path, big)
    var from_file = sha256_file(path, 7)  # a chunk size that splits blocks
    if from_file != whole:
        raise Error("sha256_file disagreed with sha256_hex: " + from_file)
    print("  sha256_file agrees with the in-memory digest")

    print("[PASS] sha256")
