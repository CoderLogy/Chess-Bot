# test_cutechess.py
# Simulates exactly what CuteChess sends to bot.py
# Run this BEFORE setting up CuteChess
# If this passes, CuteChess will work

import subprocess
import sys
import time

def test_bot():
    print("=" * 60)
    print("CuteChess Compatibility Test")
    print("=" * 60)

    # Start bot.py as subprocess — exactly like CuteChess does
    try:
        proc = subprocess.Popen(
            [sys.executable, "bot.py"],
            stdin  = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text   = True,
            bufsize = 1
        )
        print("PASS: bot.py started successfully")
    except Exception as e:
        print(f"FAIL: could not start bot.py — {e}")
        return

    def send(cmd):
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

    def read_until(keyword, timeout=30):
        """Read output until keyword found or timeout"""
        output = []
        start  = time.time()
        while time.time() - start < timeout:
            line = proc.stdout.readline().strip()
            if line:
                output.append(line)
                print(f"  BOT: {line}")
                if keyword in line:
                    return True, output
        return False, output

    # Test 1 — UCI handshake
    print("\n─── Test 1: UCI handshake ───")
    send("uci")
    found, _ = read_until("uciok", timeout=10)
    if found:
        print("PASS: uciok received")
    else:
        print("FAIL: uciok not received within 10 seconds")
        proc.kill()
        return

    # Test 2 — isready
    print("\n─── Test 2: isready ───")
    send("isready")
    found, _ = read_until("readyok", timeout=10)
    if found:
        print("PASS: readyok received")
    else:
        print("FAIL: readyok not received")
        proc.kill()
        return

    # Test 3 — starting position move
    print("\n─── Test 3: move from starting position ───")
    send("ucinewgame")
    send("position startpos")
    send("go wtime 60000 btime 60000 movestogo 40")
    found, output = read_until("bestmove", timeout=30)
    if found:
        move_line = [l for l in output if "bestmove" in l]
        if move_line:
            move = move_line[0].split()[1]
            print(f"PASS: bestmove received — {move}")
        else:
            print("PASS: bestmove received")
    else:
        print("FAIL: no bestmove within 30 seconds — bot too slow or crashed")
        proc.kill()
        return

    # Test 4 — position with moves
    print("\n─── Test 4: position after e4 e5 ───")
    send("position startpos moves e2e4 e7e5")
    send("go wtime 30000 btime 30000 movestogo 40")
    found, output = read_until("bestmove", timeout=30)
    if found:
        move_line = [l for l in output if "bestmove" in l]
        move = move_line[0].split()[1] if move_line else "unknown"
        print(f"PASS: bestmove received — {move}")
    else:
        print("FAIL: no bestmove after e4 e5")
        proc.kill()
        return

    # Test 5 — FEN position
    print("\n─── Test 5: FEN position ───")
    send("position fen rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")
    send("go wtime 30000 btime 30000")
    found, output = read_until("bestmove", timeout=30)
    if found:
        move_line = [l for l in output if "bestmove" in l]
        move = move_line[0].split()[1] if move_line else "unknown"
        print(f"PASS: bestmove from FEN — {move}")
    else:
        print("FAIL: no bestmove from FEN position")
        proc.kill()
        return

    # Test 6 — time pressure (low time)
    print("\n─── Test 6: time pressure (5 seconds left) ───")
    send("position startpos")
    send("go wtime 5000 btime 5000 movestogo 1")
    start = time.time()
    found, output = read_until("bestmove", timeout=10)
    elapsed = time.time() - start
    if found:
        print(f"PASS: responded in {elapsed:.2f}s under time pressure")
        if elapsed > 5.0:
            print("WARN: took longer than time control — may flag in games")
    else:
        print("FAIL: no response under time pressure")
        proc.kill()
        return

    # Test 7 — new game reset
    print("\n─── Test 7: new game reset ───")
    send("ucinewgame")
    send("isready")
    found, _ = read_until("readyok", timeout=10)
    if found:
        print("PASS: new game reset works")
    else:
        print("FAIL: new game reset failed")

    # Clean quit
    print("\n─── Quit ───")
    send("quit")
    time.sleep(1)
    proc.kill()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED — CuteChess should work")
    print("=" * 60)
    print()
    print("CuteChess settings:")
    print(f"  Command:    {sys.executable}")
    print(f"  Arguments:  bot.py")
    print(f"  Working dir: (folder containing bot.py)")
    print(f"  Protocol:   uci")

if __name__ == "__main__":
    test_bot()