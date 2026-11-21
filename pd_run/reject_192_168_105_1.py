#!/usr/bin/env python3
"""Reject one source address and splice every other TCP connection upstream."""

import argparse
import asyncio
import logging

LOG = logging.getLogger("reject-105")


def peer_ip(peer) -> str:
    if not peer:
        return ""
    host = peer[0]
    if host.startswith("::ffff:"):
        return host[7:]
    return host


async def splice(reader, writer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError, BrokenPipeError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle(deny: str, upstream: tuple[str, int], reader, writer):
    peer = writer.get_extra_info("peername")
    ip = peer_ip(peer)
    if ip == deny:
        LOG.info("reject %s", ip)
        try:
            writer.write(
                b"HTTP/1.1 403 Forbidden\r\n"
                b"Content-Length: 39\r\n"
                b"Connection: close\r\n"
                b"\r\n"
                b"rejected source 192.168.105.1\n"
            )
            await writer.drain()
        except (ConnectionError, BrokenPipeError):
            pass
        writer.close()
        return
    try:
        ureader, uwriter = await asyncio.open_connection(*upstream)
    except OSError as exc:
        LOG.warning("upstream down for %s: %s", ip, exc)
        writer.close()
        return
    await asyncio.gather(
        splice(reader, uwriter),
        splice(ureader, writer),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="0.0.0.0:8000")
    parser.add_argument("--upstream", default="127.0.0.1:8001")
    parser.add_argument("--deny", default="192.168.105.1")
    args = parser.parse_args()
    host, port = args.listen.rsplit(":", 1)
    uhost, uport = args.upstream.rsplit(":", 1)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    async def run():
        server = await asyncio.start_server(
            lambda r, w: handle(args.deny, (uhost, int(uport)), r, w),
            host,
            int(port),
        )
        LOG.info("listen %s reject %s -> %s", args.listen, args.deny, args.upstream)
        async with server:
            await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
