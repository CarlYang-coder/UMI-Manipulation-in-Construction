"""
Record3D WiFi Streaming Probe Script
Tests WebRTC connection and dumps all available data to check for camera pose.

Usage:
    1. On iPhone: Open Record3D -> press red button (WiFi streaming)
    2. Note the IP address shown (e.g. 10.207.112.39)
    3. Run: python test_record3d_wifi.py --ip 10.207.112.39

This script will:
    - Fetch metadata (intrinsics)
    - Establish WebRTC connection
    - Print ALL data channel messages (looking for camera pose)
    - Print video frame info (resolution, format)
    - Run for 15 seconds then exit
"""

import argparse
import asyncio
import json
import time

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription
from av import VideoFrame


async def run_probe(ip: str, duration: float = 15.0):
    base_url = f"http://{ip}"
    frame_count = 0
    data_messages = []
    first_frame_info_printed = False

    # --- Step 1: Fetch metadata ---
    print(f"\n{'='*60}")
    print(f"Record3D WiFi Probe — {base_url}")
    print(f"{'='*60}\n")

    async with aiohttp.ClientSession() as session:
        # Metadata
        print("[1] Fetching /metadata ...")
        try:
            async with session.get(f"{base_url}/metadata") as resp:
                if resp.status == 200:
                    metadata = await resp.json()
                    print(f"    Status: {resp.status}")
                    print(f"    Response: {json.dumps(metadata, indent=2)}")
                else:
                    text = await resp.text()
                    print(f"    Status: {resp.status}")
                    print(f"    Response: {text[:500]}")
        except Exception as e:
            print(f"    Error: {e}")

        # --- Step 2: Get WebRTC offer ---
        print(f"\n[2] Fetching /getOffer ...")
        try:
            async with session.get(f"{base_url}/getOffer") as resp:
                if resp.status == 200:
                    offer_data = await resp.json()
                    print(f"    Status: {resp.status}")
                    sdp = offer_data.get("sdp", offer_data.get("data", ""))
                    sdp_type = offer_data.get("type", "offer")

                    # Parse SDP for media lines
                    sdp_lines = sdp.split("\n")
                    media_lines = [l for l in sdp_lines if l.startswith("m=")]
                    print(f"    SDP type: {sdp_type}")
                    print(f"    Media lines in SDP:")
                    for ml in media_lines:
                        print(f"      {ml.strip()}")

                    # Check for data channel in SDP
                    has_data_channel = any("application" in l for l in media_lines)
                    print(f"    Has data channel in SDP: {has_data_channel}")
                else:
                    print(f"    Status: {resp.status} (403 = another client connected)")
                    print(f"    {await resp.text()}")
                    return
        except Exception as e:
            print(f"    Error: {e}")
            return

        # --- Step 3: Establish WebRTC connection ---
        print(f"\n[3] Establishing WebRTC connection ...")
        pc = RTCPeerConnection()

        @pc.on("track")
        def on_track(track):
            print(f"    Received track: kind={track.kind}, id={track.id}")
            if track.kind == "video":
                asyncio.ensure_future(receive_video(track))

        @pc.on("datachannel")
        def on_datachannel(channel):
            print(f"\n    *** DATA CHANNEL OPENED: label={channel.label}, id={channel.id} ***")

            @channel.on("message")
            def on_message(message):
                ts = time.time()
                if isinstance(message, bytes):
                    data_messages.append(("bytes", len(message), message[:200]))
                    print(f"    [DC {ts:.3f}] bytes({len(message)}): {message[:100]}")
                else:
                    data_messages.append(("str", len(message), message[:200]))
                    print(f"    [DC {ts:.3f}] str({len(message)}): {message[:200]}")

            @channel.on("close")
            def on_close():
                print(f"    *** DATA CHANNEL CLOSED: {channel.label} ***")

        @pc.on("connectionstatechange")
        async def on_state_change():
            print(f"    Connection state: {pc.connectionState}")

        async def receive_video(track):
            nonlocal frame_count, first_frame_info_printed
            while True:
                try:
                    frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                    frame_count += 1

                    if not first_frame_info_printed:
                        # Convert to numpy for inspection
                        img = frame.to_ndarray(format="bgr24")
                        h, w = img.shape[:2]
                        print(f"\n[4] First video frame received:")
                        print(f"    Resolution: {w} x {h}")
                        print(f"    Format: {frame.format.name}")
                        print(f"    Left half (depth): {w//2} x {h}")
                        print(f"    Right half (RGB):  {w//2} x {h}")

                        # Check pixel values
                        left = img[:, :w//2]
                        right = img[:, w//2:]
                        print(f"    Left half  — mean pixel: {left.mean():.1f}, range: [{left.min()}, {left.max()}]")
                        print(f"    Right half — mean pixel: {right.mean():.1f}, range: [{right.min()}, {right.max()}]")
                        first_frame_info_printed = True

                    if frame_count % 30 == 0:
                        print(f"    Frames received: {frame_count}")

                except asyncio.TimeoutError:
                    print("    Video track timeout (5s no frames)")
                    break
                except Exception as e:
                    print(f"    Video track error: {e}")
                    break

        # Set remote description (offer from Record3D)
        # Record3D returns {"type": "offer", "sdp": "..."} format
        offer_sdp = offer_data.get("sdp", offer_data.get("data", ""))
        offer_type = offer_data.get("type", "offer")
        print(f"    SDP keys in response: {list(offer_data.keys())}")
        offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
        await pc.setRemoteDescription(offer)

        # Create and set answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # Wait for ICE gathering
        while pc.iceGatheringState != "complete":
            await asyncio.sleep(0.1)

        # Send answer (endpoint is /answer, not /sendAnswer)
        print(f"\n[3b] Sending answer to /answer ...")
        answer_payload = {
            "type": "answer",
            "data": pc.localDescription.sdp
        }
        try:
            async with session.post(
                f"{base_url}/answer",
                json=answer_payload,
                headers={"Content-Type": "application/json"}
            ) as resp:
                print(f"    Status: {resp.status}")
                text = await resp.text()
                if text:
                    print(f"    Response: {text[:200]}")
        except Exception as e:
            print(f"    Error sending answer: {e}")
            return

        # --- Step 4: Listen for data ---
        print(f"\n[5] Listening for {duration}s ...")
        print(f"    (Watching for data channel messages with camera pose)\n")

        await asyncio.sleep(duration)

        # --- Summary ---
        print(f"\n{'='*60}")
        print(f"PROBE SUMMARY")
        print(f"{'='*60}")
        print(f"  Video frames received: {frame_count}")
        print(f"  Data channel messages: {len(data_messages)}")
        if data_messages:
            print(f"\n  Data channel message samples:")
            for i, (dtype, length, content) in enumerate(data_messages[:10]):
                print(f"    [{i}] type={dtype}, len={length}")
                print(f"        content: {content}")
        else:
            print(f"\n  *** NO DATA CHANNEL MESSAGES RECEIVED ***")
            print(f"  Camera pose is NOT available via WiFi streaming.")
        print(f"{'='*60}\n")

        await pc.close()


def main():
    parser = argparse.ArgumentParser(description="Record3D WiFi Streaming Probe")
    parser.add_argument("--ip", type=str, required=True,
                        help="iPhone IP address (e.g. 10.207.112.39)")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="How long to listen (seconds, default: 15)")
    args = parser.parse_args()

    asyncio.run(run_probe(args.ip, args.duration))


if __name__ == "__main__":
    main()
