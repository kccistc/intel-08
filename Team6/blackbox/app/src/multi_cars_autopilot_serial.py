#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, glob, time, math, argparse, random, struct

# ===== Optional: Serial =====
try:
    import serial
except Exception:
    serial = None

# ===== CARLA egg path (if needed) =====
try:
    sys.path.append(
        glob.glob(os.path.join('..', 'carla', 'dist',
            f"carla-*{sys.version_info.major}.{sys.version_info.minor}-linux-x86_64.egg"))[0]
    )
except Exception:
    pass

import carla

# ===== Protocol (hero only) =====
STX0, STX1 = 0xAA, 0x55
MSG_VEH_STATUS = 0x01  # <HhBBBB> (speed0.1, steer0.1, thr, brk, gear, flags)

def crc8_xor(bs: bytes) -> int:
    c = 0
    for b in bs: c ^= b
    return c & 0xFF

def pack_payload(speed01: int, steer01: int, thr: int, brk: int, gear: int, flags: int) -> bytes:
    return struct.pack('<HhBBBB', speed01, steer01, thr, brk, gear, flags)

def build_frame(payload: bytes) -> bytes:
    head = bytes([STX0, STX1, MSG_VEH_STATUS, len(payload)])
    return head + payload + bytes([crc8_xor(head + payload)])

def kph(v: carla.Vector3D) -> float:
    return 3.6 * math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

# ---------- Spawn helpers ----------
def spawn_hero(world: carla.World, bp_filter='vehicle.*model3*', color='255,0,0'):
    bp_lib = world.get_blueprint_library()
    cand = bp_lib.filter(bp_filter)
    bp = cand[0] if cand else bp_lib.filter('vehicle.*')[0]
    bp.set_attribute('role_name', 'hero')
    if bp.has_attribute('color'):
        bp.set_attribute('color', color)
    for sp in world.get_map().get_spawn_points():
        a = world.try_spawn_actor(bp, sp)
        if a: return a
    return None

def spawn_traffic(world: carla.World, n: int, seed: int = 42):
    if n <= 0: return []
    random.seed(seed)
    bp_lib = world.get_blueprint_library()
    veh_bps = bp_lib.filter('vehicle.*')
    sps = world.get_map().get_spawn_points()
    random.shuffle(sps)
    spawned = []
    for sp in sps:
        if len(spawned) >= n: break
        bp = random.choice(veh_bps)
        if bp.has_attribute('driver_id'):
            bp.set_attribute('driver_id', random.choice(bp.get_attribute('driver_id').recommended_values))
        a = world.try_spawn_actor(bp, sp)
        if a: spawned.append(a)
    return spawned

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Spawn hero + (optional) traffic; send ONLY hero state over serial.")
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--town', default='', help='e.g., Town03 (empty = keep current)')
    ap.add_argument('--hz', type=float, default=20.0)

    # Hero / traffic
    ap.add_argument('--hero-filter', default='vehicle.*model3*')
    ap.add_argument('--hero-color', default='255,0,0')
    ap.add_argument('--hero-slowdown', type=float, default=75.0, help='클수록 더 느림 (75 → 제한속도의 25%)')
    ap.add_argument('--num-traffic', type=int, default=0, help='배경차 대수(히어로 제외)')
    ap.add_argument('--traffic-slowdown', type=float, default=0.0)
    ap.add_argument('--destroy-others', action='store_true')
    ap.add_argument('--follow', action='store_true', help='간단 3인칭 추적 카메라')

    # Serial
    ap.add_argument('--serial', default='', help='/dev/ttyACM0 (비우면 전송X)')
    ap.add_argument('--baud', type=int, default=115200)
    args = ap.parse_args()

    # Serial open
    ser = None
    if args.serial:
        if serial is None:
            print("[WARN] pyserial not installed; serial disabled.")
        else:
            try:
                ser = serial.Serial(args.serial, args.baud, timeout=0.02)
                # UNO auto-reset settle
                try:
                    ser.dtr = False; time.sleep(0.2)
                    ser.reset_input_buffer(); ser.reset_output_buffer()
                    ser.dtr = True
                except Exception: pass
                time.sleep(2.0)
                print(f"[INFO] Serial open: {args.serial} @ {args.baud}")
            except Exception as e:
                print(f"[WARN] Serial open failed: {e}; continue without serial")
                ser = None

    # CARLA
    client = carla.Client(args.host, args.port); client.set_timeout(10.0)
    world = client.load_world(args.town) if args.town else client.get_world()
    tm = client.get_trafficmanager()

    original = world.get_settings()
    new = world.get_settings()
    new.synchronous_mode = True
    new.fixed_delta_seconds = 1.0 / args.hz
    world.apply_settings(new)
    tm.set_synchronous_mode(True)
    world.tick()

    # Clean existing
    if args.destroy_others:
        for a in world.get_actors().filter('vehicle.*'):
            try: a.destroy()
            except: pass
        world.tick()

    # Spawn hero
    hero = spawn_hero(world, args.hero_filter, args.hero_color)
    if hero is None:
        vs = world.get_actors().filter('vehicle.*')
        if vs:
            hero = vs[0]
            print(f"[WARN] Hero spawn failed → using existing vehicle id={hero.id}")
        else:
            print("[ERR] No vehicle available.")
            tm.set_synchronous_mode(False); world.apply_settings(original)
            if ser: ser.close()
            return

    # Spawn traffic
    traffic = spawn_traffic(world, args.num_traffic)
    world.tick()

    # TM setup
    tm.ignore_lights_percentage(hero, 0)
    tm.ignore_signs_percentage(hero, 0)
    tm.vehicle_percentage_speed_difference(hero, args.hero_slowdown)
    hero.set_autopilot(True, tm.get_port())

    for v in traffic:
        try:
            tm.ignore_lights_percentage(v, 0)
            tm.ignore_signs_percentage(v, 0)
            if args.traffic_slowdown > 0:
                tm.vehicle_percentage_speed_difference(v, args.traffic_slowdown)
            v.set_autopilot(True, tm.get_port())
        except: pass

    # Visual identify hero
    try:
        hero.set_light_state(hero.get_light_state() |
                             carla.VehicleLightState.LeftBlinker |
                             carla.VehicleLightState.RightBlinker)
    except: pass

    print(f"[INFO] HERO id={hero.id}, type={hero.type_id}, role={hero.attributes.get('role_name','')}")
    print(f"[INFO] traffic={len(traffic)}  hz={args.hz}  hero_slowdown={args.hero_slowdown}%")

    try:
        while True:
            world.tick()

            # follow camera (simple)
            if args.follow:
                tf = hero.get_transform()
                yaw = tf.rotation.yaw
                rad = math.radians(yaw)
                back = carla.Location(x=-8.0*math.cos(rad), y=-8.0*math.sin(rad), z=3.0)
                world.get_spectator().set_transform(
                    carla.Transform(tf.location + back, carla.Rotation(pitch=-10.0, yaw=yaw))
                )

            # hero state
            tf = hero.get_transform()
            vel = hero.get_velocity()
            ctrl = hero.get_control()

            spd = kph(vel)
            steer_deg = ctrl.steer * 30.0
            thr = int(max(0.0, min(1.0, ctrl.throttle)) * 100)
            brk = int(max(0.0, min(1.0, ctrl.brake)) * 100)
            gear_map = {0:0, 1:1, -1:2}
            gear = gear_map.get(ctrl.gear, 0)
            flags = 1 if getattr(hero, 'is_autopilot_enabled', False) else 0

            # print terminal (optional)
            print("HERO POS(%.2f, %.2f, %.2f) Yaw=%6.2f | SPD=%6.2f kph  STR=%6.2f deg  THR=%3d%%  BRK=%3d%%  GEAR=%d  FLG=0x%02X"
                  % (tf.location.x, tf.location.y, tf.location.z,
                     tf.rotation.yaw, spd, steer_deg, thr, brk, gear, flags))

            # serial send (HERO ONLY)
            if ser:
                speed01 = int(round(spd * 10))
                steer01 = int(round(steer_deg * 10))
                payload = pack_payload(speed01, steer01, thr, brk, gear, flags)
                frame = build_frame(payload)
                n = ser.write(frame)
                ser.flush()
                # simple TX log
                # print(f"[TX] {n}B -> {args.serial} (SPD0.1={speed01}, STR0.1={steer01}, THR={thr}, BRK={brk}, G={gear}, F=0x{flags:02X})")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        try: hero.set_autopilot(False)
        except: pass
        for v in traffic:
            try: v.set_autopilot(False)
            except: pass
        try: hero.destroy()
        except: pass
        for v in traffic:
            try: v.destroy()
            except: pass
        if ser: ser.close()
        try: tm.set_synchronous_mode(False)
        except: pass
        world.apply_settings(original)
        print("[INFO] Cleaned up.")

if __name__ == '__main__':
    main()
