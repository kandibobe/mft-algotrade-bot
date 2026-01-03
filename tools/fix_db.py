import subprocess
import time
import sys
import os

def run_command(command, description):
    print(f"\n🚀 {description}...")
    print(f"Running: {' '.join(command)}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        if result.stdout:
            print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during '{description}':")
        print(e.stdout)
        print(e.stderr)
        return None

def main():
    print("🛠️ DATABASE FACTORY RESET & FIX")
    print("================================")

    # 1. Stop Containers
    run_command(["docker-compose", "down"], "Stopping containers")

    # 2. Identify Volume
    print("\n🔍 Identifying volume...")
    volumes_out = run_command(["docker", "volume", "ls"], "Listing volumes")
    
    target_volume = None
    if volumes_out:
        for line in volumes_out.splitlines():
            if "postgres_data" in line:
                # Docker volume ls output format: DRIVER VOLUME NAME
                parts = line.split()
                if len(parts) >= 2:
                    target_volume = parts[1]
                    break

    # 3. Destroy Volume
    if target_volume:
        run_command(["docker", "volume", "rm", target_volume], f"Destroying volume: {target_volume}")
    else:
        print("⚠️ Target volume not found in 'docker volume ls'. Attempting prune...")
        run_command(["docker", "volume", "prune", "--force"], "Pruning volumes")

    # 4. Restart DB
    run_command(["docker-compose", "up", "-d", "db"], "Restarting Database container")

    # 5. Wait for initialization
    print("\n⏳ Waiting 10 seconds for Postgres to initialize...")
    time.sleep(10)

    # 6. Verify
    run_command(["docker", "ps"], "Verifying container status")

    # 7. Re-seed Data
    run_command([sys.executable, "manage.py", "verify"], "Re-seeding simulation data")

    print("\n✅ Database fixed and re-seeded! You can now run the dashboard.")
    print("Run: streamlit run src/dashboard/app.py")

if __name__ == "__main__":
    main()
