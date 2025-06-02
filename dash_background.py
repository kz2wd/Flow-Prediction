from dask.distributed import Client, LocalCluster
import threading
import signal
import sys

def main():
    # Create a long-lived LocalCluster with fixed ports
    cluster = LocalCluster(
        scheduler_port=8786,          # fixed scheduler port
        dashboard_address=":8787",    # fixed dashboard port
        silence_logs=False,
    )
    client = Client(cluster)

    print("=== Dask cluster is running ===")
    print(f"Scheduler: {client.scheduler_info()['address']}")
    print(f"Dashboard: {client.dashboard_link}")
    print("\nPress Ctrl+C to shut down...")

    # Use an Event to wait indefinitely in a clean way
    stop_event = threading.Event()
    try:
        stop_event.wait()  # blocks until stop_event is set
    except KeyboardInterrupt:
        print("\nShutting down Dask cluster...")
        client.close()
        cluster.close()
        sys.exit(0)

if __name__ == "__main__":
    main()
