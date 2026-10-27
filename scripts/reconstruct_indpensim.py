"""Reconstruct IndPenSim CSV headers and batch IDs using streaming I/O."""

import csv
from pathlib import Path

SOURCE = Path("data/raw/100_Batches_IndPenSim_V3.csv")
DEST = Path("data/raw/100_Batches_IndPenSim_V3_reconstructed.csv")


def main():
    row_count = 0
    batch_id = 1
    previous_time = None

    with SOURCE.open("r", newline="") as source, \
         DEST.open("w", newline="") as destination:

        reader = csv.reader(source)
        writer = csv.writer(destination)

        header = next(reader)

        if len(header) != 2239:
            raise ValueError(f"Unexpected header length: {len(header)}")

        corrected_header = header[:37] + header[39:] + ["Batch ID"]
        writer.writerow(corrected_header)

        for row in reader:
            if len(row) != 2237:
                raise ValueError(
                    f"Unexpected field count at row {row_count + 1}"
                )

            current_time = float(row[0])

            if previous_time is not None and current_time < previous_time:
                batch_id += 1

            writer.writerow(row + [batch_id])

            previous_time = current_time
            row_count += 1

            if row_count % 20000 == 0:
                print(f"Processed {row_count:,} rows...", flush=True)

    print(f"Saved: {DEST}")
    print(f"Rows: {row_count:,}")
    print(f"Batches: {batch_id}")
    print(f"Columns: {len(corrected_header)}")


if __name__ == "__main__":
    main()
